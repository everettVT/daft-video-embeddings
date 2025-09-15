# Video Embeddings



# Notes

- Right form the get go, this workload is different from normal image or text embedding workloads.
- Video embeddings demand us to pass in a list of frames with a specific size and shape, normalized to be between 0 and 1. 
- Its not clear to me how to convert an image dtype to a tensor of [288,288,3]
- If I try to convert the image to a tensor with a new precision I get a core exception.

```python
df_norm = df_frames.with_column("data_f32_01", col("data").cast(dt.tensor(dt.float32())))
```

```text
DaftCoreException: DaftError::External task 131 panicked with message "StructArray::new received an array with dtype: List[UInt8] but expected child field: data#List[Float32]"
```

As a naive user, its not clear if the image format or mode is related to native compatibility to this normalized format. Theres just really no descriptions or details on what any of the image modes are for a naive user. The only reference I could find for what an image dtype is when passed to a UDF is from the [Querying Images](https://docs.getdaft.io/en/stable/examples/querying-images/#working-with-complex-data) example. So I ran with that at first, focusing just on the normalization.

This was my initial approach, where I normalized in one udf and then stacked in another following a groupby

```python 
@daft.func()
def normalize(image: np.ndarray) -> dt.tensor(dt.float32()):
    return np.asarray(image).astype(np.float32) / 255.0

df_norm = df_frames.with_column("image_tensor", normalize(col("data")))
```

This gets you a nice pretty column of image tensors to groupby and stack later, which are easy to understand and implement. This is great and all, but you eventually realize that, when you want to stack your frames you end up needing another UDF. Recognizing the performance tradeoff of running two udfs instead of one, you end up trying to just group the image dtype directly and performing normalization and stacking all in UDF.

This gives you a chance to tackle a couple problems at once. 
1. Normalization and Stacking (Main goals)
2. Deterministic frame ordering (sorted groupby not always guaranteed in distributed)
3. As well as handling edge cases like the tail end of a video that doesn't neatly splice into your chosen clip size. Here we just duplicate that frames until the end

Eventually you end up with something like this:

```python
@daft.func(return_dtype=dt.tensor(dt.float32(), shape=(1, 16,288, 288, 3)))
def stack_clip(frames: list[np.ndarray], indices: list[int], clip_size: int):
    """Stacks a list of frames into a single numpy array
    
    Args:
        frames: List[T] of (H,W,3) float32
        indices: List[T] of int

    Returns:
        (1,T,H,W,3) float32 in [0,1]

    In a parallel/distributed groupby, a pre-group sort isn’t guaranteed 
    to survive aggregation order; partitions can concatenate in 
    non-deterministic order. Additionally, the image dtype is natively a
    list[uint8], so we need to cast to float32 before normalizing from 
    [0,255] to [0,1].

    Steps:
    1. Aggregate both image_tensor and frame_index.
    2. Sort by frame_index inside the group-level UDF, then stack.
    3. Normalize and cast in one step.
    4. Add a batch dimension and return.

    """

    # Don't assume frames are sorted already: 
    order = np.argsort(np.asarray(indices))

    # Convert Daft Image to np.ndarray
    def to_np(x):
        if hasattr(x, "to_numpy"):
            return x.to_numpy()          # Daft Image -> np.ndarray (H,W,C) uint8
        return np.asarray(x)

    # Sort frames by frame_index
    frames_sorted = [to_np(frames[i]) for i in order]

    # Ensure Tails are padded with duplicates
    if len(order) < clip_size:
        frames_sorted.extend([frames_sorted[-1]] * (clip_size - len(order)))

    # Stack, Normalize, and Cast in one step
    x = np.stack(frames_sorted[:clip_size], axis=0).astype(np.float32) / 255.0 # (T,H,W,3) float32 in [0,1]
    
    return x[None, ...] # [1,T,H,W,C] where T=clip_size

df_clips = df_grouped.with_column("clip", stack_clip(df_grouped["data"], df_grouped["frame_index"]))
df_clips.show(3)
```

Here you cannot specify the return type in the funciton definition type hint. 

```python
@daft.func()
def stack_clip(frames: list[np.ndarray], indices: list[int], clip_size: int) -> np.ndarray:
    ...
```

Yields

```text
ValueError: Unrecognized Python type, cannot convert to Daft type: <class 'numpy.ndarray'>
```

and adding the entire daft datatype return type hint at the end just looks terrible, so I opt for defining in the udf itself.

Our new clips column sets us up to batch inference requests for higher throughput. It's pretty cool that the videoprism model already supports this, but we can stack our clips at inference time.

This introduces another complexity, padding clip batches. If our video isn't easily divisible by our B*T Batch size, then we will have perform compile two forward functions. This is a worthwhile effort however and addresses the main two inference cases.

- In an interesting turn of events, it doesn't really make sense to pass both text and video embeddings together with this model. Sending them seperately or together makes no difference on the output. The pipeline for a dataframe however is pretty different, so it's broken up. 

### Jax compatibility and configuration

You know considering the word jax doesn't even really appear on daft's documentation anywere, I was expecting to run into a lot more problems than I did.

I got inference to work on a jitted inference function on both cpu and gpu, boasting some impresive batching throughput by stacking clips up to 24 deep on an A100. 

I tried running it on the TPU, but it looks like the inference was trying to run all clips (the entire dataframe) at a time regardless of the batch size. I presume this is the caveat with how jax delegate compute to XLA.

Even if I limited the dataframe to just 8 clips my session would crash. 

2 rows works
4 clips works
8 clips work in 20 sec
12 clips in 20 sec
2 x 12 clips in 23 sec
120 / 12 clips in 48 sec
319 / 12 clips batch OOM in 148

```error
XlaRuntimeError: RESOURCE_EXHAUSTED: XLA:TPU compile permanent error. Ran out of memory in memory space hbm. Used 55.24G of 31.25G hbm. Exceeded hbm capacity by 23.99G.

Total hbm usage >= 55.49G:
    reserved        260.00M 
    program          55.24G 
    arguments            0B 

Output size 0B; shares 0B with arguments.

Program hbm requirement 55.24G:
    global          260.02M
    HLO temp         54.99G (100.0% utilization: Unpadded (54.70G) Padded (54.70G), 0.5% fragmentation (289.57M))
```


I was able to get my read_video_frames approach to work on