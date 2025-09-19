# Video Embeddings

Date: Aug 25, 2025
Author: Everett Kleven
Size: L
Persona: UDF Naive User
Notebooks:
- from_video_frames to video embeddings
- end to end video processing from file

Scripts: 
- 


## Purpose

Explore fricton points in processing video ai pipelines where it is prohibitive to load all frames into memory.


## Summary

Video processing is hard. My experience echoed similar pains to that of the [VideoType discussion](https://github.com/Eventual-Inc/Daft/discussions/5054), where `read_video_frames()` is convenient, but insufficient. For the naive use case of reading images to a row limit and generating video embeddings on 16 frame clips, I was able to get the happy path working within a few work sessions. Once I faced the prospect of video segmentation and leveraging seeking to concurrently read videos with daft.File things became overwhelming.

What makes video processing particularly complex isn't just memory management, but the number of early decisions an engineer has to commit to when designing their workload. While my particular workload of video embeddings is straightforward, if I were building the pipeline for a more specific downstream task, I may implement things very differently.

It can be overwhelming to consider the various permutations of video processing approaches, especially concerning ingestion and segmentation. Inference is where the problem becomes more concrete, but if you have multiple downstream AI/ML tasks with different batching requirements things can get hairy quickly. This leads us to wan't to canonicalize our preprocessing stages into a standard form that can then be repackaged and shaped downstream. 

### Ingestion

1. read_video_frames - which decodes video frames into images and stores them as rows against a frame index
2. probe_video_metadata() + read_video_file(...,hist,sbd,audio) - which probes for metadata as a "cheap" pass, enabling early content filtering, then opening the video file with enriched inputs for extracting image histograms, shot boundary flag, and audio frames. Naturally the audio reading can be broken out into a seperate function entirely, but I'm including it here for brevity.
3. probe_video_metadata() + seek_video_file(...,hist,sbd,audio) - same as above, except distribute reads reading each video file concurrently from pre-planned frame timestamps.

### Segmentation 





Each of these patterns approach handling segmentation, shotboundary 
Segmentation in particular presents the problem or chunking your video into semantic pieces. While most downstream ai/ml tasks require samples in clips, usually on the order of 16 frame batches, any operation that occurs outside the clip context requires an additional groupby/explode. 
Shot boundary detection and other video segmentation strategies incentivize early preprocessing during frame decoding at the file level. 
File seeking can help parallelize reads, early computations like histograms and chi-squared distance are more convenient prior to dataframe ingestion. 

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




# Parakeet

i'm now trying to get parakeet to run. I used yt-dlp to download a few youtube videos from the deep dive playlist and I've preprocessed the audio into a list of numpy arrays... but I am getting an error saying 

```python

```



```text
ValueError: Input `audio` is of type <class 'daft.series.Series'>. Only `str` (path to audio file), `np.ndarray`, and `torch.Tensor` are supported as input.

The above exception was the direct cause of the following exception:

UDFException                              Traceback (most recent call last)
/tmp/ipython-input-3135087487.py in <cell line: 0>()
----> 1 df_transcribed = df_audio.with_column("text", ParakeetTranscribeUDF(df_audio["audio"])).collect()

UDFException: User-defined function `<__main__.ParakeetTranscribeUDF object at 0x7f52bfdb48f0>` failed when executing on inputs:
  - audio (Tensor(Float32), length=2)
```



# Working with the File Object

My initial task working with `daft.File` was to read video file metadata. After attempting to disover the video duration or fps myself and running into multiple headaches with PyAv, I eventually gave up and asked GPT-5 which gave me a much longer answer than I was expecting. Apparently duration is not always something you can just read from the file header and you end up having to set a compute budget to calculate the duration from a stream. 

The Metadata extraction function is exhaustive and goes way beyond the requested 4 attributes mentioned in the VideoType discussion, but it can be easily broken down. 

I ran into my first friction point with daft.File when trying to pass the daft.File object as a path to the metadata probe function. At first, I tested to see if naively passing the daft.File itself would work, but I got a type error.

```zsh 
Cell In[9], line 44, in probe_video_header_with_pyav(path, probesize, analyzeduration_us)
     39 audio_stream = next((s for s in container.streams if s.type == "audio"), None)
     41 # General/container
     42 meta: Dict[str, Any] = {
     43     "path": path,
---> 44     "name": os.path.basename(path),
     45     "size_bytes": os.path.getsize(path),
     46     "format_name": getattr(container.format, "name", None),
     47     "format_long_name": getattr(container.format, "long_name", None),
     48     "tags": dict(container.metadata or {}),
     49     "bit_rate": getattr(container, "bit_rate", None),
     50     "duration_seconds": _duration_seconds(container, video_stream),
     51 }
     53 # Video
     54 if video_stream:

File <frozen posixpath>:142, in basename(p)

TypeError: expected str, bytes or os.PathLike object, not PathFile
```

This is a minor hurdle, and I didn't necessarily feel frustrated that it didn't work, so I tried passing in `str(daft.File)` and received a new error:

```zsh
Cell In[9], line 36, in probe_video_header_with_pyav(path, probesize, analyzeduration_us)
     31 # Open read-only with constrained probe budgets
     32 options = {
     33     "probesize": str(probesize),
     34     "analyzeduration": str(analyzeduration_us),
     35 }
---> 36 with av.open(path, mode="r", options=options, metadata_encoding="utf-8") as container:
     37     # Choose the first video stream if present
     38     video_stream = next((s for s in container.streams if s.type == "video"), None)
     39     audio_stream = next((s for s in container.streams if s.type == "audio"), None)

FileNotFoundError: [Errno 2] No such file or directory: 'File(file:///Users/everett/Movies/Running.mp4)'
```

This one was more confusing and frustrating because I wasn't sure how to just retrieve the original file path without just completely avoiding the `daft.File` class. Looking at the the class definition I see a `_from_path()` method but no `to_path` method. Additionally, I am personally unfamiliar with the *read, seek, tell* interface, so I had to look this up. It would be nice if there was a short description of what seek and tell means in the docstring of the methods.

Once I learned what seek and tell were, this metadata reading problem became a lot more exciting. I now have the prospect of just grabbing the first frame directly? For this I needed to do some exploration. I got quickly confused, after trying to retrieve a container stream with `File.tell()` or `File.seek()` and defaulted back to gpt to return our core metadata details, which ended up working! I could spend time trying to make the function more daft native, but I'll let you guys do that. 

COOL WE NOW HAVE CHEAP METADATA READING. Goal #1 of daft.DataType.video() accomplished! 

Now for the hard part. 

## Streaming

We can extract image and video frames easily enough by reading the entire file, but streaming in a dataframe context is less intuitive. 

Our goal is to read a batch of frames at a time for inference. Inference is almost always our bottleneck, so whatever we can pack into that job will define the rest of our pipeline. 

I'll start with a generator pattern to yield batches of frames and iterate on a small subset until I OOM.

I immediately ran into issues trying to yield values from a udf. In fact, the first time I tried it, the jupyter kernel crashed. I tried debugging the yield by replacing it with `return` but then I was running into issues with returning lists. I knew that was an issue I could solve, but there was something else bothering me. How would I capture results? The thought of concatenating datafrmes crossed my mind but that didn't sound like a viable approach.

I then spent some time investigating a *seek* based strategy where I built a list of lists of frame indices (list of clips) and would convert those clips to the time_base format needed to seek to the desired frame. I got pretty much to the end of the implementation where I'd grab clips of frames and return a stacked numpy array but this felt wrong too.

Lets say we proceeded with the seeking approach. Can we even open and stream a file more than once? 


... oh shit that worked. How do I know if it's correct though? ... hold on.. never mind that. So I can seek and read chunks of a video in parallel... Thats what I just did. Before I was just reading the whole video in series and materializing the result. 

If we are looking to keep the memory low, we will need fine tuned control over how rapidly we preprocess the videos. We don't want to just finish the job, we want to pipeline everything... I think this still works once its lazy, but I may need to change my row-wise udf to a batch udf to fully control batch size and concurrency. Pretty sure I'd have to do that. 

So here's a question, if we don't have to worry about files locking up, then I think this preprocessing step has atomized our memory. I don't need to worry about pipe load balancing yet, lets just see how this ends up processing data.

it would be good to return the full frame index with the numpy tensor, just to be sure. 

Man, this is getting a little more exciting, its going to feel natural to attach the preprocessing to the inference udf as a pipeline.

This also means I will probably want to abstract the audioframe and video frame seeking 

Friction Points in Video Processing Video processing fundamentally represents a different data processing paradigm due to a few specific things that almost all video processing pipelines do. I was able to get from daft.readvideoframes to video in prints with NumPy-stacked batches of clips of frames very quickly, within a few lines, a few transformations. The biggest headaches I ran into there were converting the image data type from its int8 representation and just having no documentation around it on how it is supposed to turn into an NDArray, a NumPy NDArray. But I was able to stack those frames into clips and then stack those clips into batches for inference, and then just whatever your compute can handle, you just load up the batch size, which for, you know, an A100, you could stack 24 clips of 16 frames each and get some solid performance. On a TPU, you couldn't get as many batches, but inference was probably a third of the time. But there's no concurrency with a TPU, and there wasn't really built-in support since it was running through jacks. But once you start to try to do, like, shot boundary detection and reading metadata and extracting audio as well as video frames, things start to get a lot airier, quickly. I was trying to run shot boundary detection using read video frames, and you run into this issue once you get to the actual shot boundary detection. So the traditional way of calculating this is with a histogram, and that's easy enough. That's a row-wise operation. But a shot boundary is fundamentally checking the previous frame against the next frame and doing some calculations to see if one is larger than the other, and then some thresholds. And the chi-squared distance needs to be changed enough for something. And then, you know, you're like, okay, how do I run this efficiently? Well, maybe the naive way to do it is that you group by... you do a modulus on the frame ID, divide it by two, and then you get this group ID. But then you're only checking really half of the samples for boundaries, so you've got to do it twice. And that's, like, you know, not great. Then the other idea is, you know, you try to pack it, you try to create these lists, you can do these group by lists. But then, you know, the thought occurs to you that, like, wait, couldn't I run this in batches in a batch UDF? You know, I'll have access to a series of frames. You know, I'll just sort it, and then I'll run it back and return. No, you can't do that. You can't sort inside of batch UDF. And this ends up becoming a problem, because you don't know when you perform batch inference whether or not you're actually getting a sequence of frames. You could perform a sort, but once you go distribute it, there's not a guarantee. And so that's why you have to do the group by. You do the group by, and then you assign the label of whether or not this is a shot boundary, and then you have to explode. But guess what? You have to do the exact same thing again if you're going to go and make your clips. So, like, guess what you do after you go and make your clips? You do your, you already did your group by to make your clips of 16 frames, and then inside your batch inference UDF, you can perform a stacking. It doesn't matter, because we're generating embeddings here. But even then, you're supposed to have spatial temporal metrics, right? So fundamentally, you're supposed to have this group by operation, and then if you want to go all the way back to frames, you have to explode again. And that's pretty normal. It's just that we're dealing with image frames, and that means that there's a decent amount of data. And the reality is, when we're reading frames from the file, that's the easiest time to do all of these calculations. We can pull the amount of data cheaply enough, but we can do all of these concurrent reads by seeking and creating a seek plan, and then pulling all of this data up again, right? Like, at the end of the day, you have to read all of the frames in the file. You do have to perform shot boundary detection on every pair of frames. And at the end of the day, it's just going to take as long as... Your ability to horizontally scale is about whether or not you're performing these operations in some sort of vectorized format, it's cheapest to do, especially like these histogram things, these conversions, as you're reading the frame, right? So, you know, it might be fine to just do the multiple group-bys and explode, group-by and explode, group-by and explode. You know, I'm sure you can process a lot of things a lot faster, especially since we're going to be throwing these numpy arrays into GPUs, right? But, like, especially for things like the shot boundary detection, it honestly makes a little bit more sense to do so when you are reading from the file. Otherwise, you have to do all these group-bys and explode. So, you know, supposedly, that's not a big deal. Until you start to think about what that looks like, you know, when you're shuffling, trying to get the sword, and all of the cool stuff. So, you know, I guess this is where maybe window functions would be more efficient. There are several different strategies here, but there's just a lot of pain, right? Like, you're not sure as you're developing this which one's the right way, and you're forced really, really early on to commit. And if you're trying to keep your memory footprint low, because you're throttled at inference, and you're hoping that your pipeline is just feeding you a steady stream of frames that you can then eventually pack into your inference, But, fundamentally, you will consistently run into this, you know, windowing problem. And, um, that's why Read Video Frames isn't as useful as it could be if it came with a few options to say, you know, also give me the audio, also give me the histogram, also give me these other things. Because, you know, we have to start with FFmpeg, and at the frame level there's all this data, and if you already have to decode it, you've already done the work. So, decide up front whether you just want to do that work, and do it. But, I think you can tell that there's these natural friction points that make video processing especially difficult. And, again, if we're talking about pipeline sizing, and we're talking about the different ways that we're batching this stuff, um, you know, and trying to minimize memory footprint because inference takes so much memory, um, and we don't just want to give our lives away to Redis, then, you know, we're going to be stuck.

Just to add to the list, because I'm mostly just taking notes here. I did try a few different metadata reading approaches. I tried launching a subprocess inside of a UDF to read a file. Just to get the width and height. Frame rate, or frames per second. Then you want some keyframe data, right? Or you want something else. And I found that using PyAV with the open file as container pattern to be 10x faster. I think there's something to be said about that, where subprocess is probably just really slow inside of a UDF. And I think that makes sense. There's a whole host of metadata that we could read, but maybe we choose not to. And from what it sounds like, I'm covering my bases for the most part. But yeah, I thought that was important to include as well. I think that there's a couple approaches here. The full end-to-end workload that I'm trying to demonstrate is that we can take a directory of videos and we can read the video frames and the audio frames. We can generate embeddings from those audio frames and video frames. And then perform all these traditional WAG types of use cases against that. I think as much as I would love to fully demonstrate Q&A, I think the reality is that I don't have enough time. And I'll just generate the embeddings and demonstrate them. Q&A would be more useful. I'm already going to be doing transcription of audio. But there's these two different paradigms here as well. So reading image frames versus reading audio packets are not necessarily the same workload. It's not the same amount of data, but it's more data than your average image or text. And you might not need the same type of fashion strategy for audio as you do video. And you can potentially just do it separately. But technically, you would think that it would be the most efficient thing to read, to grab both the audio and the video at the same time. Simply because you have to decode the frame. So maybe that's not the case. Maybe you always want to read the audio differently. But when you're processing audio, there's a bunch of different sampling techniques. But at the end of the day, it's a little bit more memory bound. I don't know. Audio just isn't as prohibitively big as the image files can be. But I guess it can be much bigger. And technically, you're at a higher sampling rate. I don't know what actually ends up being the end. So I think there's several approaches here. There's this mega read where you take everything in a single pass. And naturally, it makes sense to break that up with metadata. And maybe it never makes sense to read both audio and video at the same time. It's just cheap enough to justify not having to do that. But if you're segmenting your audio and your video at the same levels, at the same frame indices, are you going to want to process them the same way? Especially if you're doing shot boundaries and textures. So in that case, I would say, yeah, you probably do want to segment at the shot boundary. But who knows? It might not be as important. I think it depends on the use case. But you just see how many different decisions you have to make. It can be overwhelming.

---


## Friction: Shot Boundary Detection (Histograms) + Streaming + Daft UDFs

- Sorting before batch UDFs: For sequence algorithms (SBD), the row order in a group is not guaranteed. I must group by `path` and aggregate lists, then sort by `frame_index` inside the list-UDF. I cannot sort the DataFrame inside a UDF. This makes groupby+agg_list the default pattern for sequence ops.
- Two-frame grouping is brittle: Bucketing into size-2 groups loses edges across bucket boundaries and forces extra filters. A per-path sequence pass computing adjacent diffs is cleaner and more accurate.
- Streaming is cheaper: SBD is naturally streaming-friendly. Computing histograms and adjacent diffs while decoding avoids storing/sorting large per-frame lists. For now, a frame-level lead/lag join is a good relational compromise.
- Keyframes are not semantic cuts: Relying on codec keyframes can miss true cuts or add false positives. Use them only to narrow refinement windows after a coarse pass.
- Join-based alternative (no list sorting): Pre-sort by (`path`,`frame_index`), self-join current↔next rows, compute distances row-wise, then only group to assign contiguous `shot_id`s (or to enforce minimum shot length).
- Minimum shot length: Needed to suppress flicker; parameterizing this in the SBD UDF is useful (e.g., 6–12 frames at working FPS).
- UDF ergonomics:
  - Type hints: returning `np.ndarray` in the signature isn’t supported; must declare `@daft.func(return_dtype=...)` explicitly.
  - Bool dtype naming is inconsistent across examples (`dt.bool` vs `dt.bool_`). Minor paper-cut.
  - Image dtype normalization: casting directly to `dt.tensor(dt.float32())` from image dtype fails; need an explicit UDF to `np.asarray(image).astype(np.float32)/255.0` first.
  - Yield from UDF not supported; streaming has to return lists/structs or be done in a file-level UDF that returns aggregated results.
- Aggregation naming: After `.agg_list("col")`, the resulting column is auto-suffixed (e.g., `col_agg`). Explicit `.alias(...)` is clearer.
- Explode late: Keep data as grouped lists to stay vectorized; explode to per-clip rows only when feeding the model.
- Warmup shapes: Model warmup needs known `(B,T,H,W,C)`. If baked into UDF `__init__`, these shapes must come from outer config; otherwise skip warmup to avoid lints.

Actionables I’d like from Daft:
- Clarify bool dtype naming, document list aggregation column names, and provide a simple lead/lag helper without joins.
- Add a streaming-friendly example for video SBD (histograms + adjacent diffs) and a cookbook pattern for “detect shots → assign shot_id → window clips inside shots”.