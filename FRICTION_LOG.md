# Video Embeddings Friction Log (Reorganized)

**Date**: Aug 25, 2025  
**Author**: Everett Kleven  
**Persona**: UDF‑naive user exploring end‑to‑end video pipelines

## Executive Summary

- **Video processing is decision-heavy**: Early choices (ingestion, segmentation, batching) shape the entire pipeline. Convenience APIs like `read_video_frames()` are great for demos but insufficient for real workloads needing streaming, seeking, or sequence ops.
- **Sequence algorithms need ordering guarantees**: Group by `path` → aggregate lists → sort inside list‑UDF is the reliable pattern. Avoid assuming stable row order in distributed settings.
- **Do work at decode time when possible**: Histograms, SBD diffs, and simple precomputations are cheaper while decoding than after materialization + shuffles.
- **UDF ergonomics matter**: Image dtypes require explicit normalization; ndarray return typing must be declared with Daft dtypes, not `np.ndarray` hints.
- **Performance observations**: GPU batching is strong; TPU JAX/XLA can compile huge graphs and OOM unexpectedly if not carefully batched. PyAV beats subprocess for metadata.

## Scope and Goals

Explore friction points for video AI pipelines when loading all frames into memory is prohibitive. Build clips (e.g., 16 frames), batch for inference, and evaluate streaming/seek‑based strategies including SBD and audio transcription.

## Workflows Attempted

- **Ingestion modes**
  - `read_video_frames()` → decode frames as rows with `frame_index`.
  - `probe_video_metadata()` + `read_video_file(..., hist, sbd, audio)`.
  - `probe_video_metadata()` + `seek_video_file(..., hist, sbd, audio)` with preplanned frame timestamps for concurrent reads.
- **Segmentation**
  - SBD via frame histograms and adjacent diffs; groupby+agg_list patterns for sequence ops; clip assembly (T=16) and batch packing.
- **Inference**
  - Stack clips to `(B,T,H,W,C)` for model throughput; pad tails and batches as needed.
- **JAX/TPU/GPU**
  - JIT paths on CPU/GPU worked; A100 handled large batch stacks; TPU suffered XLA compile and HBM OOM for larger DF sizes.
- **Audio (Parakeet)**
  - Preprocessed audio as `np.ndarray` list; UDF complained when passed a Series rather than `str | np.ndarray | torch.Tensor` per‑row.
- **File I/O**
  - `daft.File` exploration; need path extraction for PyAV; successful concurrent seeks across the same file.
- **Streaming**
  - Generator attempts from UDFs failed (no `yield` in UDF); pivoted to seek plans and returning lists/structs.

## Detailed Findings & Frictions

### UDF Ergonomics and Data Types

- **Image dtype → float tensor**: Direct cast fails; use an explicit normalization UDF to `float32` in `[0,1]`.
- **Return typing**: `@daft.func(return_dtype=...)` is required; Python hint `-> np.ndarray` is not recognized.
- **Shape discipline**: Models expect `(B,T,H,W,C)`; define clip size `T` and normalization in a single UDF to avoid extra passes.

Minimal recipes:

```python
@daft.func()
def normalize(image: np.ndarray) -> dt.tensor(dt.float32()):
    return np.asarray(image).astype(np.float32) / 255.0
```

```python
@daft.func(return_dtype=dt.tensor(dt.float32(), shape=(1, 16, 288, 288, 3)))
def stack_clip(frames: list[np.ndarray], indices: list[int], clip_size: int):
    order = np.argsort(np.asarray(indices))
    def to_np(x):
        return x.to_numpy() if hasattr(x, "to_numpy") else np.asarray(x)
    frames_sorted = [to_np(frames[i]) for i in order]
    if len(order) < clip_size:
        frames_sorted.extend([frames_sorted[-1]] * (clip_size - len(order)))
    x = np.stack(frames_sorted[:clip_size], axis=0).astype(np.float32) / 255.0
    return x[None, ...]
```

Key errors seen:

```text
DaftCoreException: StructArray::new received ... expected child field: List[Float32]
```

```text
ValueError: Unrecognized Python type ... <class 'numpy.ndarray'>
```

### Sequence Ops and Ordering

- **Don’t rely on row order** in distributed groupbys. Use `groupby(path).agg_list([...])`, then sort inside the list‑UDF using the aggregated `frame_index`.
- **Explode late** to stay vectorized. Keep list columns through sequence transforms; explode only before model inference or when row‑wise outputs are required.
- **Two‑frame bucketing is brittle**: Edges are missed across bucket boundaries; prefer a single per‑path pass or a join‑based adjacent diff.

### Shot Boundary Detection (SBD)

- **Streaming‑friendly**: Compute histograms and adjacent chi‑squared distances while decoding to avoid heavy shuffles.
- **Join‑based alternative**: Pre‑sort by (`path`,`frame_index`), self‑join `current ↔ next`, compute distances row‑wise, then group to assign contiguous `shot_id`s (enforce min shot length).
- **Keyframes are not cuts**: Use codec keyframes only to narrow search windows post coarse pass; they are not semantic boundaries.
- **Minimum shot length**: Parameterize (e.g., 6–12 frames at working FPS) to suppress flicker.

### Streaming vs Frame Materialization

- **Do early compute at decode**: Histograms, simple transforms, and ordering metadata are cheapest during decode.
- **Seek plans work**: Multiple seeks into the same file can run concurrently and atomize memory if batching is tuned to inference capacity.
- **UDF constraints**: No `yield` from UDFs; return lists/structs or move streaming to file‑level ops that aggregate results.

### JAX/TPU/GPU Observations

- **GPU batching**: A100 handled up to 24 clips × 16 frames with good throughput.
- **TPU/XLA**: Larger DFs triggered massive program HBM use and OOM during compile; small batch sizes worked, but scaling hit compiler limits.

Excerpt:

```text
XlaRuntimeError: RESOURCE_EXHAUSTED ... Used 55.24G of 31.25G hbm
```

### Audio / Parakeet Integration

- **Input types**: UDF received a `Series`; Parakeet expects per‑row `str | np.ndarray | torch.Tensor`.

Excerpt:

```text
ValueError: Input `audio` is of type <class 'daft.series.Series'> ...
```

### File I/O with `daft.File` and PyAV

- **Path extraction**: Passing `daft.File` directly to PyAV fails; `str(daft.File)` includes a wrapper (`File(file://...)`) that PyAV can’t open. A helper to extract the raw path would remove confusion.
- **Read/seek/tell**: Minimal docs in class docstrings would help. Once learned, seeking to timestamps unlocked concurrent chunk reads.
- **PyAV vs subprocess**: PyAV metadata reads were ~10× faster than subprocess calls inside UDFs.

Error examples:

```text
TypeError: expected str, bytes or os.PathLike object, not PathFile
```

```text
FileNotFoundError: ... 'File(file:///Users/everett/Movies/Running.mp4)'
```

## Practical Recipes

- **Normalize images to float32**: Use a tiny UDF (above). Avoid direct `dt.tensor(float32)` casts from image dtype.
- **Clip assembly with ordering + padding**: Use `stack_clip` (above). Pass `clip_size` and indices; add a batch dimension.
- **Adjacent diff without list sorting**: Sort DF by (`path`,`frame_index`), self‑join `lead(frame_index, 1)` semantics via join, compute histogram distances row‑wise, then group to form `shot_id`s and enforce min‑length.

## Actionables for the Daft Team

- **Clarify dtypes**: Boolean dtype naming consistency (`dt.bool` vs `dt.bool_`), document list aggregation column names.
- **Lead/lag helper**: Provide a simple lead/lag without self‑joins to support sequence ops ergonomically.
- **Streaming SBD example**: Cookbook for histograms + adjacent diffs + `shot_id` assignment + clip windowing inside shots.
- **VideoType ergonomics**: Options to compute histograms/SBD/audio during decode; guidance on when to read audio with video.
- **`daft.File` UX**: Add `to_path()` or similar; docstring hints for read/seek/tell.

## Open Questions

- When is it beneficial to read audio and video together vs separately?
- Should SBD always be computed during decode, or are relational window functions competitive enough?
- Best practices for batching with JAX/TPU to avoid XLA OOMs while preserving throughput?

## Appendix: Error Catalog (selected)

```text
DaftCoreException: StructArray::new ... expected child field: List[Float32]
```

```text
ValueError: Unrecognized Python type ... <class 'numpy.ndarray'>
```

```text
FileNotFoundError: 'File(file:///...)'
```

```text
XlaRuntimeError: RESOURCE_EXHAUSTED: XLA:TPU compile permanent error. Ran out of memory ...
```

References: VideoType discussion (`https://github.com/Eventual-Inc/Daft/discussions/5054`), Daft examples on images, PyAV docs.
