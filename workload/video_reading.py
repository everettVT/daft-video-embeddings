import daft
from daft import col, DataType as dt
from daft.functions import file
import av
from av.audio.resampler import AudioResampler
import time
import numpy as np

daft.read_video_frames()
# Helpers -----------------------------------

def frame_to_rgb_float32(frame: av.VideoFrame, w: int, h: int, interp: str = None) -> np.ndarray:
    """Convert an AV frame to a normalized RGB float32 array."""
    return frame.to_ndarray(
        width=w,
        height=h,
        format="rgb24",
        interpolation=interp,
    ).astype(np.float32) / 255.0

def frame_to_daft_image(frame: av.VideoFrame) -> daft.Image:
    return daft.Image.from_numpy(frame.to_ndarray(format="rgb24"))

## Probe Metadata ----------------------------------- 

@daft.func(return_dtype = dt.struct({
    "width": dt.int32(),
    "height": dt.int32(),
    "fps": dt.float64(),
    "duration": dt.float64(),
    "frame_count": dt.int32(),
    "time_base": dt.float64(),
    "keyframe_pts": dt.list(dt.float64()),
    "keyframe_indices": dt.list(dt.int32()),
}))
def get_video_metadata(
    file: daft.File,
    *,
    probesize: str = "64k",
    analyzeduration_us: int = 200_000,
) -> dict:
    """
    Extract basic video metadata from container headers.

    Returns
    -------
    dict
        width, height, fps, frame_count, time_base, keyframe_pts, keyframe_indices
    """
    options = {
        "probesize": str(probesize),
        "analyzeduration": str(analyzeduration_us),
    }

    with av.open(file,mode="r", options=options, metadata_encoding="utf-8") as container:
        video = next(
            (stream for stream in container.streams if stream.type == "video"),
            None,
        )
        if video is None:
            return {
                "width": None,
                "height": None,
                "fps": None,
                "frame_count": None,
                "time_base": None,
                "keyframe_pts": [],
                "keyframe_indices": [],
            }

        # Basic stream properties ----------
        width = video.width
        height = video.height
        time_base = float(video.time_base) if video.time_base else None

        # Frame rate -----------------------
        fps = None
        if video.average_rate:
            fps = float(video.average_rate)
        elif video.guessed_rate:
            fps = float(video.guessed_rate)

        # Duration -------------------------
        duration = None
        if container.duration and container.duration > 0:
            duration = container.duration / 1_000_000.0
        elif video.duration:
            # Fallback time_base only for duration computation if missing
            tb_for_dur = float(video.time_base) if video.time_base else (1.0 / 1_000_000.0)
            duration = float(video.duration * tb_for_dur)

        # Frame count -----------------------
        frame_count = video.frames
        if not frame_count or frame_count <= 0:
            if duration and fps:
                frame_count = int(round(duration * fps))
            else:
                frame_count = None

        # Keyframes -----------------------
        keyframe_pts = []
        try:
            for packet in container.demux(video):
                if packet.is_keyframe and packet.pts is not None:
                    pts_seconds = float(packet.pts * float(video.time_base))
                    keyframe_pts.append(pts_seconds)
        except Exception:
            keyframe_pts = []

        keyframe_indices = (
            [int(round(t * fps)) for t in keyframe_pts] if fps else []
        )

        return {
            "width": width,
            "height": height,
            "fps": fps,
            "duration": duration,
            "frame_count": frame_count,
            "time_base": time_base,
            "keyframe_pts": keyframe_pts,
            "keyframe_indices": keyframe_indices,
        }


## Traditional Frame-by-Frame Ingestion -----------------------------------
# We read all frames in a file and index them by frame index. 

# Image Frames


# Audio Frames



## Distributed Concurrent Read - Seek-then-Read Ingestion -----------------------------------
# Build a Seek Plan, then distribute reads concurrently from the plan.

# Plan a uniform time-based start for each batch. Index by frame index. 
@daft.func(return_dtype=dt.list(dt.float64()))
def make_uniform_clip_starts(
    fps: float,
    duration: float,
    T: int,
    stride_frames: int | None = None,
) -> list[float]:
    """Create time-based start seconds for fixed-length T-frame clips.

    Uses fps and duration to produce deterministic starts: t = k * (stride_frames/fps)
    Keep starts while t + T/fps <= duration.
    """
    if fps is None or duration is None or fps <= 0 or duration <= 0 or T <= 0:
        return []
    if stride_frames is None or stride_frames <= 0:
        stride_frames = T
    stride_sec = stride_frames / float(fps)
    clip_dur_sec = T / float(fps)
    starts: list[float] = []
    t = 0.0
    eps = 1e-6
    while t + clip_dur_sec <= duration + eps:
        starts.append(float(t))
        t += stride_sec
    return starts

@daft.func(return_dtype=dt.struct({
    "start_sec": dt.float64(),
    "end_sec": dt.float64(),
    "start_frame": dt.int32(),
    "end_frame": dt.int32(),
    "is_keyframe_start": dt.bool(),
}))
def build_plan_clip_seek(
    keyframe_pts: list[float],
    keyframe_indices: list[int],
    fps: float,
    clip_duration: float = 2.0,
    max_clips: int = None,
) -> list[dict]:
    """
    Plan clip extraction using keyframe information.
    
    Returns a list of clip plans, each specifying:
    - start_sec: start time in seconds
    - end_sec: end time in seconds  
    - start_frame: start frame index
    - end_frame: end frame index
    - is_keyframe_start: whether clip starts on a keyframe
    
    Parameters
    ----------
    keyframe_pts : list[float]
        Keyframe timestamps in seconds
    keyframe_indices : list[int]
        Keyframe frame indices
    fps : float
        Frames per second
    clip_duration : float, default 2.0
        Duration of each clip in seconds
    max_clips : int, optional
        Maximum number of clips to return
    """
    if not keyframe_pts or not keyframe_indices or fps is None or fps <= 0:
        return []
    
    clips = []
    clip_frames = int(round(clip_duration * fps))
    
    # Generate clips starting from each keyframe
    for i, (kf_sec, kf_frame) in enumerate(zip(keyframe_pts, keyframe_indices)):
        start_sec = kf_sec
        end_sec = start_sec + clip_duration
        start_frame = kf_frame
        end_frame = start_frame + clip_frames
        
        clips.append({
            "start_sec": start_sec,
            "end_sec": end_sec,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "is_keyframe_start": True,
        })
    
    # Also generate clips between keyframes if there are gaps
    for i in range(len(keyframe_pts) - 1):
        kf_sec = keyframe_pts[i]
        next_kf_sec = keyframe_pts[i + 1]
        
        # Skip if gap is smaller than clip duration
        if next_kf_sec - kf_sec <= clip_duration:
            continue
            
        # Generate intermediate clips
        current_sec = kf_sec + clip_duration
        while current_sec + clip_duration <= next_kf_sec:
            start_frame = int(round(current_sec * fps))
            end_frame = start_frame + clip_frames
            
            clips.append({
                "start_sec": current_sec,
                "end_sec": current_sec + clip_duration,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "is_keyframe_start": False,
            })
            current_sec += clip_duration
    
    # Sort by start time
    clips.sort(key=lambda x: x["start_sec"])
    
    # Limit number of clips if requested
    if max_clips is not None:
        clips = clips[:max_clips]
    
    return clips



def seek_video_frames(
    file: daft.File, 
    start_sec: float,
    probesize: str = "64k",
    analyzeduration_us: int = 200_000,
    num_frames: int = 16, 
    width: int = 288, 
    height: int = 288, 
    interp: str = None
    ):
    
    options = {
        "probesize": str(probesize),
        "analyzeduration": str(analyzeduration_us),
    }

    with av.open(file,mode="r", options=options, metadata_encoding="utf-8") as container:
        vs = container.streams.video[0]
        vs.thread_type = "AUTO"

        # 1) Compute seek offset in stream ticks
        ts = int(start_sec / float(vs.time_base))  # seconds -> ticks

        # 2) Seek to keyframe <= start_sec
        container.seek(ts, stream=vs, any_frame=False, backward=True)

        # 3) New decode loop; drop until PTS >= start_sec
        out = np.empty((num_frames, height, width, 3), dtype=np.float32)
        got = 0
        target = start_sec
        eps = 1e-6

        for frame in container.decode(video=0):
            if frame.pts is None:
                continue
            t = frame.pts * float(vs.time_base)
            if t + eps < target:
                continue  # not reached start yet
             
            # 4) Collect frames
            arr = frame_to_rgb_float32(frame, w=width, h=height, interp=interp)
            out[got] = arr
            got += 1
            if got == num_frames:
                break

        # If fewer than requested frames exist, pad by repetition for deterministic shape
        if got < num_frames:
            if got == 0:
                return np.zeros((0, height, width, 3), dtype=np.float32)
            last = out[got - 1]
            for i in range(got, num_frames):
                out[i] = last
        return out



@daft.func()
def seek_audio_frames(file: daft.File, start_sec: float, end_sec: float, num_frames: int = 16, ) -> np.ndarray:

    # Check mime_type is audio
    if file.mime_type != "audio/mp4":
        raise ValueError("File is not an audio file")

    options = {
        "probesize": str(probesize),
        "analyzeduration": str(analyzeduration_us),
    }
    resampler = AudioResampler(format='s16', layout='mono', rate=16000)

    with av.open(file, mode="r", options=options, metadata_encoding="utf-8") as container:
        aus = container.streams.audio[0]
        aus.thread_type = "AUTO"

        # 1) Compute seek offset in stream ticks
        ts = int(start_sec / float(aus.time_base))  # seconds -> ticks

        # 2) Seek to keyframe <= start_sec
        container.seek(ts, stream=aus, any_frame=False, backward=True)

        # 3) New decode loop; drop until PTS >= start_sec

        chunks = []
        try:
            for frame in container.decode(audio=0):
                # Resample to desired SR/mono/PCM16; result can be a frame or list of frames
                res = resampler.resample(frame)
                frames = res if isinstance(res, (list, tuple)) else [res]

                for f in frames:
                    arr = f.to_ndarray()  # typically (channels, samples) or (samples,)

                    # Flatten to 1-D mono
                    if arr.ndim == 2:
                        # (1, N) or (N, 1) → (N,)
                        if arr.shape[0] == 1:
                            arr = arr[0]
                        elif arr.shape[1] == 1:
                            arr = arr[:, 0]
                        else:
                            # Unexpected multi-channel after mono resample: average as fallback
                            arr = arr.mean(axis=0)
                    elif arr.ndim > 2:
                        arr = arr.reshape(-1)

                    # Convert PCM16 → float32 in [-1, 1]
                    if arr.dtype != np.float32:
                        arr = (arr.astype(np.float32) / 32768.0).clip(-1.0, 1.0)

                    chunks.append(arr)
        finally:
            container.close()

    if not chunks:
        return np.zeros((0,), dtype=np.float32)

    audio = np.concatenate(chunks, axis=0).astype(np.float32, copy=False)
    return audio


# Clip Segmentation -----------------------------------

@daft.func(return_dtype=dt.tensor(dt.float32()))
def stack_clip(frames: list[np.ndarray], indices: list[int], clip_size: int):
    """Sort by indices, stack to (T,H,W,3) float32 [0,1], pad tail by repetition."""
    order = np.argsort(np.asarray(indices))
    def to_np(x):
        if hasattr(x, "to_numpy"):
            return x.to_numpy()
        return np.asarray(x)
    frames_sorted = [to_np(frames[i]) for i in order]
    if len(frames_sorted) == 0:
        return np.zeros((clip_size, 1, 1, 3), dtype=np.float32)
    if len(frames_sorted) < clip_size:
        frames_sorted.extend([frames_sorted[-1]] * (clip_size - len(frames_sorted)))
    x = np.stack(frames_sorted[:clip_size], axis=0).astype(np.float32)
    if x.max() > 1.0:
        x = x / 255.0
    return x

@daft.func(return_dtype=dt.list(dt.tensor(dt.float32())))
def decode_all_frames_to_clips(
    file: daft.File,
    num_frames: int,
    width: int,
    height: int,
    interp: str | None = None,
):
    """Decode full video and return list of (T,H,W,3) float32 clips (non-overlapping)."""
    clips: list[np.ndarray] = []
    frames: list[np.ndarray] = []
    try:
        with av.open(file, mode="r") as container:
            vs = container.streams.video[0]
            vs.thread_type = "AUTO"

            for frame in container.decode(video=0):
                if frame.pts is None:
                    continue
                arr = frame_to_rgb_float32(frame, w=width, h=height, interp=interp)
                frames.append(arr)
                if len(frames) == num_frames:
                    clips.append(np.stack(frames, axis=0))
                    frames.clear()
            # tail
            if len(frames) > 0:
                last = frames[-1]
                while len(frames) < num_frames:
                    frames.append(last)
                clips.append(np.stack(frames, axis=0))
    except Exception:
        return []
    return clips

## Pipelines -----------------------------------

def ingest_with_read_video_frames(uri: str, T: int, H: int, W: int):
    """Approach 1: daft.read_video_frames → groupby stack into fixed-size clips."""
    df = (
        daft.read_video_frames(uri, image_height=H, image_width=W)
        .with_column("clip_index", col("frame_index") // T)
        .groupby("path", "clip_index").agg_list("data", "frame_index")
        .with_column("clip", stack_clip(col("data"), col("frame_index"), T))
        .select("path", "clip")
    )
    return df

def ingest_with_pyav_full(uri: str, T: int, H: int, W: int, interp: str | None = None):
    """Approach 2: Decode all frames in a file-level UDF, emit list of clips, then explode."""
    df = (
        daft.from_glob_path(uri)
        .with_column("file", file(col("path")))
        .with_column("clips", decode_all_frames_to_clips(col("file"), T, W, H, interp))
        .explode("clips")
        .rename({"clips": "clip"}) `      `
        .select("path", "clip")
    )
    return df

def ingest_with_seek_plan(uri: str, T: int, H: int, W: int, interp: str | None = None):
    """Approach 3: Build a uniform time-based plan, seek+decode per start, return clips."""
    df_files = daft.from_glob_path(uri).with_column("file", file(col("path")))
    df_meta = df_files.with_column("meta", get_video_metadata(col("file")))
    df_plan = (
        df_meta
        .with_column(
            "starts",
            make_uniform_clip_starts(
                col("meta")["fps"], col("meta")["duration"], T, stride_frames=T
            ),
        )
        .explode("starts")
        .with_column("start_sec", col("starts"))
        .drop("starts")
    )
    df = df_plan.with_column(
        "clip",
        seek_video_frames(col("file"), col("start_sec"), num_frames=T, width=W, height=H, interp=interp),
    ).select("path", "clip")
    return df

def main(uri: str, row_limit: int, B: int, T: int, W: int, H: int, interp: str = None):
    start = time.time()

    print("Approach 1: read_video_frames → groupby stack")
    t0 = time.time()
    df1 = ingest_with_read_video_frames(uri, T, H, W)
    df1.limit(row_limit).collect()
    print(f"  took {time.time() - t0:.2f}s")

    print("Approach 2: pyav full decode → list-of-clips")
    t0 = time.time()
    df2 = ingest_with_pyav_full(uri, T, H, W, interp)
    df2.limit(row_limit).collect()
    print(f"  took {time.time() - t0:.2f}s")

    print("Approach 3: seek plan → deterministic starts")
    t0 = time.time()
    df3 = ingest_with_seek_plan(uri, T, H, W, interp)
    df3.limit(row_limit).collect()
    print(f"  took {time.time() - t0:.2f}s")

    print(f"Total time: {time.time() - start:.2f}s")
    return {"read_video_frames": df1, "pyav_full": df2, "seek_plan": df3}



if __name__ == "__main__":
    uri = "../videoprism/videoprism/assets/*.mp4"
    _ = main(uri, row_limit=10, B=2, T=16, W=288, H=288, interp=None)


   