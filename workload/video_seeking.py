import daft
from daft import col, DataType as dt
from daft.functions import file
import av
import time
import numpy as np
from dataclasses import dataclass
from fractions import Fraction





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
def fetch_video_metadata(
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

@dataclass
class _MultiStreamVideoFrame:
    """Represents a single video frame.

    Note:
        The field name 'data' is required due to a casting bug.
        See: https://github.com/Eventual-Inc/Daft/issues/4872
    """

    path: str
    stream_index: int
    frame_time_ns: int
    frame_time: float
    frame_time_base: Fraction
    frame_pts: int
    frame_dts: int | None
    frame_duration: int | None
    is_key_frame: bool
    data: np.ndarray

def select_stream_by_index(container: av.container.input.InputContainer, stream_index: int) -> av.video.stream.VideoStream:
    vs = container.streams.video[stream_index]
    if getattr(vs.disposition, "attached_pic", False):
        raise ValueError("Selected stream is an attached picture/thumbnail.")
    return vs

def pts_time_ns(pts: int | None, time_base: Fraction) -> int | None:
    if pts is None:
        return None
    # exact integer nanoseconds without float rounding
    return (pts * time_base.numerator * 1_000_000_000) // time_base.denominator

def safe_decode_packet(packet: av.packet.Packet) -> list[av.video.frame.VideoFrame]:
    try:
        return packet.decode()
    except av.AVError:
        return []
    except Exception:
        return []
         

@daft.func(
    return_dtype=
        dt.struct(
            {
                "path": dt.string(),
                "stream_index": dt.int32(),
                "frame_time": dt.float64(),
                "frame_time_base": dt.string(),
                "frame_time_ns": dt.int64(),
                "frame_pts": dt.float64(),
                "frame_dts": dt.float64(),
                "frame_duration": dt.float64(),
                "is_key_frame": dt.bool(),
                "data": dt.binary(),  # stores RGB bytes (H x W x 3)
            }
        )
)
def seek_video_frames(
    file: daft.File,
    *,
    start_sec: float = 0.0,
    end_sec: float = float("inf"),
    probesize: str = "64k",
    analyzeduration_us: int = 200_000,
    width: int = 288, 
    height: int = 288, 
    ):
    
    options = {
        "probesize": str(probesize),
        "analyzeduration": str(analyzeduration_us),
    }
    eps = 1e-6
    with av.open(file, mode="r", options=options, metadata_encoding="utf-8") as container:
        # Select video streams (exclude attached thumbnails)
        vs = next((s for s in container.streams if s.type == "video"), None)
        vs.thread_type = "AUTO"

        # Compute seek position and optional end bound in this stream's ticks
        ts_start = int(start_sec / float(vs.time_base)) if start_sec > 0 else 0
        end_pts = None if end_sec == float("inf") else int(end_sec / float(vs.time_base))

        container.seek(ts_start, stream=vs, any_frame=False, backward=True)


        # Decode frames only from this stream
        for frame in container.decode(vs):
            if frame.pts is None:
                continue

            t = frame.pts * float(vs.time_base)
            if t + eps < start_sec:
                continue
            if end_pts is not None and frame.pts > end_pts:
                break

            # Resize & convert
            if width and height:
                frame = frame.reformat(width=width, height=height)

            yield {
                "path": str(file),
                "stream_index": int(vs.index),
                "frame_time": float(frame.time),
                "frame_time_base": str(frame.time_base),
                "frame_time_ns": pts_time_ns(frame.pts, frame.time_base),
                "frame_pts": float(frame.pts),
                "frame_dts": float(frame.dts) if frame.dts is not None else float("nan"),
                "frame_duration": float(frame.duration) if frame.duration is not None else float("nan"),
                "is_key_frame": bool(frame.key_frame),
                "data": frame.to_ndarray(format="rgb24").tobytes(),
            }


@daft.func(return_dtype=dt.struct({
    "path": dt.string(),
    "frame_time": dt.int32(),
    "tensor": dt.binary(),
}))
def fake_yield_frames(file: daft.File):
    for i in range(16):
        arr = np.zeros((288, 288, 3), dtype=np.float32)
        return {
            "path":str(file),
            "frame_time": i,
            "tensor": arr.tobytes(),
        }


@daft.func(return_dtype=dt.list(dt.float64()))
def linspace(start: float, end: float, num: int):
    step = (end - start) / (num - 1)
    return [start + i * step for i in range(int(num))]


def main(uri: str, seek_duration: float, num_batches: int, B: int, T: int, W: int, H: int):
    start = time.time()

    for i in range(num_batches):
        df = (
            daft.from_glob_path(uri)
            .with_column("file", file(col("path")))
            .with_column("meta", fetch_video_metadata(col("file")))
            .with_column("duration", col("meta")["duration"])
            .with_column("starts", 
                linspace(0.0, col("meta")["duration"], col("meta")["duration"]//max_video_seek_read_duration)
            ).explode("starts")
            .with_column("frames",
                seek_video_frames(
                    col("file"),
                    start_sec=col("starts"),
                    end_sec=col("starts") + max_video_seek_read_duration,
                    width=W,
                    height=H,
                )
            )
        )
        batches = df.iter_batches(batch_size=B)
        

if __name__ == "__main__":
    uri = "../videoprism/videoprism/assets/*.mp4"
    B, T, H, W, C = 2, 16, 288, 288, 3 # Batch Size, Clip Size (# frames), Height, Width, RGB
    ROW_LIMIT = 500
    eager = True
    interp = None
    max_video_seek_read_duration = 10.0 # How wide of a batch to read from the video at a time

    start = time.time()

    
    # Files → Metadata
    df_files = (
        daft.from_glob_path(uri)
        .with_column("file", file(col("path")))
    )
    

    df_meta = df_files.with_column("meta", fetch_video_metadata(col("file")))
   

    df_ugh = df_meta.with_column("duration", col("meta")["duration"])


    df_seek_plan = df_meta.with_column("starts", 
        linspace(0.0, col("meta")["duration"], col("meta")["duration"]//max_video_seek_read_duration)
    ).explode("starts")

    df_clips = df_seek_plan.with_column("frames",
        seek_video_frames(
            col("file"),
            start_sec=col("starts"),
            end_sec=col("starts") + max_video_seek_read_duration,
            width=W,
            height=H,
        )
    ).limit(ROW_LIMIT)


    # Unpack metadata struct into individual columns
    df_final = df_clips.select(
        col("path"),
        col("meta")["width"].alias("width"),
        col("meta.height").alias("height"),
        col("meta")["fps"].alias("fps"),
        col("meta")["duration"].alias("duration"),
        col("meta")["frame_count"].alias("frame_count"),
        col("meta")["time_base"].alias("time_base"),
        col("meta")["keyframe_pts"].alias("keyframe_pts"),
        col("meta")["keyframe_indices"].alias("keyframe_indices"),
        col("frames")["stream_index"].alias("stream_index"),
        col("frames")["frame_time"].alias("frame_time"),
        col("frames")["data"].image.decode(daft.ImageMode.RGB).alias("image"),

    )


    print(f"Time taken: {time.time() - start:.2f}s")


    df_clips.show()
