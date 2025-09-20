import daft
from daft import col, DataType as dt
from daft.functions import file
from daft.io.av._read_video_frames import _VideoFrame
import av
from av.audio.resampler import AudioResampler
import time
import numpy as np
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from typing_extensions import TypeAlias

if TYPE_CHECKING:
    from fractions import Fraction
    _VideoFrameData: TypeAlias = np.typing.NDArray[Any]


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
    data: _VideoFrameData

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


@daft.func(return_dtype=dt.struct({
    "stream_index": dt.int32(),
    "frame_index": dt.int32(),
    "frame_time": dt.float64(),
    "frame_time_base": dt.string(),
    "frame_pts": dt.float64(),
    "frame_dts": dt.float64(),
    "frame_duration": dt.float64(),
    "is_key_frame": dt.bool(),
    "data": dt.image(mode="RGB")
}))
def seek_multistream_video_frames(
    file: daft.File, 
    start_sec: float,
    end_sec: float,
    stream_indices: list[int] = None,
    probesize: str = "64k",
    analyzeduration_us: int = 200_000,
    width: int = 288, 
    height: int = 288, 
    interp: str = None
    ):
    
    options = {
        "probesize": str(probesize),
        "analyzeduration": str(analyzeduration_us),
    }

    with av.open(file, mode="r", options=options, metadata_encoding="utf-8") as container:
        # Select streams
        if stream_indices is None:
            streams = [s for s in container.streams.video if not getattr(s.disposition, "attached_pic", False)]
        else:
            streams = [container.streams.video[i] for i in stream_indices]
            streams = [s for s in streams if not getattr(s.disposition, "attached_pic", False)]

        if not streams:
            return

        for s in streams:
            s.thread_type = "AUTO"
        
        # Seek each stream to the start; use the first stream as the anchor
        anchor = streams[0]
        ts = int(start_sec / float(anchor.time_base))
        container.seek(ts, stream=anchor, any_frame=False, backward=True) # jump to the start
        
        # Per-stream bookkeeping
        end_pts_by_stream = {s.index: int(end_sec / float(s.time_base)) for s in streams}
        ended_streams = set()
        eps = 1e-6

        for packet in container.demux(streams):
            if packet.stream.type != "video":
                continue

            for frame in packet.decode():
                vs = frame.stream  # the owning stream
                sidx = vs.index

                if frame.pts is None:
                    continue

                # Cut by start
                t = frame.pts * float(vs.time_base)
                if t + eps < start_sec:
                    continue

                # Cut by end (per-stream). Mark stream ended but keep demuxing others.
                if frame.pts > end_pts_by_stream[sidx]:
                    ended_streams.add(sidx)
                    if len(ended_streams) == len(streams):
                        return
                    continue

                f = frame
                if width and height:
                    f = f.reformat(width=width, height=height, interp=interp)

                yield _MultiStreamVideoFrame(
                    path=str(file),
                    stream_index=sidx,
                    frame_time_ns=pts_time_ns(f.pts, f.time_base),
                    frame_time=f.time,
                    frame_time_base=f.time_base,
                    frame_pts=f.pts,
                    frame_dts=f.dts,
                    frame_duration=f.duration,
                    is_key_frame=f.key_frame,
                    data=f.to_ndarray(format="rgb24"),
                )

            

def main(uri: str, row_limit: int,B: int, T: int, W: int, H: int, interp: str = None):
    start = time.time()

    # Files → Metadata
    df_files = (
        daft.from_glob_path(uri)
        .with_column("file", file(col("path")))
        .where(col("path").str.endswith(".mp4"))
    )
    df_meta = df_files.with_column("meta", get_video_metadata(col("file")))

    # Time-based plan: uniform starts at target fps buckets (stride=T frames)
    df_plan = (
        df_meta
        .with_column(
            "starts",
            make_uniform_clip_starts(
                col("meta")["fps"],
                col("meta")["duration"],
                T,
                stride_frames=T,
            ),
        )
        .explode("starts")
        .with_column("start_sec", col("starts"))
        .drop("starts")
    )

    # Deterministic seek + decode per planned start
    df_clips = df_plan.with_column(
        "clip",
        seek_video_frames(
            col("file"),
            col("start_sec"),
            num_frames=T,
            width=W,
            height=H,
            interp=interp,
        ),
    )

    # Optional: preview limited rows
    _ = df_clips.limit(row_limit).collect()
    print(f"Time taken: {time.time() - start} seconds")
    return df_clips



if __name__ == "__main__":
    uri = "../videoprism/videoprism/assets/*.mp4"
    df = main(uri, row_limit=10, B=2, T=16, W=288, H=288, interp=None)


   