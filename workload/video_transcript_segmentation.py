import daft
from daft import col, DataType as dt
from daft.functions import file
import av
from av.audio.resampler import AudioResampler
import numpy as np
import time


# Utility: convert PTS + time_base to nanoseconds
def pts_time_ns(pts, time_base) -> int:
    if pts is None:
        return 0
    try:
        return int(float(pts) * float(time_base) * 1e9)
    except Exception:
        return 0


@daft.func()
def extract_audio_clips(file: daft.File, start_sec: float, end_sec: float, format: str = 's16', layout: str = 'mono', rate: int = 16000) -> np.ndarray:

    container = av.open(file)
    resampler = AudioResampler(format=format, layout=layout, rate=rate)

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

        # Flush resampler tail to avoid dropping samples at the end
        tail = resampler.resample(None)
        if tail is not None:
            frames = tail if isinstance(tail, (list, tuple)) else [tail]
            for f in frames:
                arr = f.to_ndarray()

                if arr.ndim == 2:
                    if arr.shape[0] == 1:
                        arr = arr[0]
                    elif arr.shape[1] == 1:
                        arr = arr[:, 0]
                    else:
                        arr = arr.mean(axis=0)
                elif arr.ndim > 2:
                    arr = arr.reshape(-1)

                if arr.dtype != np.float32:
                    arr = (arr.astype(np.float32) / 32768.0).clip(-1.0, 1.0)

                chunks.append(arr)
    finally:
        container.close()

    if not chunks:
        return np.zeros((0,), dtype=np.float32)

    audio = np.concatenate(chunks, axis=0).astype(np.float32, copy=False)
    return audio

def seek_audio(
    file: daft.File,
    *,
    start_sec: float = 0.0,
    end_sec: float = float("inf"),
    format: str = 's16',
    layout: str = 'mono',
    rate: int = 16000,
    options: dict | None = None,
    ):

    eps = 1e-6
    resampler = AudioResampler(format=format, layout=layout, rate=rate)

    if not options:
        options = {
            "probesize": str("64k"),
            "analyzeduration": 200_000,
        }

    with av.open(file, mode="r", options=options, metadata_encoding="utf-8") as container:
        

        # Select video streams 
        stream = next((s for s in container.streams if s.type == "audio"), None)
        stream.thread_type = "AUTO"

        # Compute seek position and optional end bound in this stream's ticks
        ts_start = int(start_sec / float(stream.time_base)) if start_sec > 0 else 0
        end_pts = None if end_sec == float("inf") else int(end_sec / float(stream.time_base))

        container.seek(ts_start, stream=stream, any_frame=False, backward=True)

        


        # Decode frames only from this stream
        for frame in container.decode(stream):
            # Skip if no PTS
            if frame.pts is None:       
                continue

            # Skip if before start_sec
            t = frame.pts * float(stream.time_base)
            if t + eps < start_sec:
                continue

            # Break if after end_sec
            if end_pts is not None and frame.pts > end_pts:
                break

            # Resample
            frame_resampled = resampler.resample(frame)

            payload = frame.to_ndarray(format="rgb24").tobytes()
            payload_size_bytes = len(payload)
            

            yield {
                
            }


# Parakeet Transcribe with Timestamps
@daft.udf(return_dtype = dt.struct({
    "segment": dt.list(dt.struct({
        "start_offset": dt.int32(),
        "end_offset": dt.int32(),
        "start": dt.float32(),
        "end": dt.float32()
    })),
}))
class ParakeetTranscribeTimestampsUDF:
    def __init__(self, context_size: int = 256):
        import nemo.collections.asr as nemo_asr
        self.asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name="nvidia/parakeet-tdt-0.6b-v3")
        self.asr_model.change_attention_model(
            self_attention_model="rel_pos_local_attn",
            att_context_size=[context_size, context_size]
        )

    def __call__(self, audio: list[np.ndarray]):
        outputs = self.asr_model.transcribe(audio, timestamps=True)   # No public flag to emit only segments
        return [o.timestamp["segment"] for o in outputs]


if __name__ == "__main__":
    uri = "../videoprism/videoprism/assets/*.mp4"
    B, T, H, W, C = 2, 16, 288, 288, 3 # Batch Size, Clip Size (# frames), Height, Width, RGB
    ROW_LIMIT = 500
    eager = True
    interp = None
    max_video_seek_read_duration = 10.0 # How wide of a batch to read from the video at a time

    start = time.time()

    df_files = (
        daft.from_glob_path(uri)
        .with_column("file", file(col("path")))
    )

    df_audio = df_files.with_column("audio", extract_audio_clips(col("file")))

    df_ts = df_audio