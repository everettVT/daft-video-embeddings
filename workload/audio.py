



@daft.func()
def extract_audio_clips(file: daft.File, start_sec: float, end_sec: float, num_frames: int = 16, ) -> np.ndarray:

    container = av.open(file)
    resampler = AudioResampler(format='s16', layout='mono', rate=16000)

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
