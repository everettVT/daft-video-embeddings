import daft
from daft import col, DataType as dt
import numpy as np
import jax
import jax.numpy as jnp


@daft.func(return_dtype=dt.tensor(dt.float32(), shape=(16,288, 288, 3)))
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


    return x # [1,T,H,W,C] where T=clip_size


@daft.func(return_dtype= dt.tensor(dt.int64()))
def histogram(array: np.ndarray, bins: int = 256):
    # Accept (H,W,3) or (N,3); flatten spatial dims if needed
    if array.ndim == 3 and array.shape[-1] == 3:
        flat = array.reshape(-1, 3)
    elif array.ndim == 2 and array.shape[-1] == 3:
        flat = array
    else:
        flat = np.asarray(array).reshape(-1, 3)

    # If image is normalized [0,1], set range accordingly; else fallback to [0,255]
    # Heuristic: values > 1 imply [0,255]
    value_range = (0.0, 1.0) if float(np.max(flat)) <= 1.0 else (0.0, 255.0)

    hist = np.zeros((3, bins), dtype=np.int64)
    for i in range(3):
        h, _ = np.histogram(flat[:, i], bins=bins, range=value_range)
        hist[i] = h.astype(np.int64, copy=False)
    return hist


@daft.func(return_dtype=dt.list(dt.int64()))
def detect_sbd_chi_squared_distance(
    hists: list[np.ndarray], 
    indices: list[int], 
    pts: list[int],
    threshold: float = 0.3, 
    min_shot_duration: float = 0.5, # seconds
    ) -> list[int]:

    # Validate inputs
    if len(hists) < 2:
        return []

    # Convert to numpy arrays
    h_arr = [np.asarray(h, dtype=np.float32) for h in hists]
    idx_arr = np.asarray(indices, dtype=np.int64)
    pts_arr = np.asarray(pts, dtype=np.int64)

    # Ensure monotonic increasing frame indices
    order = np.argsort(idx_arr)
    h_arr = [h_arr[i] for i in order]
    idx_arr = idx_arr[order]
    pts_arr = pts_arr[order]

    # Chi-squared distance between consecutive frames (channel-averaged)
    eps = 1e-8
    dists = []
    for i in range(1, len(h_arr)):
        h1 = h_arr[i-1]
        h2 = h_arr[i]
        if h1.ndim == 1:
            h1 = h1[None, :]
        if h2.ndim == 1:
            h2 = h2[None, :]
        # Normalize per-channel to probabilities
        h1n = h1 / (np.sum(h1, axis=1, keepdims=True) + eps)
        h2n = h2 / (np.sum(h2, axis=1, keepdims=True) + eps)
        num = (h1n - h2n) ** 2
        den = h1n + h2n + eps
        chisq_per_channel = 0.5 * np.sum(num / den, axis=1)
        d = float(np.mean(chisq_per_channel))
        dists.append(d)
    dists = np.asarray(dists, dtype=np.float32)  # length N-1, aligned to transition into frame i
    smoothed = dists

    # Threshold with minimum shot duration constraint
    min_us = int(min_shot_duration * 1_000_000.0)
    boundaries: list[int] = []
    last_boundary_pts = None

    # dists[i-1] corresponds to transition between frames (i-1)->i; use frame i's index/time
    for i in range(1, len(idx_arr)):
        if smoothed[i-1] > threshold:
            cand_pts = int(pts_arr[i])
            if last_boundary_pts is None or (cand_pts - last_boundary_pts) >= min_us:
                boundaries.append(int(idx_arr[i]))
                last_boundary_pts = cand_pts

    return boundaries

@daft.func(return_dtype=dt.list(dt.int64()))
def detect_sbd_chi_squared_hysteresis(
    hists: list[np.ndarray],
    indices: list[int],
    pts: list[int],
    high_threshold: float = 0.4,
    low_threshold: float = 0.25,
    min_shot_duration: float = 0.5,  # seconds
    ) -> list[int]:

    if len(hists) < 2:
        return []

    # Validate thresholds
    if not (low_threshold < high_threshold):
        # Swap to enforce proper ordering if misconfigured
        low_threshold, high_threshold = min(low_threshold, high_threshold), max(low_threshold, high_threshold)

    # Convert and sort by frame index
    h_arr = [np.asarray(h, dtype=np.float32) for h in hists]
    idx_arr = np.asarray(indices, dtype=np.int64)
    pts_arr = np.asarray(pts, dtype=np.int64)
    order = np.argsort(idx_arr)
    h_arr = [h_arr[i] for i in order]
    idx_arr = idx_arr[order]
    pts_arr = pts_arr[order]

    # Compute per-transition chi-squared distances
    eps = 1e-8
    dists = []
    for i in range(1, len(h_arr)):
        h1 = h_arr[i-1]
        h2 = h_arr[i]
        if h1.ndim == 1:
            h1 = h1[None, :]
        if h2.ndim == 1:
            h2 = h2[None, :]
        h1n = h1 / (np.sum(h1, axis=1, keepdims=True) + eps)
        h2n = h2 / (np.sum(h2, axis=1, keepdims=True) + eps)
        num = (h1n - h2n) ** 2
        den = h1n + h2n + eps
        chisq_per_channel = 0.5 * np.sum(num / den, axis=1)
        d = float(np.mean(chisq_per_channel))
        dists.append(d)
    dists = np.asarray(dists, dtype=np.float32)

    # Hysteresis scan: choose the local peak within a high-threshold crossing, end when below low-threshold
    boundaries: list[int] = []
    min_us = int(min_shot_duration * 1_000_000.0)
    last_boundary_pts = None

    inside_cluster = False
    peak_dist = -1.0
    peak_i = -1

    for i in range(1, len(idx_arr)):
        d = dists[i-1]
        if not inside_cluster:
            if d >= high_threshold:
                inside_cluster = True
                peak_dist = d
                peak_i = i
        else:
            # Track peak while in cluster
            if d > peak_dist:
                peak_dist = d
                peak_i = i
            # Exit when we drop below low threshold; commit the peak as boundary
            if d <= low_threshold:
                cand_pts = int(pts_arr[peak_i])
                if last_boundary_pts is None or (cand_pts - last_boundary_pts) >= min_us:
                    boundaries.append(int(idx_arr[peak_i]))
                    last_boundary_pts = cand_pts
                inside_cluster = False
                peak_dist = -1.0
                peak_i = -1

    # If we ended still inside a cluster, commit the peak
    if inside_cluster and peak_i >= 0:
        cand_pts = int(pts_arr[peak_i])
        if last_boundary_pts is None or (cand_pts - last_boundary_pts) >= min_us:
            boundaries.append(int(idx_arr[peak_i]))

    return boundaries

def detect_shot_boundaries(h1: np.ndarray, h2: np.ndarray, threshold: float = 0.3) -> bool:
    # Chi-square distance over channel-wise normalized histograms

    h1 = np.asarray(h1, dtype=np.float32)
    h2 = np.asarray(h2, dtype=np.float32)
    if h1.ndim == 1:
        h1 = h1[None, :]
    if h2.ndim == 1:
        h2 = h2[None, :]
    # Normalize per-channel to probabilities
    eps = 1e-8
    h1n = h1 / (np.sum(h1, axis=1, keepdims=True) + eps)
    h2n = h2 / (np.sum(h2, axis=1, keepdims=True) + eps)
    num = (h1n - h2n) ** 2
    den = h1n + h2n + eps
    chisq_per_channel = 0.5 * np.sum(num / den, axis=1)
    dist = float(np.mean(chisq_per_channel))
    return dist > threshold, dist

@daft.udf(return_dtype=dt.struct({
    "fr": dt.list(dt.float32()),
    "boundaries": dt.list(dt.bool()),
    "pair_indices": dt.list(dt.int64()),
}),
batch_size=80,
num_gpus=1,
)
def shot_boundary_detection(
    hist_list: list[np.ndarray],
    index_list: list[int],
    threshold: float = 0.3,
    min_shot_len: int = 6,
):
   
    # assert index list is sorted
    assert np.all(np.diff(index_list) > 0), "Index list must be sorted"
    order = np.argsort(np.asarray(index_list))
    hists = [np.asarray(hist_list[i]) for i in order]

    results = []

    for i in order:
        if i == 0:
            continue
        if i == len(order) - 1:
            results.append({
                "frame_id": index_list[i],
                "is_boundary": False,
                "chisq_dist": 0,
            })
            continue
        is_boundary, dist = detect_shot_boundaries(hists[i - 1], hists[i], threshold)
        results.append({
            "frame_id": index_list[i],
            "is_boundary": is_boundary,
            "chisq_dist": dist,
        })
    return results



    
def main(
    paths: list[str], 
    row_limit: int, 
    B: int, 
    T: int, 
    H: int, 
    W: int, 
    C: int, 
    bins: int,
    ) -> daft.DataFrame: 

    # start time tracking if needed

    # Read Video Frames
    df_frames = daft.read_video_frames(
            paths, 
            image_height=H, 
            image_width=W
    )
    df_frames = df_frames.with_column("image_fp32", col("data").apply(lambda x: np.asarray(x).astype(np.float32) / 255.0, return_dtype=dt.tensor(dt.float32())))
    df_frames.show()

    # Preprocess Frames to Numpy Arrays
    df_hist = df_frames.with_column("histogram", histogram(df_frames["image_fp32"], bins=bins))
    df_hist.show()

    # Detect Shot Boundaries
    df_sbd = (
        df_hist
        .groupby("path").agg_list("histogram", "frame_index", "frame_pts")
        .with_column(
            "shot_boundaries",
            detect_sbd_chi_squared_distance(
                col("histogram"),
                col("frame_index"),
                col("frame_pts"),
                threshold=0.3,
                min_shot_duration=0.5,
            ),
        )
        .with_column(
            "shot_boundaries_hyst",
            detect_sbd_chi_squared_hysteresis(
                col("histogram"),
                col("frame_index"),
                col("frame_pts"),
                high_threshold=0.4,
                low_threshold=0.25,
                min_shot_duration=0.5,
            ),
        )
    )
    df_sbd.show()

    # Stack Images into Clipss
    df = (
        df_hist
        .with_column("clip_index", col("frame_index") // T)
        .groupby("path","clip_index").agg_list("data","frame_index")
    )
    df.show()

    # Stack Clips into Clip Tensors
    df = df.with_column("clip", stack_clip(df["data"], df["frame_index"], clip_size=T))
    df.show()

    # Generate Video Embeddings
    df = df.with_column("video_embeddings", VideoPrismVideoUDF(df["clip"]))
    df.show()

    return df




if __name__ == "__main__":
    filenames = ['/Users/everett-founder/git/dream/daft-video-embeddings/videoprism/videoprism/assets/water_bottle_drumming.mp4']
    df = main(filenames, row_limit=10, B=2, T=16, W=288, H=288, C=3, bins=256)
    

    @daft.udf(
    return_dtype = dt.embedding(dt.float32(), 768),
    batch_size= 2, # clips per batch (tune for throughput)
    num_gpus=1,
    )
    class VideoPrismVideoUDF:
        def __init__(self, model_name: str = "videoprism_lvt_public_v1_base"):
            "for 'videoprism_lvt_public_v1_large', set T = 8"

            from videoprism import models as vp
            self.model = vp.get_model(model_name)
            self.params = vp.load_pretrained_weights(model_name)

            @jax.jit
            def vf_b(clips):  # [B,T,288,288,3] -> [B,D]
                v, _, _ = self.model.apply(
                    self.params,
                    clips,
                    None,
                    None,
                    train=False
                )
                return v

            self.vf_b = vf_b

            # Optional warmup can be added by caller if shapes are known

        def __call__(self,
            clips: list[np.ndarray], # List[T,H,W,C] of len B
        ):
            # Batch Inference
            xb = jnp.stack(clips, axis=0)    # [B,T,H,W,3]
            video_embeddings = self.vf_b(xb) # [B,768]
            np_emb = np.asarray(video_embeddings)  # Back to NumPy
            return [np_emb[i].tolist() for i in range(len(np_emb))]


