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

