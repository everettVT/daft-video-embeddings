import daft
from daft import col, DataType as dt
import numpy as np
import jax
import jax.numpy as jnp
from jax.extend import backend
import tensorflow as tf
from videoprism import models as vp


@daft.func()
def to_float01(img: np.ndarray) -> dt.tensor(dt.float32(), shape=(288, 288, 3)):
    arr = np.asarray(img, dtype=np.float32)
    return arr / 255.0

@daft.func(return_dtype=dt.tensor(dt.float32()))
def pad_trim_and_stack(frames: list[np.ndarray]) -> np.ndarray:
    # frames: List[T] of (H,W,3) float32
    T = len(frames)
    if T >= target_num_frames:
        use = frames[:target_num_frames]
    else:
        pad = [frames[-1]] * (target_num_frames - T)
        use = frames + pad
    arr = np.stack(use, axis=0)  # (T,H,W,3)
    return arr

@daft.udf(
    return_dtype=dt.struct({"video_embed": dt.embedding(dt.float32(), 768)}, {"text_embed": dt.embedding(dt.float32(), 768)})
    batch_size = 16,
)
class VideoPrismUDF: 
    "This UDF uses performs the read_video_frames directly as opposed to the grouped approach"
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = vp.get_model(model_name)
        self.loaded_state = vp.load_pretrained_weights(model_name)

    def __call__(self,
        frame_group: daft.Series,
        frame_index: daft.Series,
        frames: np.ndarray,
        text_ids: np.ndarray | None,
        text_paddings: np.ndarray | None,
        ): 
        return self.model.apply(self.loaded_state, img, train=False)


@daft.func()
def embed_video(
    frames: list[np.ndarray], 
    text_ids: np.ndarray | None, text_paddings: np.ndarray | None):
    """This function performs the video prism forward pass on a list of frames using the grouped approach
    
    The v1 base model takes:
    - videos with shape (16, 288, 288) as inputs and outputs
    - embeddings with shape (batch_size, 4096, 768) which could be reshaped into
    (batch_size, 16, 16, 16, 768) for spatiotemporal representations. The input
    - videos should be normalized in [0.0, 1.0].

    Returns:
        video_embeddings: Output contrastive video embeddings of shape [B, D].
            None if `inputs` is None.
        text_embeddings: Output contrastive text embeddings of shape [B, D]. None
            if `text_token_ids` is None.
        outputs: A dictionary of additional outputs, including `spatial_features`
            of shape [B, T * N, D], `spatiotemporal_features` of shape [B, T * N,
            D], and `frame_embeddings` of shape [B, T, D]. Empty if
            `return_intermediate` is False.

    """


def main(paths: list[str], target_num_frames: int, text_queries: list[str] | None = None, topk: int = 5):
    
    # Read Video Frames
    df = daft.read_video_frames(
        paths,
        image_height=288,
        image_width=288,
    )
    df = df.with_column("data_f32_01", to_float01(col("data")))
    df_tall = df.with_column("group_index", col("frame_index") // target_num_frames)

    # 1) Group frames into clips
    df_grouped = (
        df_tall
        .with_column("group_index", col("frame_index") // target_num_frames)
        .groupby("path", "group_index")
        .agg_list("frame_index", "data_f32_01")
    )

    # 2) Pad/trim to exactly target_num_frames and stack to ndarray
    df = df.with_column("clip", pad_trim_and_stack(col("clip_frames")))

    # 3) VideoPrism encode via JAX UDF
    # Load model once (base variant)
    model_name = 'videoprism_lvt_public_v1_base'
    flax_model = vp.get_model(model_name)
    loaded_state = vp.load_pretrained_weights(model_name)

    @jax.jit
    def _vp_forward(video_inputs: jax.Array, text_ids: jax.Array | None, text_paddings: jax.Array | None):
        return flax_model.apply(loaded_state, video_inputs, text_ids, text_paddings, train=False)

    # Text embeddings if provided
    text_embeds_const = None
    if text_queries:
        tok = vp.load_text_tokenizer('c4_en')
        text_ids, text_pads = vp.tokenize_texts(tok, text_queries)
        # Dummy single-frame to get video output shape; we only need text embeddings here
        dummy_video = jnp.zeros((1, target_num_frames, 288, 288, 3), dtype=jnp.float32)
        _, text_embeds, _ = _vp_forward(dummy_video, jnp.asarray(text_ids), jnp.asarray(text_pads))
        text_embeds_const = np.asarray(text_embeds)

    # Store embeddings using Daft's embedding dtype for clarity/perf
    @daft.func(return_dtype=dt.embedding(dt.float32(), 768))
    def encode_clip(clip: np.ndarray) -> np.ndarray:
        # clip: (T,H,W,3) float32 in [0,1]
        vid = jnp.asarray(clip[None, ...], dtype=jnp.float32)  # (1,T,H,W,3)
        video_embeds, _, _ = _vp_forward(vid, None, None)
        # video_embeds: (1, D)
        return np.asarray(video_embeds)[0]

    df = df.with_column("video_embed", encode_clip(col("clip")))

    # 4) Similarity to text queries (if provided)
    if text_embeds_const is not None:
        text_embeds_broadcast = text_embeds_const

        @daft.func(return_dtype=dt.tensor(dt.float32()))
        def cosine_to_text(video_embed: np.ndarray) -> np.ndarray:
            v = video_embed / (np.linalg.norm(video_embed) + 1e-12)
            Tm = text_embeds_broadcast / (np.linalg.norm(text_embeds_broadcast, axis=-1, keepdims=True) + 1e-12)
            return v @ Tm.T  # (num_texts,)

        df = df.with_column("similarity", cosine_to_text(col("video_embed")))
        df = df.select("path", "group_index", "similarity")
        df = df.orderby(col("similarity").apply(lambda x: float(np.max(x))), desc=True).limit(topk)

    df.show()
    
    
def group_by_frames(df: daft.DataFrame, target_num_frames: int):
    return (
        df
        .select("frame_index")
        .with_column("group_index", col("frame_index") // target_num_frames)
        .groupby("group_index").agg_list("frame_index")
    )



if __name__ == "__main__":
    tf.config.set_visible_devices([], "GPU")
    tf.config.set_visible_devices([], "TPU")

    PATHS = ["/Users/everett-founder/Movies/digitlism.mp4"]
    NUM_FRAMES = 16
    FRAME_SIZE = 288
    MODEL_NAME = 'videoprism_lvt_public_v1_base' # alternatively 'videoprism_lvt_public_v1_large'

    main(
        paths = PATHS, 
        num_frames= NUM_FRAMES,
        
        text_queries=["a person walking", "a car driving"],
        topk=5
    )