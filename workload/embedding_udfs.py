import numpy as np
import daft
import jax 
import jax.numpy as jnp
from daft import col, DataType as dt



video_embedding_schema = daft.Schema.from_pydict({
    "frame_index": dt.int64(),
    "image": dt.image(),
    "audio": dt.audio(),
    "embedsiglip2_": dt.embedding(dt.float32(), 768),
    "audio_siglip2_": dt.embedding(dt.float32(), 768),
    "video_siglip2_": dt.embedding(dt.float32(), 768),
    "video_embeddings": dt.embedding(dt.float32(), 768),
    "audio_embeddings": dt.embedding(dt.float32(), 768),
    "image_embeddings": dt.embedding(dt.float32(), 768),
    "video_embeddings": dt.embedding(dt.float32(), 768),
}
    image = dt.image(),
    audio = dt.audio(),
    
)



def video_frame_to_daft_image(frame: av.VideoFrame) -> daft.Image:
    return daft.from_pframe.to_ndarray(format="rgb24"))

def audio_frame_to_daft_audio(frame: av.AudioFrame) -> daft.Audio:
    return daft.Audio.from_numpy(frame.to_ndarray())

def video_to_daft_video(video: av.VideoFrame) -> daft.Video:
    return daft.Video.from_numpy(video.to_ndarray())


class EmbeddingPreprocessor:
    "Common Preprocessing for Image, Audio, and Video Embeddings"
    def image_to_numpy(self, image: dt.image()) -> np.ndarray:

    def audio_to_numpy(self, audio: dt.audio()) -> np.ndarray:
    









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