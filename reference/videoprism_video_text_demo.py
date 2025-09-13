from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

import jax
import jax.numpy as jnp
import mediapy
import numpy as np

from videoprism import models as vp
import videoprism as vp_pkg


def read_and_preprocess_video(
    filename: str, target_num_frames: int, target_frame_size: tuple[int, int]
):
    """Reads and preprocesses a video.

    - Uniformly samples to target_num_frames
    - Resizes frames to target_frame_size (H, W)
    - Normalizes pixel values to [0.0, 1.0]
    """

    frames = mediapy.read_video(filename)

    # Sample to target number of frames.
    frame_indices = np.linspace(
        0, len(frames), num=target_num_frames, endpoint=False, dtype=np.int32
    )
    frames = np.array([frames[i] for i in frame_indices])

    # Resize to target size.
    original_height, original_width = frames.shape[-3:-1]
    target_height, target_width = target_frame_size
    assert (
        original_height * target_width == original_width * target_height
    ), "Currently does not support aspect ratio mismatch."
    frames = mediapy.resize_video(frames, shape=target_frame_size)

    # Normalize pixel values to [0.0, 1.0].
    frames = mediapy.to_float01(frames)

    return frames


def compute_similarity_matrix(
    video_embeddings: Iterable[np.ndarray],
    text_embeddings: Iterable[np.ndarray],
    temperature: float | None,
    apply_softmax: str | None = None,
) -> np.ndarray:
    """Computes cosine similarity matrix (dot-product assuming normalized inputs).

    apply_softmax: one of None, 'over_texts', 'over_videos'
    """
    assert apply_softmax in [None, "over_texts", "over_videos"]
    emb_dim = video_embeddings[0].shape[-1]
    assert emb_dim == text_embeddings[0].shape[-1]

    video_embeddings = np.array(video_embeddings).reshape(-1, emb_dim)
    text_embeddings = np.array(text_embeddings).reshape(-1, emb_dim)
    similarity_matrix = np.dot(video_embeddings, text_embeddings.T)

    if temperature is not None:
        similarity_matrix = similarity_matrix / temperature

    if apply_softmax == "over_videos":
        similarity_matrix = np.exp(similarity_matrix)
        similarity_matrix = similarity_matrix / np.sum(
            similarity_matrix, axis=0, keepdims=True
        )
    elif apply_softmax == "over_texts":
        similarity_matrix = np.exp(similarity_matrix)
        similarity_matrix = similarity_matrix / np.sum(
            similarity_matrix, axis=1, keepdims=True
        )

    return similarity_matrix


def get_default_asset_video() -> str:
    """Returns absolute path to the bundled demo video in the videoprism package."""
    pkg_dir = Path(vp_pkg.__file__).resolve().parent
    candidate = pkg_dir / "assets" / "water_bottle_drumming.mp4"
    return str(candidate)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "VideoPrism video-text retrieval demo (no dependency setup). "
            "Loads a pre-trained model, computes embeddings, and prints retrieval scores."
        )
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="videoprism_lvt_public_v1_base",
        choices=[
            "videoprism_lvt_public_v1_base",
            "videoprism_lvt_public_v1_large",
        ],
        help="Pretrained VideoPrism video-text model to use.",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=16,
        help="Number of frames to sample from the video.",
    )
    parser.add_argument(
        "--frame_size",
        type=int,
        default=288,
        help="Square frame size (pixels) to resize to (H=W=frame_size).",
    )
    parser.add_argument(
        "--video",
        type=str,
        default=get_default_asset_video(),
        help=(
            "Path to an input video file. Defaults to the bundled demo asset from the videoprism package."
        ),
    )
    parser.add_argument(
        "--text_queries",
        type=str,
        default="playing drums,sitting,playing flute,playing at playground,concert",
        help="Comma-separated list of text queries (without the prompt template).",
    )
    parser.add_argument(
        "--prompt_template",
        type=str,
        default="a video of {}.",
        help="Prompt template applied to each text query.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.01,
        help="Temperature to scale similarities with before softmax (if any).",
    )
    parser.add_argument(
        "--softmax",
        type=str,
        default="over_texts",
        choices=["none", "over_texts", "over_videos"],
        help="Optional softmax normalization over scores.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="If set, attempts to display the input video (may require GUI).",
    )
    parser.add_argument(
        "--print_jax_info",
        action="store_true",
        help="If set, prints JAX version, platform, and device count.",
    )
    args = parser.parse_args()

    if args.print_jax_info:
        from jax.extend import backend

        print(f"JAX version:  {jax.__version__}")
        print(f"JAX platform: {backend.get_backend().platform}")
        print(f"JAX devices:  {jax.device_count()}")

    model_name = args.model_name
    num_frames = int(args.num_frames)
    frame_size = int(args.frame_size)
    video_path = args.video
    text_query_csv = args.text_queries
    prompt_template = args.prompt_template
    temperature = float(args.temperature) if args.temperature is not None else None
    apply_softmax = None if args.softmax == "none" else args.softmax

    if not os.path.exists(video_path):
        raise FileNotFoundError(
            f"Video file not found: {video_path}. Provide --video with a valid path."
        )

    # Load model and tokenizer
    flax_model = vp.get_model(model_name)
    loaded_state = vp.load_pretrained_weights(model_name)
    text_tokenizer = vp.load_text_tokenizer("c4_en")

    @jax.jit
    def forward_fn(inputs, text_token_ids, text_paddings, train: bool = False):
        return flax_model.apply(
            loaded_state,
            inputs,
            text_token_ids,
            text_paddings,
            train=train,
        )

    # Prepare inputs
    frames_np = read_and_preprocess_video(
        filename=video_path,
        target_num_frames=num_frames,
        target_frame_size=(frame_size, frame_size),
    )
    frames = jnp.asarray(frames_np[None, ...])  # Add batch dimension.

    text_queries = [q.strip() for q in text_query_csv.split(",") if q.strip()]
    text_queries = [prompt_template.format(q) for q in text_queries]
    text_ids, text_paddings = vp.tokenize_texts(text_tokenizer, text_queries)

    print("Input text queries:")
    for i, text in enumerate(text_queries):
        print(f"({i + 1}) {text}")

    # Forward pass
    video_embeddings, text_embeddings, _ = forward_fn(
        frames, text_ids, text_paddings
    )

    # Similarity and retrieval
    similarity_matrix = compute_similarity_matrix(
        video_embeddings, text_embeddings, temperature=temperature, apply_softmax=apply_softmax
    )

    v2t_similarity_vector = similarity_matrix[0]
    top_indices = np.argsort(v2t_similarity_vector)[::-1]

    print(f"\nQuery video: {os.path.basename(video_path)}")
    if args.show:
        try:
            mediapy.show_video(frames_np, fps=6.0)
        except Exception as exc:  # noqa: BLE001
            print(f"Warning: Could not display video ({exc}). Continuing.")

    for k, j in enumerate(top_indices):
        print(
            "Top-%d retrieved text: %s [Similarity = %0.4f]"
            % (k + 1, text_queries[j], v2t_similarity_vector[j])
        )
    print(f"\nThis is {text_queries[top_indices[0]]}")


if __name__ == "__main__":
    main()


