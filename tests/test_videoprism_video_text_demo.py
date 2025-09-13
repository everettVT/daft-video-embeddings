from __future__ import annotations

import os
import sys
import subprocess
from pathlib import Path

import pytest


def test_videoprism_video_text_demo_runs_on_cpu():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "reference" / "videoprism_video_text_demo.py"
    asset_path = repo_root / "videoprism" / "videoprism" / "assets" / "water_bottle_drumming.mp4"

    if not script_path.exists():
        pytest.skip(f"Demo script not found at {script_path}")
    if not asset_path.exists():
        pytest.skip(f"Demo asset not found at {asset_path}")

    env = os.environ.copy()
    # Force CPU for stability across environments
    env["JAX_PLATFORMS"] = "cpu"
    env["JAX_PLATFORM_NAME"] = "cpu"
    # Avoid excessive TF logging if TF gets imported indirectly
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    cmd = [
        sys.executable,
        str(script_path),
        "--video",
        str(asset_path),
        "--num_frames",
        "8",
        "--frame_size",
        "288",
        "--softmax",
        "over_texts",
    ]

    result = subprocess.run(
        cmd,
        cwd=str(repo_root),
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )

    if result.returncode != 0:
        print("STDOUT:\n" + result.stdout)
        print("STDERR:\n" + result.stderr)
    assert result.returncode == 0, "Demo script failed to run successfully"
    assert "This is" in result.stdout, "Did not find expected completion message in output"


