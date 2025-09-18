import daft
from daft import col, DataType as dt
import numpy as np
import jax
import jax.numpy as jnp
from jax.extend import backend
import tensorflow as tf
from videoprism import models as vp
import av
import subprocess
import json
from fractions import Fraction

@daft.func(return_dtype = dt.struct({
    "width": dt.int32(),
    "height": dt.int32(),
    "fps": dt.float64(),
    "frame_count": dt.int32(),
    "time_base": dt.float64(),
    "json": dt.string()
}))
def get_video_metadata(file: daft.File, ):
    "Explicitly tracks width, height, fps, frame_count, and time_base from ffprobe. Everything else dumped as json."
    
    with file:

        out = subprocess.check_output([
            "ffprobe","-v","error",
            "-select_streams","v:0",
            "-show_streams","-show_format",
            "-print_format","json", str(file)
        ])
        info = json.loads(out)

        # Cast ffprobe output to our dtype
        w = info["streams"][0]["width"]
        h = info["streams"][0]["height"]
        fps = float(info["streams"][0]["r_frame_rate"])
        nb_frames = info["streams"][0]["nb_frames"]
        time_base = float(info["streams"][0]["time_base"])

        return {
                "width": w,
                "height": h,
                "fps": fps,
                "frame_count": nb_frames,
                "time_base": time_base,
                "json": json.dumps(info),
            }

                



def get_keyframe_pts(path):
    pts = []
    with av.open(path) as c:
        vs = c.streams.video[0]
        tb = float(vs.time_base)
        for f in c.decode(video=0):
            if f.pts is None: 
                continue
            if getattr(f, "key_frame", False) or getattr(f, "pict_type", None) == "I":
                pts.append(f.pts * tb)  # seconds
    return pts