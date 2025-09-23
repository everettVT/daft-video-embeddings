# Daft Video Embeddings

*Canonical Workload Sandbox for Foundational Video Understanding*

---

This repository contains notebooks and scripts exploring friction points in crafting video ai workloads with daft. This repo contains scripts and notebooks that are 80-99% complete, but in are left due to reprioritizations. At the end of the day, the purpose was to surface friction points and that is summarized in the Friction Logs. There is an AI synthesized version and two more messy human versions for your perusal. 

Install

```bash
uv venv && make uv-sync
```


## Core Artifacts

### `/workloads`

- Shot Boundary Detection with google/siglip2 image embeddings [[Notebook](/workload/notebooks/sbd_window_siglip2.ipynb), [Script](/workload/sbd_siglip2.py)]
- Transcribing Videos to Segment into Short Form Content [[Notebook](/workload/notebooks/video_transcription_segmentation.ipynb), [Script](/workload/video_transcript_segmentation.py)]
- Generating Video Embeddings with google/videoprism [[Notebook](/workload/notebooks/videoprism.ipynb), [Script](/workload/videoprism.py)]
- Seek-based concurrent reads on videos [Script](/workload/video_seeking.py)

