# Video Embeddings

- Date: Aug 25, 2025
- Author: Everett Kleven
- Size: L
- Persona: VideoType Discussion Participant


Notebooks:
  - from_video_frames to video embeddings
  - end to end video processing from file

Scripts: 

## Abstract

Video processing is the next frontier for multimodal ai workloads. Video combines image, audio, and temporal data into a single file, which when read together, quickly exhaust in-memoryu resources. Most processing strategies leverage streaming to minimize memory overhead contrasting with traditional ETL pipelines.

## Overview

This week I focused on the pains expressed in the [VideoType discussion](https://github.com/Eventual-Inc/Daft/discussions/5054), namely:

VideoType Pipelines should:

- *"avoid storing the entire dataset in memory"*
- *"it is critical to extract key metadata to facilitate subsequent filtering of target videos prior to processing"*
- *"prioritize including only essential metadata fields—specifically frame count, height, width, and FPS"*
- *"Additional metadata can be dynamically retrieved during video processing as needed"*

Video Data Functions and UDFs

- *key_frames()* - extracting key frames
- *split_video()* - splitting videos by key frames 
- *audio()* - extracting audio

In addition to these focus areas, I also explored friction points in video ai pipelines such as: 

- generating embeddings for images, audio, and video using google/embedding-gemma, nvidia/parakeet, and google/videoprism respectively.
- shot boundary detection using embeddings (keyframe detection for video segmentation)
- concurrent reads with video seeking.
- Extract keyframe pts, batch on predined duration with predefined sizing.

1. Exploring what it looks like if it's prohibitive to load all frames into memory
2. Reading Audio from videos and using timestamped transcription for segmentation
3. Image Embedding based shot boundary detection
4. Video Embeddings for Video Understanding

## Summary

Video processing is hard.

`read_video_frames()` is convenient and well built, but could use some enrichment. 

 use case of reading images to a row limit and generating video embeddings on 16 frame clips, I was able to get the happy path working within a few work sessions. Once I faced the prospect of video segmentation and seeking to concurrently read videos with daft.File things became more complicated.

Window functions collapses into a fully daft native pipeline, so long as you are ok with reading all video data sequentially with a read_video_frames.

What makes video processing particularly complex isn't just memory management, but the number of early decisions an engineer has to commit to when designing their workload. While my particular workload of video embeddings is straightforward, if I were building the pipeline for a more specific downstream task, I may implement things very differently.

Streaming frames from a generator represents a fundamentally different mindset from traditinoal ETL pipelines. Sure we can limit the number of rows we receive, but defining the actual mechanism of throttling memory overhead is 

It can be overwhelming to consider the various permutations of video processing approaches, especially concerning ingestion and segmentation. Inference is where the problem becomes more concrete, but if you have multiple downstream AI/ML tasks with different batching requirements things can get hairy quickly. This leads us to wan't to canonicalize our preprocessing stages into a standard form that can then be repackaged and shaped downstream. 

### Ingestion 

1. read_video_frames - which decodes video frames into images and stores them as rows against a frame index
2. probe_video_metadata() + read_video_file(...,hist,sbd,audio) - which probes for metadata as a "cheap" pass, enabling early content filtering, then opening the video file with enriched inputs for extracting image histograms, shot boundary flag, and audio frames. Naturally the audio reading can be broken out into a seperate function entirely, but I'm including it here for brevity.
3. probe_video_metadata() + seek_video_file(...,hist,sbd,audio) - same as above, except distribute reads reading each video file concurrently from pre-planned frame timestamps.

### Segementation - Clips & Shot Boundary Detection 

Segmentation in particular presents the problem or chunking your video into semantic pieces. Most downstream ai/ml tasks require samples in clips, usually on the order of 16 frame batches, any operation that occurs outside the clip context requires an additional groupby/explode.

Shot boundary detection and other video segmentation strategies incentivize early preprocessing during frame decoding at the file level.
File seeking can help parallelize reads, early computations like histograms and chi-squared distance are more convenient prior to dataframe ingestion. 
