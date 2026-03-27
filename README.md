# Parallel Finder

A tool I made for video editors who are tired of manually finding repeated movements, similar shots, and parallels in their footage.

I spent a lot of time scrubbing through timelines trying to find where the same action happens twice — a dance move, a stunt, a gesture. It's easy to spot, but it takes forever. So I built this to do it automatically.

## What it does

- Analyzes videos and detects human poses using YOLOv8
- Finds similar movements across one or multiple video files
- Shows you exactly where the matches are on a timeline
- Exports results in JSON, TXT, CSV, or EDL

## How it works

The program extracts pose data from each frame, then uses FAISS to quickly find potential matches. A temporal comparison with Soft-DTW filters out false positives, and optional neural verification (via Ollama) can further improve accuracy.

It runs on GPU if available, but works fine on CPU too.

## Features

- Works with MP4, MOV, MKV, AVI, and most other video formats
- Adjustable similarity thresholds to control how strict the search is
- Timeline view with heatmap visualization
- Cache system so you don't have to re-analyze the same video twice
- Export match lists to use in editing software

## Requirements

- Windows 7, 8, 10, or 11
- NVIDIA GPU optional (but recommended for speed)
- Python 3.10 if running from source

## Download

You can find compiled releases in the Releases section, or clone the repo and run it directly.

## Credits

Built by @Pozit1vchicc

YOLOv8-pose for pose detection, FAISS for similarity search, PySide6 for the interface.

## License

MIT — free to use, modify, share. Just keep the credits.

If this tool saves you time, that's all the payment I need.
