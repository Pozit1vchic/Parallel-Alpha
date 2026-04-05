# Parallel Finder

Parallel Finder is a tool for video editors who want to quickly find repeated movements, similar shots, and visual parallels across their footage.

I built it because manually searching through timelines for matching moments — a repeated dance move, gesture, stunt, or similar composition — is slow, tedious, and frustrating. This tool automates that search and helps surface those moments in a fraction of the time.

---

## What it does

- Analyzes video files and detects human poses frame by frame using YOLOv8 Pose
- Finds similar movements across one or multiple video files
- Displays matches on an interactive timeline with heatmap-style visualization
- Exports results in JSON, TXT, CSV, or EDL formats for use in editing workflows

---

## How it works

The application extracts pose keypoints frame by frame, then uses FAISS to quickly search for candidate matches across the footage. Temporal comparison with Soft-DTW filters out false positives and improves result quality. Optional neural verification via Ollama can be added for additional refinement.

Processing runs on GPU for maximum speed, but CPU mode is fully supported as well.

---

## Current state

This is an **alpha project**. It works, it saves real time, and it is actively being developed — but it is not finished.

A few things worth knowing before you use it:

- Out of roughly 90 candidate matches the program finds, around **20–30 tend to be genuinely useful**, depending on the video. The rest are false positives. Rebalancing detection accuracy is an ongoing priority.
- **Search by photo is currently unreliable.** The logic exists but produces poor results in most cases. Avoid using it for real work — it will be improved in a future version.
- Results quality varies depending on video resolution, lighting conditions, and how distinctly the movements are performed.

Think of it as a **search assistant** that dramatically narrows down what you need to review manually — not a fully automated solution.

---

## What's in Alpha v13

- Rebalanced detection — fewer irrelevant results, better signal-to-noise ratio
- Redesigned interface with a cleaner, more consistent layout
- Noticeably faster startup and processing
- Full bilingual support — Russian and English interface
- Improved results list with better sorting and readability
- Model selection directly inside the application — switch YOLO models without reinstalling
- More stable and informative installer

---

## Features

- Supports MP4, MOV, MKV, AVI, and most common video formats
- Adjustable similarity thresholds for stricter or more flexible matching
- Timeline view with heatmap-style match visualization
- Caching system to avoid re-analyzing the same footage
- Export tools compatible with common editing workflows
- Bilingual interface — Russian and English

---

## Requirements

- Windows 7, 8, 10, or 11 (64-bit)
- Python 3.10, 3.11, or 3.12
- 8 GB RAM minimum — 16 GB recommended
- NVIDIA GPU optional, but recommended for significantly faster processing

---

## Installation

Prebuilt installers are available in the **Releases** section.

The installer will guide you through selecting a Torch backend (CPU, CUDA 11.8, or CUDA 12.8) and choosing where dependencies are installed. Use paths with Latin characters only — Cyrillic characters or spaces in folder paths will cause the installation to fail.

Dependency installation takes around 5–20 minutes depending on your internet connection.

If you prefer to run from source, clone the repository and install dependencies manually using the provided `requirements.txt`.

---

## Notes

- First launch may take longer while models are downloaded
- GPU mode is significantly faster for longer or higher-resolution videos
- FAISS is installed in CPU mode on Windows for stability
- If installation fails, the most common cause is a path containing Cyrillic characters or spaces

---

## Credits

Created by **[@Pozit1vchicc](https://t.me/Pozit1vchicc)**

Powered by:
- **YOLOv26 Pose** — pose detection
- **FAISS** — similarity search
- **Soft-DTW** — temporal movement comparison
- **Tkinter** — interface

---

## License

MIT — free to use, modify, and share. Please keep the original credits.

If this tool saves you time in the edit, that's already the best outcome.
