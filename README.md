# Parallel Finder

Parallel Finder is a tool for video editors who want to quickly find repeated movements, similar shots, and visual parallels across their footage.

I built it because manually searching through timelines for the same action happening more than once — a dance move, gesture, stunt, or similar composition — is slow, repetitive, and frustrating. This tool automates that process and helps surface those moments much faster.

## What it does

- Analyzes videos and detects human poses using YOLOv8 Pose
- Finds similar movements across one or multiple video files
- Displays matches on a timeline for quick navigation
- Exports results in JSON, TXT, CSV, or EDL formats

## How it works

The application extracts pose data frame by frame, then uses FAISS to quickly search for candidate matches. After that, temporal comparison with Soft-DTW helps filter out false positives. Optional neural verification via Ollama can be used for additional refinement.

It can run on GPU for faster processing, but also works on CPU.

## Features

- Supports MP4, MOV, MKV, AVI, and most common video formats
- Adjustable similarity thresholds for more strict or more flexible matching
- Timeline view with heatmap-style visualization
- Caching system to avoid re-analyzing the same footage
- Export tools for use in editing workflows

## Requirements

- Windows 7, 8, 10, or 11
- NVIDIA GPU optional, but recommended for better performance
- Python 3.10+ if running from source

## Installation

Prebuilt installers are available in the **Releases** section.

If you prefer to run the project from source, clone the repository and install the required dependencies manually.

## Notes

- CPU mode works on most systems, but analysis will be slower
- GPU mode requires a compatible NVIDIA GPU and suitable drivers
- If you are unsure which version to install, choose the CPU option

## Credits

Created by **@Pozit1vchicc**

Powered by:
- **YOLOv8 Pose** for pose detection
- **FAISS** for similarity search
- **Tkinter** for the interface

## License

MIT License — free to use, modify, and share. Please keep the original credits.

If this tool saves you time, that's already the best reward.