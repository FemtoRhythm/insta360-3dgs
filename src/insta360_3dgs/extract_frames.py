# -*- coding: utf-8 -*-
"""从 360° 等矩形全景视频抽帧（调 ffmpeg）。"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from . import config


def extract_frames(input_video: Path, output_dir: Path, fps: float, width: int) -> Path:
    frames_dir = config.output_layout(output_dir)["frames"]
    frames_dir.mkdir(parents=True, exist_ok=True)
    if any(frames_dir.glob("frame_*.jpg")):
        print(f"[extract] 跳过已存在帧: {frames_dir}", flush=True)
        return frames_dir

    height = width // 2  # 等矩形约束: h = w / 2
    pattern = str(frames_dir / "frame_%04d.jpg")
    command = [
        "ffmpeg", "-y", "-i", str(input_video),
        "-vf", f"fps={fps},scale={width}:{height}",
        "-q:v", "2", pattern,
    ]
    print(f"[extract] {' '.join(command)}", flush=True)
    subprocess.run(command, check=True)
    print(f"[extract] done -> {frames_dir}", flush=True)
    return frames_dir


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="insta360-3dgs extract",
                                     description="从 360° 全景视频抽帧")
    parser.add_argument("--input", required=True, help="360° 等矩形全景视频 (mp4)")
    parser.add_argument("--output", required=True, help="输出目录")
    parser.add_argument("--fps", type=float, default=0.5, help="抽帧频率 (帧/秒)")
    parser.add_argument("--width", type=int, default=config.PANORAMA_WIDTH, help="全景帧宽度")
    args = parser.parse_args(argv)
    extract_frames(Path(args.input), Path(args.output), args.fps, args.width)


if __name__ == "__main__":
    main()
