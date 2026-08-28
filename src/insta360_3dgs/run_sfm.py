# -*- coding: utf-8 -*-
"""步骤2: 等矩形 SfM — 用 COLMAP EQUIRECTANGULAR 相机模型重建真实相机轨迹。

每个全景帧用一个 EQUIRECTANGULAR 相机 (光心在球心) 建模, 得到平滑的相机运动。
这是正确还原拍摄空间的关键: 相比"把 6 个 cube 面当独立针孔相机"的朴素做法,
它避免了共光心面被误当自由相机而产生的伪基线。

用法:
    python -m insta360_3dgs sfm --output out/ [--threads 8]
输入:  <output>/frames/*.jpg (步骤1 产出)
输出:  <output>/sfm/0/{cameras.bin, images.bin, points3D.bin}

注意: 本阶段依赖 pycolmap, 由 CLI 通过 runtime python 重执行 (见 _dispatch.py)。
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from . import config

config.add_third_party_paths()
import pycolmap  # noqa: E402


def run_sfm(frames_dir: Path, sfm_dir: Path, num_threads: int) -> Path:
    sparse_dir = sfm_dir / "0"
    if (sparse_dir / "images.bin").exists():
        print(f"[sfm] 跳过已存在重建: {sparse_dir}", flush=True)
        return sparse_dir

    db_path = sfm_dir / "sfm.db"
    if db_path.exists():
        db_path.unlink()
    if sfm_dir.exists():
        shutil.rmtree(sfm_dir, ignore_errors=True)
    sfm_dir.mkdir(parents=True, exist_ok=True)

    # EQUIRECTANGULAR (model_id=17): 参数只有 w,h, 无焦距/主点
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = "EQUIRECTANGULAR"
    reader.camera_params = f"{config.PANORAMA_WIDTH},{config.PANORAMA_HEIGHT}"

    extract = pycolmap.FeatureExtractionOptions()
    extract.use_gpu = False
    extract.num_threads = num_threads
    extract.max_image_size = config.PANORAMA_WIDTH
    extract.sift.max_num_features = 8192

    pycolmap.extract_features(str(db_path), str(frames_dir), [],
                              pycolmap.CameraMode.SINGLE, reader, extract,
                              pycolmap.Device.cpu)
    print("[sfm] feature extraction done", flush=True)

    match = pycolmap.FeatureMatchingOptions()
    match.use_gpu = False
    match.num_threads = num_threads
    match.max_num_matches = 8192

    pairing = pycolmap.SequentialPairingOptions()
    pairing.overlap = 8
    pairing.num_threads = num_threads
    pairing.quadratic_overlap = True

    pycolmap.match_sequential(str(db_path), match, pairing, device=pycolmap.Device.cpu)
    print("[sfm] sequential matching done", flush=True)

    pipeline = pycolmap.IncrementalPipelineOptions()
    pipeline.num_threads = num_threads
    pipeline.ba_use_gpu = False
    pipeline.max_num_models = 1

    reconstructions = pycolmap.incremental_mapping(
        str(db_path), str(frames_dir), str(sfm_dir), pipeline)
    for model_id, reconstruction in reconstructions.items():
        print(f"[sfm] model {model_id}: {reconstruction.num_images()} images, "
              f"{reconstruction.num_points3D()} points", flush=True)
    print("[sfm] done", flush=True)

    # pycolmap 4.1 输出到 sfm/0/ 子目录
    return sparse_dir


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="insta360-3dgs sfm",
                                     description="等矩形 SfM 重建")
    parser.add_argument("--output", required=True, help="输出目录 (与 pipeline 一致)")
    parser.add_argument("--threads", type=int, default=8, help="线程数")
    args = parser.parse_args(argv)

    layout = config.output_layout(Path(args.output))
    run_sfm(layout["frames"], layout["sfm"], args.threads)


if __name__ == "__main__":
    main()
