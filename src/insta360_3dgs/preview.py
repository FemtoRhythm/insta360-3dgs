# -*- coding: utf-8 -*-
"""预览: 用训练好的模型渲染全景图, 并与原始全景帧对比, 直观查看重建效果。

用法:
    python -m insta360_3dgs preview --output out/ [--frame N] [--iteration 30000]
输出:  <output>/preview/{pano_render.jpg, pano_gt.jpg, compare.jpg}

注意: 本阶段依赖 torch(CUDA), 由 CLI 重执行。
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from . import config

sys.path.insert(0, str(config.GAUSSIAN_SPLATTING_DIR))
os.chdir(config.GAUSSIAN_SPLATTING_DIR)

from argparse import ArgumentParser  # noqa: E402
from arguments import ModelParams, OptimizationParams, PipelineParams  # noqa: E402
from gaussian_renderer import render  # noqa: E402
from scene import GaussianModel, Scene  # noqa: E402
from scene.colmap_loader import read_extrinsics_binary  # noqa: E402
from utils.general_utils import safe_state  # noqa: E402


def _load_model(source_dir: Path, model_dir: Path, iteration: int):
    parser = ArgumentParser()
    lp = ModelParams(parser)
    pp = PipelineParams(parser)
    op = OptimizationParams(parser)
    args = parser.parse_args([
        "--source_path", str(source_dir),
        "--model_path", str(model_dir),
        "--resolution", str(config.TRAIN_RESOLUTION),
        "--data_device", "cpu",
    ])
    dataset = lp.extract(args)
    pipe = pp.extract(args)
    opt = op.extract(args)

    safe_state(False)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    gaussians.load_ply(model_dir / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply")
    return dataset, pipe, scene, gaussians


def render_pano(scene, gaussians, pipe, dataset, frame_idx: int) -> np.ndarray:
    """渲染某帧等矩形全景 (逐面渲染 6 个 cube 面后拼回)。"""
    background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    cameras = {camera.image_name: camera for camera in scene.getTrainCameras()}

    # 等矩形每个像素的射线方向 (右X/下Y/前Z)
    x = (np.arange(config.PANORAMA_WIDTH, dtype=np.float64) + 0.5) / config.PANORAMA_WIDTH
    y = (np.arange(config.PANORAMA_HEIGHT, dtype=np.float64) + 0.5) / config.PANORAMA_HEIGHT
    theta = 2 * np.pi * (x - 0.5)
    phi = np.pi * (0.5 - y[:, None])
    cam_rays = np.stack([
        np.cos(phi) * np.sin(theta)[None, :],
        -np.sin(phi) * np.ones((config.PANORAMA_HEIGHT, config.PANORAMA_WIDTH)),
        np.cos(phi) * np.cos(theta)[None, :],
    ], axis=-1)

    # 逐面渲染
    face_rgbs = {}
    for face_idx, face in enumerate(config.CUBE_FACES):
        camera = cameras[f"frame_{frame_idx:04d}_{face_idx}.jpg"]
        image = render(camera, gaussians, pipe, background,
                       use_trained_exp=dataset.train_test_exp, separate_sh=False)["render"]
        face_rgbs[face] = image.detach().permute(1, 2, 0).clamp(0, 1).cpu().numpy()

    # 每像素选 z 最大的面并采样
    rotations = np.stack([np.asarray(config.CUBE_ROTATIONS[f], np.float64).T
                          for f in config.CUBE_FACES])
    face_rays = np.einsum("hwc,fcd->hwfd", cam_rays, rotations)
    face_rays /= np.linalg.norm(face_rays, axis=-1, keepdims=True)
    best_face = np.argmax(face_rays[..., 2], axis=-1)

    render_size = config.FACE_SIZE // config.TRAIN_RESOLUTION
    render_focal = config.FACE_FOCAL / config.TRAIN_RESOLUTION
    px = render_focal * face_rays[..., 0] / face_rays[..., 2] + render_size / 2.0
    py = render_focal * face_rays[..., 1] / face_rays[..., 2] + render_size / 2.0

    output = np.zeros((config.PANORAMA_HEIGHT, config.PANORAMA_WIDTH, 3), np.float32)
    for face_idx, face in enumerate(config.CUBE_FACES):
        mask = best_face == face_idx
        sample_x = np.clip(px[..., face_idx][mask], 0, render_size - 1).astype(np.int64)
        sample_y = np.clip(py[..., face_idx][mask], 0, render_size - 1).astype(np.int64)
        output[mask] = face_rgbs[face][sample_y, sample_x]
    return output


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="insta360-3dgs preview", description="渲染全景图预览")
    parser.add_argument("--output", required=True, help="输出目录 (与 pipeline 一致)")
    parser.add_argument("--frame", type=int, default=None, help="渲染帧索引 (默认中间帧)")
    parser.add_argument("--iteration", type=int, default=config.TRAIN_ITERATIONS)
    args = parser.parse_args(argv)

    layout = config.output_layout(Path(args.output))
    preview_dir = Path(args.output) / "preview"
    preview_dir.mkdir(parents=True, exist_ok=True)

    dataset, pipe, scene, gaussians = _load_model(
        layout["cubemap"], layout["model"], args.iteration)

    eq_extrinsics = read_extrinsics_binary(layout["sfm"] / "0" / "images.bin")
    n_frames = len(eq_extrinsics)
    frame_idx = args.frame if args.frame is not None else n_frames // 2
    frame_idx = max(1, min(frame_idx, n_frames))

    pano = render_pano(scene, gaussians, pipe, dataset, frame_idx)
    render_path = preview_dir / "pano_render.jpg"
    Image.fromarray((np.clip(pano, 0, 1) * 255).astype(np.uint8)).save(render_path, quality=90)

    gt_path = layout["frames"] / f"frame_{frame_idx:04d}.jpg"
    gt_out = preview_dir / "pano_gt.jpg"
    Image.open(gt_path).convert("RGB").save(gt_out, quality=90)

    # 上下拼接对比图 (上=渲染, 下=原始)
    render_img = Image.open(render_path)
    gt_img = Image.open(gt_out)
    compare = Image.new("RGB", (render_img.width, render_img.height * 2))
    compare.paste(render_img, (0, 0))
    compare.paste(gt_img, (0, render_img.height))
    compare.save(preview_dir / "compare.jpg", quality=90)

    print(f"[preview] frame={frame_idx}/{n_frames} -> {preview_dir}", flush=True)


if __name__ == "__main__":
    main()
