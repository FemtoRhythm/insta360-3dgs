# -*- coding: utf-8 -*-
"""质量评估。--mode train 算训练视图的 PSNR/SSIM/L1，--mode pano 把 6 面拼回
整张全景与真实全景做端到端对比。依赖 torch(CUDA)，由 CLI 重执行。
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
from scene.colmap_loader import qvec2rotmat, read_extrinsics_binary  # noqa: E402
from utils.general_utils import safe_state  # noqa: E402
from utils.loss_utils import l1_loss, ssim  # noqa: E402


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

    safe_state(True)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    gaussians.load_ply(model_dir / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply")
    return dataset, pipe, scene, gaussians


def _train_metrics(dataset, pipe, scene, gaussians) -> None:
    background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    cameras = scene.getTrainCameras()
    psnr_list, l1_list, ssim_list = [], [], []

    for camera in cameras:
        image = render(camera, gaussians, pipe, background,
                       use_trained_exp=dataset.train_test_exp, separate_sh=False)["render"]
        gt_image = camera.original_image.cuda()
        mse = float(((image - gt_image) ** 2).mean())
        psnr_list.append(10 * np.log10(1.0 / max(mse, 1e-12)))
        l1_list.append(float(l1_loss(image, gt_image)))
        ssim_list.append(float(ssim(image, gt_image)))

    print(f"[verify:train] PSNR mean={np.mean(psnr_list):.2f} "
          f"median={np.median(psnr_list):.2f}", flush=True)
    print(f"[verify:train] L1   mean={np.mean(l1_list):.4f}", flush=True)
    print(f"[verify:train] SSIM mean={np.mean(ssim_list):.4f}", flush=True)


def _pano_metrics(frames_dir: Path, sfm_sparse_dir: Path, dataset, pipe, scene, gaussians) -> None:
    background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    cameras = {camera.image_name: camera for camera in scene.getTrainCameras()}
    eq_extrinsics = read_extrinsics_binary(sfm_sparse_dir / "images.bin")

    def frame_pose(frame_idx: int):
        for image in eq_extrinsics.values():
            if image.name == f"frame_{frame_idx:04d}.jpg":
                return qvec2rotmat(image.qvec)
        raise KeyError(frame_idx)

    def render_equirect(frame_idx: int) -> np.ndarray:
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

    # 抽样若干帧做全景对比
    frame_indices = [1, len(eq_extrinsics) // 4, len(eq_extrinsics) // 2,
                     3 * len(eq_extrinsics) // 4, len(eq_extrinsics)]
    frame_indices = sorted(set(max(1, i) for i in frame_indices))

    for frame_idx in frame_indices:
        pano_gt = np.asarray(
            Image.open(frames_dir / f"frame_{frame_idx:04d}.jpg").convert("RGB"),
            np.float32) / 255.0
        pano_render = render_equirect(frame_idx)
        mse = float(((pano_render - pano_gt) ** 2).mean())
        psnr = 10 * np.log10(1.0 / max(mse, 1e-12))
        print(f"[verify:pano] frame {frame_idx}: PSNR={psnr:.2f} dB", flush=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="insta360-3dgs verify", description="质量评估")
    parser.add_argument("--output", required=True, help="输出目录 (与 pipeline 一致)")
    parser.add_argument("--mode", choices=["train", "pano"], default="train")
    parser.add_argument("--iteration", type=int, default=30000)
    args = parser.parse_args(argv)

    layout = config.output_layout(Path(args.output))
    dataset, pipe, scene, gaussians = _load_model(
        layout["cubemap"], layout["model"], args.iteration)

    if args.mode == "train":
        _train_metrics(dataset, pipe, scene, gaussians)
    else:
        _pano_metrics(layout["frames"], layout["sfm"] / "0",
                      dataset, pipe, scene, gaussians)


if __name__ == "__main__":
    main()
