# -*- coding: utf-8 -*-
"""3DGS 训练（手写训练循环，含初始 scale/opacity 缩放）。

依赖 torch(CUDA) + 已编译 gaussian-splatting，由 CLI 用 runtime python 重执行。
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

from . import config

sys.path.insert(0, str(config.GAUSSIAN_SPLATTING_DIR))
os.chdir(config.GAUSSIAN_SPLATTING_DIR)

import torch  # noqa: E402
from random import randint  # noqa: E402
from argparse import ArgumentParser  # noqa: E402
from arguments import ModelParams, OptimizationParams, PipelineParams  # noqa: E402
from gaussian_renderer import render  # noqa: E402
from scene import GaussianModel, Scene  # noqa: E402
from utils.general_utils import safe_state  # noqa: E402
from utils.loss_utils import l1_loss, ssim  # noqa: E402


def _inverse_sigmoid(value: float) -> float:
    return math.log(value / (1.0 - value))


def train(source_dir: Path, model_dir: Path, iterations: int) -> None:
    final_ply = model_dir / "point_cloud" / f"iteration_{iterations}" / "point_cloud.ply"
    if final_ply.exists():
        print(f"[train] 跳过已存在模型: {final_ply}", flush=True)
        return

    model_dir.mkdir(parents=True, exist_ok=True)

    parser = ArgumentParser()
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--save_iterations", nargs="+", type=int,
                        default=[7000, 15000, 30000])
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    args = parser.parse_args([
        "--source_path", str(source_dir),
        "--model_path", str(model_dir),
        "--resolution", str(config.TRAIN_RESOLUTION),
        "--data_device", "cpu",
        "--iterations", str(iterations),
    ])
    opt = op.extract(args)
    opt.iterations = iterations
    dataset = lp.extract(args)
    pipe = pp.extract(args)

    safe_state(True)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)

    with torch.no_grad():
        scale_before = gaussians.get_scaling.max().item()
        gaussians._scaling.data += math.log(config.INIT_SCALE_MUL)
        gaussians._opacity.data.fill_(_inverse_sigmoid(config.INIT_OPACITY))
        scale_after = gaussians.get_scaling.max().item()
    print(f"[train] init scale x{config.INIT_SCALE_MUL}: {scale_before:.4f} -> {scale_after:.4f}; "
          f"opacity={config.INIT_OPACITY}", flush=True)
    gaussians.training_setup(opt)

    background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    viewpoint_stack = scene.getTrainCameras().copy()
    print(f"[train] n={gaussians.get_xyz.shape[0]} cams={len(viewpoint_stack)} "
          f"extent={scene.cameras_extent:.2f}", flush=True)

    for iteration in range(1, opt.iterations + 1):
        gaussians.update_learning_rate(iteration)
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        camera = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        render_pkg = render(camera, gaussians, pipe, background,
                            use_trained_exp=dataset.train_test_exp, separate_sh=False)
        image = render_pkg["render"]
        viewspace_points = render_pkg["viewspace_points"]
        visibility = render_pkg["visibility_filter"]
        radii = render_pkg["radii"]

        gt_image = camera.original_image.cuda()
        loss_l1 = l1_loss(image, gt_image)
        loss_ssim = ssim(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * loss_l1 + opt.lambda_dssim * (1.0 - loss_ssim)
        loss.backward()

        with torch.no_grad():
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility] = torch.max(
                    gaussians.max_radii2D[visibility], radii[visibility])
                gaussians.add_densification_stats(viewspace_points, visibility)
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold, 0.005, scene.cameras_extent,
                        size_threshold, radii)
                if iteration % opt.opacity_reset_interval == 0:
                    gaussians.reset_opacity()
            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none=True)

            if iteration % 500 == 0:
                print(f"[train] iter {iteration:5d} loss={loss.item():.4f} "
                      f"n={gaussians.get_xyz.shape[0]} "
                      f"nan={torch.isnan(image).sum().item()}", flush=True)

    for checkpoint in [7000, 15000, iterations]:
        scene.save(checkpoint)
    print("[train] done", flush=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="insta360-3dgs train", description="3DGS 训练")
    parser.add_argument("--output", required=True, help="输出目录 (与 pipeline 一致)")
    parser.add_argument("--iterations", type=int, default=config.TRAIN_ITERATIONS)
    args = parser.parse_args(argv)

    layout = config.output_layout(Path(args.output))
    train(layout["cubemap"], layout["model"], args.iterations)


if __name__ == "__main__":
    main()
