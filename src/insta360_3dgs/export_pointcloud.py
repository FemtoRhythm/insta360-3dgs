# -*- coding: utf-8 -*-
"""步骤5: 导出标准点云 PLY (x,y,z + rgb)。

3DGS 的 point_cloud.ply 字段是 f_dc_*/opacity/scale/rot, 普通查看器不认,
需转成标准 x,y,z + rgb 并过滤低不透明度漂浮点。

用法:
    python -m insta360_3dgs export --output out/ [--iteration 30000]
输入:  <output>/model/point_cloud/iteration_<n>/point_cloud.ply
输出:  <output>/pointcloud.ply
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from . import config

config.add_third_party_paths()
from plyfile import PlyData, PlyElement  # noqa: E402


def export_pointcloud(model_dir: Path, output_path: Path, iteration: int) -> None:
    if output_path.exists():
        print(f"[export] 跳过已存在: {output_path}", flush=True)
        return

    source = model_dir / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
    ply = PlyData.read(source)
    vertex = ply["vertex"]

    positions = np.vstack([vertex["x"], vertex["y"], vertex["z"]]).T
    direct_current = np.vstack([vertex["f_dc_0"], vertex["f_dc_1"], vertex["f_dc_2"]]).T
    colors = np.clip(direct_current * config.SH_C0 + 0.5, 0, 1) * 255
    colors = np.round(colors).astype(np.uint8)
    opacities = 1.0 / (1.0 + np.exp(-vertex["opacity"]))

    keep = opacities >= config.EXPORT_OPACITY_THRESHOLD
    positions, colors = positions[keep], colors[keep]
    print(f"[export] points: {len(vertex)} -> {len(positions)} "
          f"(opacity >= {config.EXPORT_OPACITY_THRESHOLD})", flush=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dtype = [("x", "f4"), ("y", "f4"), ("z", "f4"),
             ("red", "u1"), ("green", "u1"), ("blue", "u1")]
    elements = np.empty(len(positions), dtype=dtype)
    elements["x"], elements["y"], elements["z"] = positions[:, 0], positions[:, 1], positions[:, 2]
    elements["red"], elements["green"], elements["blue"] = colors[:, 0], colors[:, 1], colors[:, 2]
    PlyData([PlyElement.describe(elements, "vertex")]).write(output_path)
    print(f"[export] saved -> {output_path}", flush=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="insta360-3dgs export", description="导出标准点云")
    parser.add_argument("--output", required=True, help="输出目录 (与 pipeline 一致)")
    parser.add_argument("--iteration", type=int, default=30000, help="训练迭代点")
    args = parser.parse_args(argv)

    layout = config.output_layout(Path(args.output))
    export_pointcloud(layout["model"], layout["pointcloud"], args.iteration)


if __name__ == "__main__":
    main()
