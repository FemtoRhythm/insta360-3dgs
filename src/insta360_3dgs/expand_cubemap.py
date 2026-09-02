# -*- coding: utf-8 -*-
"""从等矩形 SfM 位姿展开 6 个共光心透视面，写 PINHOLE sparse。

对每帧等矩形相机（旋转 world_to_cam，光心 center）：
    face_rotation    = CUBE_ROTATIONS[face] @ world_to_cam
    face_translation = -face_rotation @ center          # 6 面共光心
再按 COLMAP EQUIRECTANGULAR 的反投影公式做双线性采样，保证图像与位姿自洽。
"""
from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

from . import config

config.add_third_party_paths()
from plyfile import PlyData, PlyElement  # noqa: E402
from scene.colmap_loader import (  # noqa: E402
    qvec2rotmat,
    read_extrinsics_binary,
    read_points3D_binary,
    rotmat2qvec,
)

PINHOLE_MODEL_ID = 1
EQUIRECTANGULAR_MODEL_ID = 17


def _read_equirect_camera_size(cameras_path: Path) -> tuple[int, int]:
    """手工解析 EQUIRECTANGULAR cameras.bin (model_id=17 不在 colmap_loader 表中)。"""
    with open(cameras_path, "rb") as f:
        f.read(8)  # num_cameras
        _, model_id = struct.unpack("<ii", f.read(8))
        width, height = struct.unpack("<QQ", f.read(16))
    if model_id != EQUIRECTANGULAR_MODEL_ID:
        raise SystemExit(
            f"unexpected camera model id {model_id}, expected EQUIRECTANGULAR(17)")
    return int(width), int(height)


def _build_sample_grid(face_name: str) -> tuple[np.ndarray, ...]:
    """为某个面预计算双线性采样网格 (x0, x1, fx, y0, y1, fy)。"""
    rotation = np.asarray(config.CUBE_ROTATIONS[face_name], np.float64)
    coords = np.arange(config.FACE_SIZE, dtype=np.float64)
    xx, yy = np.meshgrid(coords, coords)

    # 面坐标系下的单位射线
    face_rays = np.stack([
        (xx - config.FACE_SIZE / 2.0) / config.FACE_FOCAL,
        (yy - config.FACE_SIZE / 2.0) / config.FACE_FOCAL,
        np.ones_like(xx),
    ], axis=-1)
    face_rays /= np.linalg.norm(face_rays, axis=-1, keepdims=True)

    # 转回等矩形相机系, 反投影到全景像素坐标
    cam_rays = face_rays @ rotation
    theta = np.arctan2(cam_rays[..., 0], cam_rays[..., 2])
    phi = np.arcsin(np.clip(-cam_rays[..., 1], -1.0, 1.0))
    sample_x = np.mod((theta / (2 * np.pi) + 0.5) * config.PANORAMA_WIDTH, config.PANORAMA_WIDTH)
    sample_y = np.clip((0.5 - phi / np.pi) * config.PANORAMA_HEIGHT, 0.0, config.PANORAMA_HEIGHT - 1e-6)

    x0 = np.floor(sample_x).astype(np.int64)
    fx = sample_x - x0
    x1 = (x0 + 1) % config.PANORAMA_WIDTH
    y0 = np.floor(sample_y).astype(np.int64)
    fy = sample_y - y0
    y1 = np.minimum(y0 + 1, config.PANORAMA_HEIGHT - 1)
    return x0, x1, fx, y0, y1, fy


def _sample_bilinear(pano: np.ndarray, grid: tuple[np.ndarray, ...]) -> np.ndarray:
    x0, x1, fx, y0, y1, fy = grid
    top = pano[y0, x0] * (1 - fx)[..., None] + pano[y0, x1] * fx[..., None]
    bottom = pano[y1, x0] * (1 - fx)[..., None] + pano[y1, x1] * fx[..., None]
    return top * (1 - fy)[..., None] + bottom * fy[..., None]


def _write_cameras(path: Path) -> None:
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", 1))  # num_cameras
        f.write(struct.pack("<iiQQ", 1, PINHOLE_MODEL_ID, config.FACE_SIZE, config.FACE_SIZE))
        f.write(np.asarray([config.FACE_FOCAL, config.FACE_FOCAL,
                            config.FACE_SIZE / 2.0, config.FACE_SIZE / 2.0],
                           np.float64).tobytes())


def _write_images(path: Path, images: list[tuple]) -> None:
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(images)))
        for image_id, qvec, tvec, name in images:
            f.write(struct.pack("<i", image_id))
            f.write(np.asarray(qvec, np.float64).tobytes())
            f.write(np.asarray(tvec, np.float64).tobytes())
            f.write(struct.pack("<i", 1))  # camera_id (共享内参)
            f.write(name.encode("utf-8") + b"\x00")
            f.write(struct.pack("<Q", 0))  # 无 2D 观测点


def expand_cubemap(frames_dir: Path, sfm_sparse_dir: Path, cubemap_dir: Path) -> None:
    sparse_dir = cubemap_dir / "sparse" / "0"
    if (sparse_dir / "images.bin").exists():
        print(f"[cubemap] 跳过已存在: {sparse_dir}", flush=True)
        return

    images_dir = cubemap_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir.mkdir(parents=True, exist_ok=True)

    _read_equirect_camera_size(sfm_sparse_dir / "cameras.bin")
    extrinsics = read_extrinsics_binary(sfm_sparse_dir / "images.bin")

    # 解析每帧位姿 (world->cam 旋转 + 光心)
    frame_order = sorted(extrinsics.keys(),
                         key=lambda i: int(extrinsics[i].name.split("_")[1].split(".")[0]))
    frames = []
    for key in frame_order:
        image = extrinsics[key]
        world_to_cam = qvec2rotmat(image.qvec)
        center = -world_to_cam.T @ np.asarray(image.tvec)
        frames.append((image.name, world_to_cam, center))
    print(f"[cubemap] frames={len(frames)}", flush=True)

    grids = {face: _build_sample_grid(face) for face in config.CUBE_FACES}
    output_images = []
    image_id = 1

    for name, world_to_cam, center in frames:
        frame_idx = int(name.split("_")[1].split(".")[0])
        pano = np.asarray(
            Image.open(frames_dir / name).convert("RGB"), dtype=np.float32)
        if pano.shape[0] != config.PANORAMA_HEIGHT or pano.shape[1] != config.PANORAMA_WIDTH:
            raise SystemExit(f"unexpected pano size {pano.shape} for {name}")

        for face_idx, face in enumerate(config.CUBE_FACES):
            face_rotation = np.asarray(config.CUBE_ROTATIONS[face], np.float64) @ world_to_cam
            face_translation = -face_rotation @ center
            qvec = rotmat2qvec(face_rotation)
            out_name = f"frame_{frame_idx:04d}_{face_idx}.jpg"

            rendered = _sample_bilinear(pano, grids[face])
            Image.fromarray(np.clip(rendered, 0, 255).astype(np.uint8)).save(
                images_dir / out_name, quality=92)
            output_images.append((image_id, qvec, face_translation, out_name))
            image_id += 1

        if frame_idx % 20 == 0:
            print(f"  frame {frame_idx} done ({len(output_images)} images)", flush=True)

    _write_cameras(sparse_dir / "cameras.bin")
    _write_images(sparse_dir / "images.bin", output_images)
    print(f"[cubemap] wrote sparse: {len(output_images)} images", flush=True)

    # points3D: 过滤高误差点与远离轨迹的漂浮点
    xyz, rgb, errors = read_points3D_binary(sfm_sparse_dir / "points3D.bin")
    errors = errors.ravel()
    centers = np.array([c for _, _, c in frames])
    dist_to_traj, _ = cKDTree(centers).query(xyz, k=1)
    keep = (errors < 4.0) & (dist_to_traj < 30.0)
    xyz, rgb = xyz[keep], rgb[keep]

    dtype = [("x", "f4"), ("y", "f4"), ("z", "f4"),
             ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
             ("red", "u1"), ("green", "u1"), ("blue", "u1")]
    vertices = np.empty(len(xyz), dtype=dtype)
    vertices["x"], vertices["y"], vertices["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    vertices["nx"] = vertices["ny"] = vertices["nz"] = 0.0
    vertices["red"], vertices["green"], vertices["blue"] = rgb.astype(np.uint8).T
    PlyData([PlyElement.describe(vertices, "vertex")]).write(
        sparse_dir / "points3D.ply")
    print(f"[cubemap] points3D: {len(errors)} -> {len(xyz)} (err<4, dist<30)",
          flush=True)

    _verify_concentric(sparse_dir)
    print("[cubemap] done", flush=True)


def _verify_concentric(sparse_dir: Path) -> None:
    """自检: 同一帧的 6 个面光心必须重合 (否则位姿推导有误)。"""
    extrinsics = read_extrinsics_binary(sparse_dir / "images.bin")
    by_frame: dict[str, list] = {}
    for image in extrinsics.values():
        by_frame.setdefault(image.name.split("_")[1], []).append(image)

    offsets = []
    for images in by_frame.values():
        centers = np.array([
            -qvec2rotmat(image.qvec).T @ np.asarray(image.tvec) for image in images])
        offsets.append(np.max(np.linalg.norm(centers - centers.mean(axis=0), axis=1)))
    print(f"[cubemap] same-frame center offset: "
          f"mean={np.mean(offsets):.3e} max={np.max(offsets):.3e}", flush=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="insta360-3dgs cubemap",
                                     description="展开 cube 面训练数据")
    parser.add_argument("--output", required=True, help="输出目录 (与 pipeline 一致)")
    args = parser.parse_args(argv)

    layout = config.output_layout(Path(args.output))
    expand_cubemap(layout["frames"], layout["sfm"] / "0", layout["cubemap"])


if __name__ == "__main__":
    main()
