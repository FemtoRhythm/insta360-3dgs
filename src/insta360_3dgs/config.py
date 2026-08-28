# -*- coding: utf-8 -*-
"""共享配置: 输出目录结构、几何参数、外部依赖路径解析。

约定输出目录结构:
    <output>/
        stitched/          # 步骤0: MediaSDK 拼接出的等矩形 mp4
        frames/            # 步骤1: 抽出的全景帧 frame_%04d.jpg
        sfm/               # 步骤2: 等矩形 SfM 稀疏重建 (结果在 sfm/0/)
        cubemap/           # 步骤3: cube 面训练数据 (images/ + sparse/0/)
        model/             # 步骤4: 3DGS 训练输出
        logs/              # 日志
        pointcloud.ply     # 最终导出的标准点云
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# ---- 项目根: src/insta360_3dgs/config.py -> parents[2] = insta360-3dgs ----
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve_env(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value) if value else default


# ---- 外部依赖路径 (均可用环境变量覆盖) ----
# gaussian-splatting 代码 + 已编译 CUDA 扩展 (diff-gaussian-rasterization / simple-knn)
GAUSSIAN_SPLATTING_DIR = _resolve_env(
    "INSTA360_3DGS_GS_DIR",
    PROJECT_ROOT.parent / "20260823_1" / "gaussian-splatting",
)
# pycolmap 4.1.1 (cp311) 预编译包目录
PYLIBS_DIR = _resolve_env(
    "INSTA360_3DGS_PYLIBS",
    PROJECT_ROOT.parent / "20260823_1" / "pylibs",
)
# MediaSDK 离线拼接可执行文件
MEDIASDK_EXE = _resolve_env(
    "INSTA360_3DGS_MEDIASDK_EXE",
    PROJECT_ROOT.parent / "sdk" / "MediaSDK" / "bin" / "MediaSDKTest.exe",
)
# 运行 sfm / train 的解释器 (需含 torch(CUDA) + 已编译扩展 + 可 import pycolmap)
RUNTIME_PYTHON = _resolve_env(
    "INSTA360_3DGS_RUNTIME_PYTHON",
    GAUSSIAN_SPLATTING_DIR / ".venv" / "Scripts" / "python.exe",
)

# ---- 全景 / cube 面几何参数 ----
PANORAMA_WIDTH = 2880
PANORAMA_HEIGHT = 1440
FACE_SIZE = 1536             # cube 面分辨率 (像素)
FACE_FOCAL = 768.0           # 焦距, 90° FOV => f = FACE_SIZE / 2

# ---- 训练参数 ----
TRAIN_ITERATIONS = 30000
TRAIN_RESOLUTION = 2         # 训练降采样倍数
INIT_SCALE_MUL = 0.25        # 初始高斯 scale 缩放 (防止漂浮/过冲)
INIT_OPACITY = 0.5           # 初始不透明度

# ---- 3DGS 点云导出 ----
SH_C0 = 0.28209479177387814  # 球谐第 0 阶系数, 用于 f_dc -> RGB
EXPORT_OPACITY_THRESHOLD = 0.1

# ---- MediaSDK 拼接默认参数 ----
STITCH_OUTPUT_WIDTH = 3840
STITCH_OUTPUT_HEIGHT = 1920
STITCH_TYPE = "optflow"      # template | optflow | dynamicstitch | aistitch

# ---- cube 面定义 (等矩形相机系: 右X / 下Y / 前Z) ----
CUBE_ROTATIONS = {
    "f": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    "r": [[0, 0, -1], [0, 1, 0], [1, 0, 0]],
    "b": [[-1, 0, 0], [0, 1, 0], [0, 0, -1]],
    "l": [[0, 0, 1], [0, 1, 0], [-1, 0, 0]],
    "u": [[1, 0, 0], [0, 0, 1], [0, -1, 0]],
    "d": [[1, 0, 0], [0, 0, -1], [0, 1, 0]],
}
# 面序号 0..5 (与 ffmpeg v360 c3x2 布局一致), 注意 "f" 在 index 2
CUBE_FACES = ["r", "u", "f", "d", "l", "b"]


def output_layout(output_dir: Path | str) -> dict[str, Path]:
    """返回输出目录下各子路径。"""
    output_dir = Path(output_dir)
    return {
        "stitched": output_dir / "stitched",
        "frames": output_dir / "frames",
        "sfm": output_dir / "sfm",
        "cubemap": output_dir / "cubemap",
        "model": output_dir / "model",
        "logs": output_dir / "logs",
        "pointcloud": output_dir / "pointcloud.ply",
    }


def add_third_party_paths() -> None:
    """把 gaussian-splatting 与 pylibs (pycolmap) 加入 sys.path。"""
    for path in (GAUSSIAN_SPLATTING_DIR, PYLIBS_DIR):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def runtime_python() -> str:
    """返回重型阶段 (sfm/train) 使用的解释器路径。"""
    return str(RUNTIME_PYTHON)
