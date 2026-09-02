# -*- coding: utf-8 -*-
"""重型阶段重执行。

uv 环境只装轻量依赖，sfm/train 这些需要 torch/pycolmap 的阶段交给
gaussian-splatting/.venv 里的解释器跑，并注入 PYTHONPATH 指向本项目 src/。
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from . import config

# src/ 目录 (insta360_3dgs 包的父目录)
_SRC_DIR = str(Path(__file__).resolve().parent.parent)


def _absolutize(args: list[str]) -> list[str]:
    """把 `--output <path>` 转成绝对路径。

    train/verify 等阶段会在模块加载时 os.chdir 到 gaussian-splatting 目录,
    若这里传相对路径, 后续 `output/cubemap` 会被解析到错误位置。统一转绝对路径规避。
    """
    out: list[str] = []
    i = 0
    while i < len(args):
        out.append(args[i])
        if args[i] == "--output" and i + 1 < len(args):
            out.append(str(Path(args[i + 1]).resolve()))
            i += 1
        i += 1
    return out


def run_stage(module: str, args: list[str]) -> None:
    python = config.runtime_python()
    if not Path(python).exists():
        raise SystemExit(
            f"运行时解释器不存在: {python}\n"
            f"请检查 gaussian-splatting/.venv 是否已创建, 或用环境变量 "
            f"INSTA360_3DGS_RUNTIME_PYTHON 指定含 torch(CUDA)+pycolmap 的解释器")

    env = os.environ.copy()
    env["PYTHONPATH"] = _SRC_DIR + os.pathsep + env.get("PYTHONPATH", "")

    command = [python, "-m", f"insta360_3dgs.{module}", *_absolutize(args)]
    print(f"\n==> {' '.join(command)}", flush=True)
    subprocess.run(command, env=env, check=True)


if __name__ == "__main__":
    sys.exit("此模块不应直接运行, 请通过 cli 调用")
