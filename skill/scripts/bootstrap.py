#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""insta360-3dgs skill 引导脚本。

职责：
    check   诊断环境（源码 / uv / ffmpeg / MediaSDK / gaussian-splatting /
            pycolmap / runtime python / GPU）
    setup   准备源码（定位或 git clone）并 uv sync 安装轻量依赖
    run     注入外部依赖路径后运行端到端流水线（.insv -> 全景 3DGS -> 点云）

仅依赖 Python 标准库，可在 Python 3.8+ 直接运行。
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

GITHUB_REPO = "https://github.com/FemtoRhythm/insta360-3dgs"

# 本机已知部署位置，作为「自动识别」的候选路径；可自行增删。
_KNOWN_PROJECT_CANDIDATES = [
    Path(r"d:\Documents\WorkSpace\python\i2\20260828_3\insta360-3dgs"),
]
_KNOWN_MEDIASDK_CANDIDATES = [
    Path(r"d:\Documents\WorkSpace\python\i2\20260828_3\sdk\MediaSDK\bin\MediaSDKTest.exe"),
]
_KNOWN_GS_DIR_CANDIDATES = [
    Path(r"d:\Documents\WorkSpace\python\i2\20260828_3\20260823_1\gaussian-splatting"),
]
_KNOWN_PYLIBS_CANDIDATES = [
    Path(r"d:\Documents\WorkSpace\python\i2\20260828_3\20260823_1\pylibs"),
]


# --------------------------------------------------------------------------- #
# 定位
# --------------------------------------------------------------------------- #
def _is_project(path: Path) -> bool:
    return (path / "pyproject.toml").is_file() and (path / "src" / "insta360_3dgs").is_dir()


def find_project() -> Path | None:
    """按优先级定位源码项目根：环境变量 -> cwd -> 已知候选。"""
    env = os.environ.get("INSTA360_3DGS_PROJECT_DIR")
    if env and _is_project(Path(env)):
        return Path(env)

    cwd = Path.cwd()
    if _is_project(cwd):
        return cwd

    for p in _KNOWN_PROJECT_CANDIDATES:
        if _is_project(p):
            return p
    return None


def _first_existing(candidates: list[Path]) -> Path | None:
    for p in candidates:
        if p.exists():
            return p
    return None


def _runtime_python_candidates(gs_dir: Path) -> list[Path]:
    return [gs_dir / ".venv" / "Scripts" / "python.exe",   # Windows
            gs_dir / ".venv" / "bin" / "python"]           # POSIX


def resolve_deps(project: Path | None) -> dict[str, Path | None]:
    """解析四个外部依赖路径。优先级：环境变量 > config 默认推导 > 已知候选。"""
    parent = project.parent if project else Path.cwd().parent

    def _pick(name: str, default: list[Path], known: list[Path]) -> Path | None:
        env = os.environ.get(name)
        if env:
            return Path(env)
        return _first_existing(default + known)

    gs_dir = _pick(
        "INSTA360_3DGS_GS_DIR",
        [parent / "20260823_1" / "gaussian-splatting"],
        _KNOWN_GS_DIR_CANDIDATES,
    )
    return {
        "mediasdk_exe": _pick(
            "INSTA360_3DGS_MEDIASDK_EXE",
            [parent / "sdk" / "MediaSDK" / "bin" / "MediaSDKTest.exe"],
            _KNOWN_MEDIASDK_CANDIDATES,
        ),
        "gs_dir": gs_dir,
        "pylibs": _pick(
            "INSTA360_3DGS_PYLIBS",
            [parent / "20260823_1" / "pylibs"],
            _KNOWN_PYLIBS_CANDIDATES,
        ),
        "runtime_python": (
            Path(os.environ["INSTA360_3DGS_RUNTIME_PYTHON"])
            if os.environ.get("INSTA360_3DGS_RUNTIME_PYTHON")
            else _first_existing(_runtime_python_candidates(gs_dir) if gs_dir else [])
        ),
    }


# --------------------------------------------------------------------------- #
# check
# --------------------------------------------------------------------------- #
def _ok(b: bool) -> str:
    return "OK " if b else "MISS"


def cmd_check(args: argparse.Namespace) -> int:
    project = find_project()
    deps = resolve_deps(project)

    uv = shutil.which("uv")
    ffmpeg = shutil.which("ffmpeg")
    git = shutil.which("git")
    nvidia_smi = shutil.which("nvidia-smi")

    print("=== insta360-3dgs 环境诊断 ===")
    print(f"[{_ok(project is not None)}] 项目源码        {project or '未找到'}")
    print(f"[{_ok(uv is not None)}] uv               {uv or ''}")
    print(f"[{_ok(ffmpeg is not None)}] ffmpeg           {ffmpeg or ''}")
    print(f"[{_ok(git is not None)}] git              {git or ''}")
    print(f"[{_ok(nvidia_smi is not None)}] NVIDIA GPU       {nvidia_smi or ''}")
    print(f"[{_ok(deps['mediasdk_exe'] is not None)}] MediaSDKTest.exe {deps['mediasdk_exe'] or ''}")
    print(f"[{_ok(deps['gs_dir'] is not None)}] gaussian-splatting {deps['gs_dir'] or ''}")
    print(f"[{_ok(deps['pylibs'] is not None)}] pylibs(pycolmap) {deps['pylibs'] or ''}")
    print(f"[{_ok(deps['runtime_python'] is not None)}] runtime python   {deps['runtime_python'] or ''}")

    missing = [
        ("项目源码", project is None),
        ("uv", uv is None),
        ("ffmpeg", ffmpeg is None),
        ("MediaSDKTest.exe", deps["mediasdk_exe"] is None),
        ("gaussian-splatting", deps["gs_dir"] is None),
        ("pylibs(pycolmap)", deps["pylibs"] is None),
        ("runtime python", deps["runtime_python"] is None),
    ]
    miss_list = [name for name, is_miss in missing if is_miss]
    if miss_list:
        print(f"\n缺失项: {', '.join(miss_list)}")
        print("提示: 运行 `python bootstrap.py setup` 可自动准备源码与轻量依赖；"
              "重型依赖(MediaSDK/gaussian-splatting/pycolmap)需预先放置并用环境变量指定。")
        return 1
    print("\n环境就绪，可运行: python bootstrap.py run --input <xx.insv> --output <out>")
    return 0


# --------------------------------------------------------------------------- #
# setup
# --------------------------------------------------------------------------- #
def cmd_setup(args: argparse.Namespace) -> int:
    project = find_project()
    if project is None:
        dest = Path(os.environ.get("INSTA360_3DGS_PROJECT_DIR", Path.home() / "insta360-3dgs"))
        if not shutil.which("git"):
            print("错误: 未安装 git，无法 clone 源码。请先安装 git 或用 INSTA360_3DGS_PROJECT_DIR 指定已有源码。")
            return 1
        print(f"克隆源码: {GITHUB_REPO} -> {dest}")
        subprocess.run(["git", "clone", GITHUB_REPO, str(dest)], check=True)
        project = dest

    print(f"项目目录: {project}")
    uv = shutil.which("uv")
    if not uv:
        print("错误: 未安装 uv，无法同步依赖。请先安装 uv (https://docs.astral.sh/uv/)。")
        return 1
    print("同步轻量依赖: uv sync")
    subprocess.run(["uv", "sync"], cwd=project, check=True)

    # 校验外部依赖
    deps = resolve_deps(project)
    for key, label in [("mediasdk_exe", "MediaSDKTest.exe"),
                       ("gs_dir", "gaussian-splatting"),
                       ("pylibs", "pylibs(pycolmap)"),
                       ("runtime_python", "runtime python")]:
        if deps[key] is None:
            print(f"警告: 未检测到 {label}，请放置后用对应 INSTA360_3DGS_* 环境变量指定。")
    print("setup 完成。")
    return 0


# --------------------------------------------------------------------------- #
# run
# --------------------------------------------------------------------------- #
def cmd_run(args: argparse.Namespace) -> int:
    project = find_project()
    if project is None:
        print("错误: 未找到项目源码。请先运行 `python bootstrap.py setup`。")
        return 1

    deps = resolve_deps(project)
    env = os.environ.copy()
    mapping = {
        "INSTA360_3DGS_MEDIASDK_EXE": deps["mediasdk_exe"],
        "INSTA360_3DGS_GS_DIR": deps["gs_dir"],
        "INSTA360_3DGS_PYLIBS": deps["pylibs"],
        "INSTA360_3DGS_RUNTIME_PYTHON": deps["runtime_python"],
    }
    for k, v in mapping.items():
        if v is None:
            print(f"错误: 依赖缺失 {k}，请先运行 `python bootstrap.py check` 诊断。")
            return 1
        env[k] = str(v)

    argv = ["uv", "run", "insta360-3dgs", "run",
            "--input", *args.input, "--output", str(Path(args.output).resolve())]
    if args.iterations:
        argv += ["--iterations", str(args.iterations)]
    if args.verify:
        argv += ["--verify"]

    print("==>", " ".join(argv))
    subprocess.run(argv, cwd=project, env=env, check=True)
    print(f"\n完成! 点云输出: {Path(args.output).resolve() / 'pointcloud.ply'}")
    return 0


# --------------------------------------------------------------------------- #
# 入口
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bootstrap.py", description="insta360-3dgs skill 引导脚本")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("check", help="诊断环境").set_defaults(func=cmd_check)
    sub.add_parser("setup", help="准备源码与轻量依赖").set_defaults(func=cmd_setup)

    p_run = sub.add_parser("run", help="运行端到端流水线")
    p_run.add_argument("--input", required=True, nargs="+", help="一个或多个 .insv 文件或目录")
    p_run.add_argument("--output", required=True, help="输出目录")
    p_run.add_argument("--iterations", type=int, default=None, help="3DGS 训练迭代数")
    p_run.add_argument("--verify", action="store_true", help="训练后质量评估")
    p_run.set_defaults(func=cmd_run)

    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
