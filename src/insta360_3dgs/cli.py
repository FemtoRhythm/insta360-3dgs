# -*- coding: utf-8 -*-
"""统一命令行入口: Insta360 .insv -> 全景 3DGS -> 点云。

子命令:
    run      一键端到端: .insv -> 等矩形 mp4 -> SfM -> cube 面 -> 3DGS -> 点云
    stitch   仅拼接 .insv -> 等矩形 mp4 (MediaSDK)
    extract  仅抽帧
    sfm      仅等矩形 SfM
    cubemap  仅展开 cube 面
    train    仅 3DGS 训练
    export   仅导出标准点云
    verify   质量评估 (train/pano)

轻量阶段 (stitch/extract) 在 uv 环境内执行; 重型阶段 (sfm/cubemap/train/export/verify)
由 _dispatch 用 runtime python 重执行。
"""
from __future__ import annotations

import argparse
from pathlib import Path

from . import _dispatch
from . import config
from . import extract_frames
from . import sdk_stitch


def _cmd_stitch(args: argparse.Namespace) -> None:
    argv = ["--input", *args.input, "--output", args.output]
    if args.stitch_type:
        argv += ["--stitch-type", args.stitch_type]
    if args.output_size:
        argv += ["--output-size", args.output_size]
    sdk_stitch.main(argv)


def _cmd_extract(args: argparse.Namespace) -> None:
    extract_frames.main(["--input", args.input, "--output", args.output,
                         "--fps", str(args.fps), "--width", str(args.width)])


def _cmd_sfm(args: argparse.Namespace) -> None:
    _dispatch.run_stage("run_sfm", ["--output", args.output,
                                    "--threads", str(args.threads)])


def _cmd_cubemap(args: argparse.Namespace) -> None:
    _dispatch.run_stage("expand_cubemap", ["--output", args.output])


def _cmd_train(args: argparse.Namespace) -> None:
    _dispatch.run_stage("train", ["--output", args.output,
                                  "--iterations", str(args.iterations)])


def _cmd_export(args: argparse.Namespace) -> None:
    _dispatch.run_stage("export_pointcloud", ["--output", args.output,
                                              "--iteration", str(args.iteration)])


def _cmd_verify(args: argparse.Namespace) -> None:
    _dispatch.run_stage("verify", ["--output", args.output, "--mode", args.mode,
                                   "--iteration", str(args.iteration)])


def _cmd_preview(args: argparse.Namespace) -> None:
    argv = ["--output", args.output, "--iteration", str(args.iteration)]
    if args.frame is not None:
        argv += ["--frame", str(args.frame)]
    _dispatch.run_stage("preview", argv)


def _cmd_run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    layout = config.output_layout(output_dir)

    # 1. 拼接 .insv -> 等矩形 mp4
    stitch_argv = ["--input", *args.input, "--output", str(layout["stitched"])]
    if args.stitch_type:
        stitch_argv += ["--stitch-type", args.stitch_type]
    if args.output_size:
        stitch_argv += ["--output-size", args.output_size]
    sdk_stitch.main(stitch_argv)

    # 2. 选择拼接产物 (默认处理第一个片段)
    mp4s = sorted(layout["stitched"].glob("*.mp4"))
    if not mp4s:
        raise SystemExit("拼接阶段未产生任何 mp4, 请检查输入 .insv 与 MediaSDK 输出")
    if len(mp4s) > 1:
        print(f"[run] 检测到多个片段 {[m.name for m in mp4s]}, "
              f"仅处理第一个: {mp4s[0].name}", flush=True)
    video = mp4s[0]

    # 3. 抽帧
    extract_frames.main(["--input", str(video), "--output", str(output_dir),
                         "--fps", str(args.fps), "--width", str(args.width)])

    # 4-7. 重型阶段
    _dispatch.run_stage("run_sfm", ["--output", str(output_dir),
                                    "--threads", str(args.threads)])
    _dispatch.run_stage("expand_cubemap", ["--output", str(output_dir)])
    _dispatch.run_stage("train", ["--output", str(output_dir),
                                  "--iterations", str(args.iterations)])
    _dispatch.run_stage("export_pointcloud", ["--output", str(output_dir),
                                              "--iteration", str(args.iterations)])

    if args.verify:
        _dispatch.run_stage("verify", ["--output", str(output_dir), "--mode", "train",
                                       "--iteration", str(args.iterations)])

    print(f"\n完成! 点云已输出到: {layout['pointcloud']}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="insta360-3dgs",
        description="Insta360 .insv -> 全景 3D Gaussian Splatting 点云流水线")
    sub = parser.add_subparsers(dest="command", required=True)

    # run
    p_run = sub.add_parser("run", help="一键端到端重建")
    p_run.add_argument("--input", required=True, nargs="+",
                       help="一个或多个 .insv 文件或包含它们的目录")
    p_run.add_argument("--output", required=True, help="输出目录")
    p_run.add_argument("--stitch-type", default=config.STITCH_TYPE,
                       choices=["template", "optflow", "dynamicstitch", "aistitch"])
    p_run.add_argument("--output-size", default=None, help="拼接分辨率 WxH")
    p_run.add_argument("--fps", type=float, default=0.5, help="抽帧频率")
    p_run.add_argument("--width", type=int, default=config.PANORAMA_WIDTH, help="全景帧宽度")
    p_run.add_argument("--iterations", type=int, default=config.TRAIN_ITERATIONS)
    p_run.add_argument("--threads", type=int, default=8, help="SfM 线程数")
    p_run.add_argument("--verify", action="store_true", help="训练后做质量评估")
    p_run.set_defaults(func=_cmd_run)

    # stitch
    p_stitch = sub.add_parser("stitch", help="拼接 .insv -> 等矩形 mp4")
    p_stitch.add_argument("--input", required=True, nargs="+",
                          help="一个或多个 .insv 文件或目录")
    p_stitch.add_argument("--output", required=True, help="输出目录")
    p_stitch.add_argument("--stitch-type", default=config.STITCH_TYPE,
                          choices=["template", "optflow", "dynamicstitch", "aistitch"])
    p_stitch.add_argument("--output-size", default=None, help="拼接分辨率 WxH")
    p_stitch.set_defaults(func=_cmd_stitch)

    # extract
    p_extract = sub.add_parser("extract", help="从全景视频抽帧")
    p_extract.add_argument("--input", required=True, help="360° 等矩形全景视频 (mp4)")
    p_extract.add_argument("--output", required=True, help="输出目录")
    p_extract.add_argument("--fps", type=float, default=0.5)
    p_extract.add_argument("--width", type=int, default=config.PANORAMA_WIDTH)
    p_extract.set_defaults(func=_cmd_extract)

    # sfm
    p_sfm = sub.add_parser("sfm", help="等矩形 SfM")
    p_sfm.add_argument("--output", required=True)
    p_sfm.add_argument("--threads", type=int, default=8)
    p_sfm.set_defaults(func=_cmd_sfm)

    # cubemap
    p_cubemap = sub.add_parser("cubemap", help="展开 cube 面")
    p_cubemap.add_argument("--output", required=True)
    p_cubemap.set_defaults(func=_cmd_cubemap)

    # train
    p_train = sub.add_parser("train", help="3DGS 训练")
    p_train.add_argument("--output", required=True)
    p_train.add_argument("--iterations", type=int, default=config.TRAIN_ITERATIONS)
    p_train.set_defaults(func=_cmd_train)

    # export
    p_export = sub.add_parser("export", help="导出标准点云")
    p_export.add_argument("--output", required=True)
    p_export.add_argument("--iteration", type=int, default=config.TRAIN_ITERATIONS)
    p_export.set_defaults(func=_cmd_export)

    # verify
    p_verify = sub.add_parser("verify", help="质量评估")
    p_verify.add_argument("--output", required=True)
    p_verify.add_argument("--mode", choices=["train", "pano"], default="train")
    p_verify.add_argument("--iteration", type=int, default=config.TRAIN_ITERATIONS)
    p_verify.set_defaults(func=_cmd_verify)

    # preview
    p_preview = sub.add_parser("preview", help="渲染全景图预览")
    p_preview.add_argument("--output", required=True)
    p_preview.add_argument("--frame", type=int, default=None, help="渲染帧索引 (默认中间帧)")
    p_preview.add_argument("--iteration", type=int, default=config.TRAIN_ITERATIONS)
    p_preview.set_defaults(func=_cmd_preview)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
