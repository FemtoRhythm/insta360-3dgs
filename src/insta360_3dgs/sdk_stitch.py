# -*- coding: utf-8 -*-
"""用 MediaSDK 把 .insv 双鱼眼视频拼成等矩形 mp4。

双镜头按 `_00_`（前）/ `_10_`（后）成对保存，要成对传给 MediaSDKTest.exe
才能拼出完整的 360° 画面。
"""
from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

from . import config

# 匹配 Insta360 双镜头命名: <base>_00_<seq>.insv / <base>_10_<seq>.insv
_LENS = re.compile(r"^(?P<base>.+)_(?P<lens>00|10)_(?P<seq>\d+)\.insv$", re.IGNORECASE)


def _collect_insv(inputs: list[str]) -> list[Path]:
    """把文件/目录输入展开为 .insv 文件列表 (去重)。"""
    files: list[Path] = []
    seen: set[str] = set()
    for raw in inputs:
        p = Path(raw)
        if p.is_dir():
            candidates = sorted(p.glob("*.insv"))
        else:
            candidates = [p]
        for c in candidates:
            key = str(c.resolve())
            if c.suffix.lower() == ".insv" and key not in seen:
                seen.add(key)
                files.append(c)
    return files


def group_insv(files: list[Path]) -> list[tuple[str, list[Path]]]:
    """把 .insv 按拍摄片段分组, 返回 [(base_name, [front, back]), ...]。"""
    groups: dict[str, list[Path]] = {}
    singles: list[Path] = []
    for f in files:
        m = _LENS.match(f.name)
        if m:
            groups.setdefault(m.group("base"), []).append(f)
        else:
            singles.append(f)

    result: list[tuple[str, list[Path]]] = []
    for base, fs in groups.items():
        fs.sort(key=lambda p: (_LENS.match(p.name).group("lens"), p.name))
        result.append((base, fs))
    for f in singles:
        result.append((f.stem, [f]))
    return result


def stitch(files: list[Path], output_path: Path, stitch_type: str,
           width: int, height: int) -> subprocess.CompletedProcess:
    exe = config.MEDIASDK_EXE
    if not Path(exe).exists():
        raise SystemExit(f"MediaSDKTest.exe 不存在: {exe} (请检查 INSTA360_3DGS_MEDIASDK_EXE)")

    command = [
        str(exe),
        "-inputs", *[str(f) for f in files],
        "-output", str(output_path),
        "-stitch_type", stitch_type,
        "-output_size", f"{width}x{height}",
    ]
    print(f"[stitch] {' '.join(command)}", flush=True)
    return subprocess.run(command, check=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="insta360-3dgs stitch",
                                     description="MediaSDK 拼接 .insv -> 等矩形 mp4")
    parser.add_argument("--input", required=True, nargs="+",
                        help="一个或多个 .insv 文件或目录")
    parser.add_argument("--output", required=True, help="输出目录")
    parser.add_argument("--stitch-type", default=config.STITCH_TYPE,
                        choices=["template", "optflow", "dynamicstitch", "aistitch"],
                        help="拼接算法 (默认 optflow 接缝质量最佳)")
    parser.add_argument("--output-size", default=None,
                        help="输出分辨率 WxH (默认 %dx%d)"
                             % (config.STITCH_OUTPUT_WIDTH, config.STITCH_OUTPUT_HEIGHT))
    args = parser.parse_args(argv)

    width, height = config.STITCH_OUTPUT_WIDTH, config.STITCH_OUTPUT_HEIGHT
    if args.output_size:
        w, h = args.output_size.lower().split("x")
        width, height = int(w), int(h)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = _collect_insv(args.input)
    if not files:
        raise SystemExit("未找到任何 .insv 文件")
    groups = group_insv(files)

    for base, fs in groups:
        out_path = out_dir / f"{base}.mp4"
        if out_path.exists():
            print(f"[stitch] 跳过已存在: {out_path}", flush=True)
            continue
        print(f"[stitch] 拼接 {len(fs)} 个镜头 -> {out_path.name}", flush=True)
        stitch(fs, out_path, args.stitch_type, width, height)

    print(f"[stitch] done -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
