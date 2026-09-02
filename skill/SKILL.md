---
name: insta360-3dgs
description: 'Insta360 .insv 双鱼眼视频到全景 3D Gaussian Splatting 点云的重建流水线。当用户提到 insta360、insv、全景重建、3DGS、gaussian splatting、全景点云、全景高斯时使用。基于 MediaSDK 拼接 + pycolmap 等矩形 SfM + 3DGS 训练，uv 管理环境。'
license: MIT
allowed-tools: Bash
---

# Insta360 → 全景 3DGS

把 Insta360 相机输出的 `.insv` 双鱼眼视频，自动重建为全景 3D Gaussian Splatting（3DGS）并导出标准点云（`x,y,z + rgb` 的 PLY）。

## 流水线

```
.insv (双镜头) → ① stitch 等矩形 mp4 → ② extract 抽帧 → ③ sfm 等矩形位姿
             → ④ cubemap 展开 6 面 → ⑤ train 3DGS → ⑥ export 点云
```

源码单一真源在 GitHub 仓库 `FemtoRhythm/insta360-3dgs`，本 skill 采用「引用式」：不内嵌源码，由引导脚本定位/拉取并运行。

## 触发场景

- 用户提供 `.insv` 文件或含 `.insv` 的目录，要求重建全景/点云/3DGS
- 用户想跑通 Insta360 全景高斯泼溅流水线
- 用户要求"预览全景"或"导出点云"

## 前置依赖（外部，脚本只检测不自动安装）

| 依赖 | 用途 | 说明 |
| --- | --- | --- |
| `uv` | Python 环境管理 | 轻量依赖用 `uv sync` 装 |
| `ffmpeg` | 抽帧 | PATH 中可用即可 |
| Insta360 MediaSDK (`MediaSDKTest.exe`) | 离线拼接 .insv | 大体积 SDK，需用户预先放置 |
| gaussian-splatting 环境（含 torch CUDA + 已编译 `diff-gaussian-rasterization`/`simple-knn`） | 训练 | 复用已有 `.venv` |
| `pycolmap` (cp311) | 等矩形 SfM | 预编译包目录 |

这些重型依赖由项目 `config.py` 通过环境变量解析，脚本会检测并注入正确路径。

## 工作流程（AI 执行顺序）

### 1. 诊断环境

```bash
python <skill目录>/scripts/bootstrap.py check
```

输出每个依赖的命中/缺失状态。根据结果决定下一步。

### 2. 准备源码与轻量依赖

```bash
python <skill目录>/scripts/bootstrap.py setup
```

内部逻辑：定位本地源码 → 缺失则 `git clone https://github.com/FemtoRhythm/insta360-3dgs` → `uv sync` 安装轻量依赖。

### 3. 运行端到端

```bash
python <skill目录>/scripts/bootstrap.py run --input <xx.insv或目录> --output <输出目录> [--iterations 30000] [--verify]
```

等价于在项目内执行 `uv run insta360-3dgs run ...`，脚本会自动把外部依赖路径注入环境变量。

### 4.（可选）分步执行 / 预览

参考 `references/pipeline.md`。

## 环境变量

脚本与 `config.py` 共用同一套约定，均可被环境变量覆盖：

| 环境变量 | 默认值（相对项目父目录） | 说明 |
| --- | --- | --- |
| `INSTA360_3DGS_PROJECT_DIR` | — | 源码项目根（引导脚本用） |
| `INSTA360_3DGS_MEDIASDK_EXE` | `<parent>/sdk/MediaSDK/bin/MediaSDKTest.exe` | 拼接可执行文件 |
| `INSTA360_3DGS_GS_DIR` | `<parent>/20260823_1/gaussian-splatting` | gaussian-splatting 代码目录 |
| `INSTA360_3DGS_PYLIBS` | `<parent>/20260823_1/pylibs` | pycolmap 预编译包目录 |
| `INSTA360_3DGS_RUNTIME_PYTHON` | `<GS_DIR>/.venv/Scripts/python.exe` | 重型阶段解释器 |

## 关键约束与已知坑

- 双镜头文件必须成对：`<base>_00_<seq>.insv`（前）+ `<base>_10_<seq>.insv`（后）。
- 训练阶段会 `os.chdir` 到 gaussian-splatting 目录，`--output` 必须传绝对路径（脚本已处理）。
- 各阶段幂等：产物存在即跳过。
- 详见 `references/troubleshooting.md`。
