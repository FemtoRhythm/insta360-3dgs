# insta360-3dgs

把 Insta360 相机输出的 `.insv` 双鱼眼视频，自动重建为全景 3D Gaussian Splatting（3DGS）并导出标准点云。

## 流程

```
.insv (双镜头)
   │ ① stitch  — MediaSDK 拼接为等矩形 mp4
   ▼
等矩形 mp4
   │ ② extract — ffmpeg 抽帧 (frame_%04d.jpg)
   ▼
等矩形帧
   │ ③ sfm     — COLMAP EQUIRECTANGULAR 相机模型重建相机轨迹
   ▼
等矩形位姿
   │ ④ cubemap — 每帧展开 6 个共光心透视面 (PINHOLE sparse)
   ▼
cube 面训练数据
   │ ⑤ train   — 3DGS 训练
   ▼
高斯模型
   │ ⑥ export  — 导出标准 x,y,z+rgb 点云
   ▼
pointcloud.ply
```

## 环境准备

- `uv`（Python 3.11 环境管理）
- `ffmpeg`（抽帧）
- Insta360 MediaSDK 3.1.5（`MediaSDKTest.exe` 离线拼接）
- 已有 `gaussian-splatting` 环境（含 torch CUDA + 已编译的 `diff-gaussian-rasterization` / `simple-knn`）
- `pycolmap`（cp311，等矩形 SfM）

```bash
uv sync
```

## 外部依赖路径

默认从 `config.py` 解析，可用环境变量覆盖：

| 环境变量 | 默认值 | 说明 |
| --- | --- | --- |
| `INSTA360_3DGS_MEDIASDK_EXE` | `../sdk/MediaSDK/bin/MediaSDKTest.exe` | 拼接可执行文件 |
| `INSTA360_3DGS_GS_DIR` | `../20260823_1/gaussian-splatting` | gaussian-splatting 代码目录 |
| `INSTA360_3DGS_PYLIBS` | `../20260823_1/pylibs` | pycolmap 预编译包目录 |
| `INSTA360_3DGS_RUNTIME_PYTHON` | `<GS_DIR>/.venv/Scripts/python.exe` | 重型阶段解释器 |

## 使用

一键端到端（输入 `.insv` 或包含它们的目录）：

```bash
uv run insta360-3dgs run --input VID_20260823_215314_00_006.insv --output out/
# 或整个目录
uv run insta360-3dgs run --input ./raw_clips --output out/ --verify
```

仅拼接为等矩形 mp4：

```bash
uv run insta360-3dgs stitch --input VID_20260823_215314_00_006.insv --output out/stitched
```

分步执行：

```bash
uv run insta360-3dgs extract --input out/stitched/xxx.mp4 --output out/
uv run insta360-3dgs sfm     --output out/ --threads 8
uv run insta360-3dgs cubemap --output out/
uv run insta360-3dgs train   --output out/ --iterations 30000
uv run insta360-3dgs export  --output out/ --iteration 30000
uv run insta360-3dgs verify  --output out/ --mode train
```

各阶段幂等：产物已存在时自动跳过。

## 主要参数

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--stitch-type` | `optflow` | 拼接算法：`template` / `optflow` / `dynamicstitch` / `aistitch` |
| `--output-size` | `3840x1920` | 拼接分辨率 `WxH` |
| `--fps` | `0.5` | 抽帧频率 |
| `--width` | `2880` | 全景帧宽度（等矩形高度 = 宽度 / 2） |
| `--iterations` | `30000` | 3DGS 训练迭代数 |
| `--threads` | `8` | SfM 线程数 |

## 输出

```
<output>/
  stitched/         拼接出的等矩形 mp4
  frames/           抽出的全景帧
  sfm/              等矩形 SfM (结果在 sfm/0/)
  cubemap/          cube 面训练数据 (images/ + sparse/0/)
  model/            3DGS 训练输出
  pointcloud.ply    最终标准点云 (x,y,z + rgb)
```
