# insta360-3dgs

把 Insta360 拍出来的 `.insv` 双鱼眼视频，自动重建成全景 3D Gaussian Splatting，最后导出一个标准的 `x,y,z + rgb` 点云（PLY），可以直接扔进 CloudCompare、MeshLab 这类工具看。

## 它做了什么

Insta360 的双镜头是两颗鱼眼分开存的（`_00_` 前镜头、`_10_` 后镜头），要变成能用的 3D 结果，得一路处理：

1. 用官方 MediaSDK 把两颗鱼眼拼成一张 360° 等矩形视频
2. 抽帧
3. 用 COLMAP 的等矩形相机模型做位姿重建
4. 把每帧全景展开成 6 个共光心的透视面，喂给 3DGS
5. 训练高斯模型
6. 导出标准点云

### 为什么先做等矩形 SfM

全景图每个像素对应一个方向，光心在球心。如果图省事直接把 6 个 cube 面当成独立相机去做 SfM，共光心的面会被误当成自由相机，凭空多出伪基线，位姿会乱。所以先在等矩形空间用 EQUIRECTANGULAR 相机模型跑一遍 SfM 拿到平滑的相机轨迹，再基于这个轨迹展开 6 个透视面，保证每一帧的 6 个面光心都重合在同一点。

## 依赖

分两类：轻量依赖交给 uv 装，重型依赖复用机器上已经配好的环境。

| 依赖 | 用途 | 说明 |
| --- | --- | --- |
| `uv` | Python 环境管理 | `uv sync` 装轻量依赖 |
| `ffmpeg` | 抽帧 | 系统 PATH 里可用即可 |
| Insta360 MediaSDK | 拼接 `.insv` | 官方 SDK，手动放好 `MediaSDKTest.exe` |
| gaussian-splatting | 训练 | 复用已有 `.venv`（含 torch CUDA + 编译好的 `diff-gaussian-rasterization` / `simple-knn`） |
| `pycolmap`（cp311） | 等矩形 SfM | 预编译包目录 |

## 安装

```bash
git clone https://github.com/FemtoRhythm/insta360-3dgs
cd insta360-3dgs
uv sync
```

## 配置外部依赖

项目通过环境变量找重型依赖，默认路径都假设它们和仓库同级（`../`）。对不上的话用环境变量覆盖：

| 环境变量 | 默认值 | 说明 |
| --- | --- | --- |
| `INSTA360_3DGS_MEDIASDK_EXE` | `../sdk/MediaSDK/bin/MediaSDKTest.exe` | MediaSDK 拼接程序 |
| `INSTA360_3DGS_GS_DIR` | `../20260823_1/gaussian-splatting` | gaussian-splatting 目录 |
| `INSTA360_3DGS_PYLIBS` | `../20260823_1/pylibs` | pycolmap 预编译包 |
| `INSTA360_3DGS_RUNTIME_PYTHON` | `<GS_DIR>/.venv/Scripts/python.exe` | 跑重型阶段的解释器 |

## 使用

一条命令跑完，输入可以是单个 `.insv` 文件，也可以是装了一堆 `.insv` 的目录：

```bash
uv run insta360-3dgs run --input VID_20260823_215314_00_006.insv --output out/
# 或整个目录，训练完顺手做质量评估
uv run insta360-3dgs run --input ./raw_clips --output out/ --verify
```

想单独跑某一步也行，每个子命令都是幂等的（产物已存在就直接跳过）：

```bash
uv run insta360-3dgs stitch  --input ./raw_clips --output out/stitched
uv run insta360-3dgs extract --input out/stitched/xxx.mp4 --output out/
uv run insta360-3dgs sfm     --output out/ --threads 8
uv run insta360-3dgs cubemap --output out/
uv run insta360-3dgs train   --output out/ --iterations 30000
uv run insta360-3dgs export  --output out/ --iteration 30000
```

另外两个可选的子命令：`verify` 算 PSNR/SSIM/L1，`preview` 渲染全景图和原图对比。

## 常用参数

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--stitch-type` | `optflow` | 拼接算法：`template` / `optflow` / `dynamicstitch` / `aistitch` |
| `--output-size` | `3840x1920` | 拼接分辨率 `WxH` |
| `--fps` | `0.5` | 抽帧频率 |
| `--width` | `2880` | 全景帧宽度（高度 = 宽度 / 2） |
| `--iterations` | `30000` | 3DGS 训练迭代数 |
| `--threads` | `8` | SfM 线程数 |

## 输出结构

```
<output>/
  stitched/         拼接出的等矩形 mp4
  frames/           抽出的全景帧
  sfm/              等矩形 SfM（结果在 sfm/0/）
  cubemap/          cube 面训练数据（images/ + sparse/0/）
  model/            3DGS 训练输出
  pointcloud.ply    最终点云（x,y,z + rgb）
```

## 已知问题

- 双镜头文件必须成对（`_00_` 前 + `_10_` 后），只传一个会拼不完整；传目录时会自动配对。
- 训练阶段内部会切到 gaussian-splatting 目录，`--output` 记得传绝对路径（CLI 已经帮你处理了）。
- 想重跑某一步，删掉对应的输出子目录就行，比如 `rm -rf out/model` 再跑一次 train。
