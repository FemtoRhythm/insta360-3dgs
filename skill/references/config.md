# 配置与环境变量

外部依赖路径统一由 `src/insta360_3dgs/config.py` 解析，均可被环境变量覆盖。

## 环境变量

| 环境变量 | 默认值 | 说明 |
| --- | --- | --- |
| `INSTA360_3DGS_MEDIASDK_EXE` | `<项目父目录>/sdk/MediaSDK/bin/MediaSDKTest.exe` | MediaSDK 拼接可执行文件 |
| `INSTA360_3DGS_GS_DIR` | `<项目父目录>/20260823_1/gaussian-splatting` | gaussian-splatting 代码目录 |
| `INSTA360_3DGS_PYLIBS` | `<项目父目录>/20260823_1/pylibs` | pycolmap 4.1.1 (cp311) 预编译包 |
| `INSTA360_3DGS_RUNTIME_PYTHON` | `<GS_DIR>/.venv/Scripts/python.exe` | 重型阶段解释器（torch CUDA + pycolmap） |

引导脚本 `bootstrap.py` 另识别：

| 环境变量 | 说明 |
| --- | --- |
| `INSTA360_3DGS_PROJECT_DIR` | 源码项目根，缺失时脚本会 git clone |

## 关键几何/训练参数（config.py 常量）

| 常量 | 值 | 说明 |
| --- | --- | --- |
| `PANORAMA_WIDTH` / `HEIGHT` | 2880 / 1440 | 等矩形全景分辨率 |
| `FACE_SIZE` | 1536 | cube 面分辨率 |
| `FACE_FOCAL` | 768.0 | 90° FOV 焦距 |
| `TRAIN_ITERATIONS` | 30000 | 训练迭代数 |
| `TRAIN_RESOLUTION` | 2 | 训练降采样倍数 |
| `INIT_SCALE_MUL` / `INIT_OPACITY` | 0.25 / 0.5 | 高斯初始化参数 |
| `STITCH_OUTPUT_WIDTH` / `HEIGHT` | 3840 / 1920 | 拼接分辨率 |
| `EXPORT_OPACITY_THRESHOLD` | 0.1 | 点云导出不透明度阈值 |

## 输出目录结构

```
<output>/
  stitched/         拼接出的等矩形 mp4
  frames/           抽出的全景帧 frame_%04d.jpg
  sfm/              等矩形 SfM（结果在 sfm/0/）
  cubemap/          cube 面训练数据（images/ + sparse/0/）
  model/            3DGS 训练输出
  logs/             日志
  pointcloud.ply    最终标准点云
```

## 环境架构说明

轻量阶段（stitch/extract）在 uv 环境内执行，仅依赖 numpy/scipy/Pillow/plyfile。
重型阶段（sfm/cubemap/train/export/verify）由 `_dispatch.py` 用 `RUNTIME_PYTHON` 重执行，
并注入 `PYTHONPATH` 指向本项目 `src/`，复用已有 gaussian-splatting 环境的 torch + 已编译扩展。
