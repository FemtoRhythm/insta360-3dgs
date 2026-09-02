---
name: insta360-3dgs
description: 'Insta360 .insv 双鱼眼视频到全景 3D Gaussian Splatting 点云的重建流水线。基于 MediaSDK 拼接 + pycolmap 等矩形 SfM + 3DGS 训练。'
license: MIT
allowed-tools: Bash
---

# insta360-3dgs

把 Insta360 的 `.insv` 双鱼眼视频重建为全景 3DGS 点云。源码单一真源在 GitHub 仓库 `FemtoRhythm/insta360-3dgs`，本 skill 用引导脚本定位/拉取源码并运行，不内嵌源码。

## 用法

```bash
python <skill目录>/scripts/bootstrap.py check    # 诊断环境
python <skill目录>/scripts/bootstrap.py setup    # 拉取源码 + uv sync
python <skill目录>/scripts/bootstrap.py run --input <xx.insv或目录> --output <输出目录>
```

完整的流水线说明、环境变量、排错都在仓库的 README 里。
