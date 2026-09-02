# 已知坑与排查

## 1. 双镜头文件必须成对

MediaSDK 拼接要求 `<base>_00_<seq>.insv`（前置）与 `<base>_10_<seq>.insv`（后置）成对出现。
只传一个会报错或产出不完整。传目录时脚本用 `_00_` / `_10_` 正则自动配对。

## 2. `Could not recognize scene type!`（train 阶段）

根因：`train.py` 内部 `os.chdir(GAUSSIAN_SPLATTING_DIR)` 改变了 cwd，相对路径 `output/cubemap` 被解析到 gaussian-splatting 目录。

修复：`--output` 必须传绝对路径。`_dispatch.py` 的 `_absolutize()` 已自动处理；`bootstrap.py run` 也会把 `--output` 转绝对路径。

## 3. 运行时解释器不存在

报错 `运行时解释器不存在: ...`，说明 `<GS_DIR>/.venv/Scripts/python.exe` 未命中。

排查：
- 确认 gaussian-splatting 环境已创建（含 torch CUDA + 已编译 `diff-gaussian-rasterization`/`simple-knn`）
- 用 `INSTA360_3DGS_RUNTIME_PYTHON` 指向正确解释器

## 4. pycolmap 导入失败（sfm 阶段）

需 `pycolmap` 4.1.1 (cp311) 预编译包在 `PYLIBS_DIR` 中。检查 `INSTA360_3DGS_PYLIBS` 是否指向含 pycolmap 的目录，且与 runtime python 版本匹配（cp311）。

## 5. 阶段产物已存在即跳过（幂等）

如需重跑某阶段，删除对应输出子目录即可，例如：

```bash
# 重跑训练
rm -rf out/model
```

## 6. GitHub 主站网络不可达

clone 失败时，`api.github.com` 通常仍可达。可改用：

```bash
git clone https://api.github.com/repos/FemtoRhythm/insta360-3dgs
```

或用已下载的本地源码 + `INSTA360_3DGS_PROJECT_DIR` 指定，跳过 clone。

## 7. MediaSDKTest.exe 未找到

默认路径 `<项目父目录>/sdk/MediaSDK/bin/MediaSDKTest.exe`。若 SDK 解压到别处，用 `INSTA360_3DGS_MEDIASDK_EXE` 指定绝对路径。
