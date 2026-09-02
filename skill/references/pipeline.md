# 流水线分阶段命令

在项目根目录下，用 `uv run insta360-3dgs <子命令>` 分步执行。各阶段幂等，产物存在即跳过。

## ① stitch — MediaSDK 拼接

`.insv`（双镜头）→ 等矩形 mp4。双镜头文件必须成对传入。

```bash
uv run insta360-3dgs stitch --input VID_xxx_00_006.insv VID_xxx_10_006.insv --output out/stitched
# 或传目录自动配对
uv run insta360-3dgs stitch --input ./raw_clips --output out/stitched
```

参数：`--stitch-type`（template/optflow/dynamicstitch/aistitch，默认 optflow）、`--output-size`（`WxH`，默认 3840x1920）。

## ② extract — 抽帧

```bash
uv run insta360-3dgs extract --input out/stitched/xxx.mp4 --output out/ --fps 0.5 --width 2880
```

## ③ sfm — 等矩形 SfM（重型，走 runtime python）

```bash
uv run insta360-3dgs sfm --output out/ --threads 8
```

结果在 `out/sfm/0/`（cameras.bin 用 EQUIRECTANGULAR 模型，model_id=17）。

## ④ cubemap — 展开 6 面（重型）

```bash
uv run insta360-3dgs cubemap --output out/
```

从每帧等矩形位姿渲染 6 个共光心 PINHOLE 面，输出到 `out/cubemap/`。

## ⑤ train — 3DGS 训练（重型）

```bash
uv run insta360-3dgs train --output out/ --iterations 30000
```

训练输出在 `out/model/`，checkpoint 保存在 7000/15000/30000 迭代。

## ⑥ export — 导出标准点云（重型）

```bash
uv run insta360-3dgs export --output out/ --iteration 30000
```

产出 `out/pointcloud.ply`（`x,y,z + rgb`）。

## ⑦ verify — 质量评估（可选）

```bash
uv run insta360-3dgs verify --output out/ --mode train --iteration 30000
# 全景端到端对比
uv run insta360-3dgs verify --output out/ --mode pano
```

## ⑧ preview — 全景渲染预览（可选）

```bash
uv run insta360-3dgs preview --output out/ --iteration 30000
```

生成 `out/model/pano_render.jpg` / `pano_gt.jpg` / `compare.jpg`。

## 一键端到端

```bash
uv run insta360-3dgs run --input ./raw_clips --output out/ --verify
```
