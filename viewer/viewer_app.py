# -*- coding: utf-8 -*-
"""3DGS 点云查看器桌面应用。

功能：选择 PLY 点云文件后，启动本地 HTTP 服务并在浏览器中打开交互式 3D 视图。

打包：pyinstaller --onefile --windowed --name PLYViewer viewer_app.py
"""
import os
import shutil
import socket
import sys
import tempfile
import threading
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import tkinter as tk
from tkinter import filedialog, messagebox

# 内嵌的查看器页面（three.js 从 CDN 加载，需要联网）
INDEX_HTML = r'''<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Insta360 3DGS 点云查看器</title>
<style>
  html, body { margin: 0; height: 100%; overflow: hidden; background: #1a1a1a; font-family: system-ui, sans-serif; }
  #loading { position: absolute; top: 50%; left: 50%; transform: translate(-50%,-50%); color: #fff; font-size: 16px; z-index: 10; }
  #info { position: absolute; top: 12px; left: 12px; color: #ddd; font-size: 13px; background: rgba(0,0,0,0.55); padding: 8px 14px; border-radius: 6px; z-index: 10; }
  #hint { position: absolute; top: 12px; right: 12px; color: #aaa; font-size: 12px; background: rgba(0,0,0,0.55); padding: 8px 14px; border-radius: 6px; z-index: 10; }
  #panel { position: absolute; bottom: 18px; left: 50%; transform: translateX(-50%); background: rgba(0,0,0,0.6); padding: 10px 18px; border-radius: 8px; color: #eee; display: flex; gap: 20px; align-items: center; font-size: 13px; z-index: 10; }
  #panel label { display: flex; align-items: center; gap: 8px; }
  input[type=range] { width: 140px; }
</style>
</head>
<body>
<div id="loading">加载点云中…</div>
<div id="info">Insta360 3DGS 点云</div>
<div id="hint">左键自由旋转（可任意角度） · 滚轮缩放 · 右键平移</div>
<div id="panel">
  <label>点大小 <input id="size" type="range" min="0.01" max="0.8" step="0.01" value="0.15"></label>
  <label><input id="autoRotate" type="checkbox" checked> 自动旋转</label>
</div>

<script type="importmap">
{
  "imports": {
    "three": "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
    "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"
  }
}
</script>
<script type="module">
import * as THREE from 'three';
import { TrackballControls } from 'three/addons/controls/TrackballControls.js';
import { PLYLoader } from 'three/addons/loaders/PLYLoader.js';

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x1a1a1a);

const camera = new THREE.PerspectiveCamera(60, innerWidth / innerHeight, 0.01, 1000);
camera.position.set(0, 0, 50);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(innerWidth, innerHeight);
document.body.appendChild(renderer.domElement);

const controls = new TrackballControls(camera, renderer.domElement);
controls.rotateSpeed = 2.5;
controls.zoomSpeed = 1.2;
controls.panSpeed = 0.8;
controls.dynamicDampingFactor = 0.2;
controls.staticMoving = false;

let points = null;
let autoRotate = true;
const sizeInput = document.getElementById('size');
const autoRotateInput = document.getElementById('autoRotate');

const loader = new PLYLoader();
loader.load('pointcloud.ply', (geometry) => {
  document.getElementById('loading').style.display = 'none';

  geometry.computeBoundingBox();
  const box = geometry.boundingBox;
  const center = box.getCenter(new THREE.Vector3());
  const extent = box.getSize(new THREE.Vector3()).length();

  geometry.translate(-center.x, -center.y, -center.z);
  const s = 40 / extent;
  geometry.scale(s, s, s);
  geometry.computeBoundingBox();

  const material = new THREE.PointsMaterial({
    size: 0.15,
    vertexColors: true,
    sizeAttenuation: true,
  });
  points = new THREE.Points(geometry, material);
  scene.add(points);

  camera.position.set(0, 0, 50);
  controls.target.set(0, 0, 0);
  controls.update();

  document.getElementById('info').textContent =
    `点云：${geometry.attributes.position.count.toLocaleString()} 点`;
}, undefined, (err) => {
  document.getElementById('loading').textContent = '加载失败：' + err;
});

sizeInput.addEventListener('input', () => {
  if (points) points.material.size = parseFloat(sizeInput.value);
});
autoRotateInput.addEventListener('change', () => {
  autoRotate = autoRotateInput.checked;
});

addEventListener('resize', () => {
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
});

function animate() {
  requestAnimationFrame(animate);
  if (autoRotate) controls.rotateLeft(0.006);
  controls.update();
  renderer.render(scene, camera);
}
animate();
</script>
</body>
</html>
'''


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _make_handler(directory: str):
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=directory, **kwargs)

        def log_message(self, *args):
            pass  # 静默日志，避免干扰 GUI

    return Handler


class ViewerApp:
    def __init__(self):
        self.workdir = tempfile.mkdtemp(prefix="ply_viewer_")
        self._write_index()
        self.server = None
        self.port = None

        self.root = tk.Tk()
        self.root.title("3DGS 点云查看器")
        self.root.geometry("620x300")
        self.root.resizable(False, False)
        self.root.attributes("-topmost", True)

        self._build_ui()
        self._start_server()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _write_index(self):
        with open(os.path.join(self.workdir, "index.html"), "w", encoding="utf-8") as f:
            f.write(INDEX_HTML)

    def _build_ui(self):
        tk.Label(self.root, text="3DGS 点云查看器",
                 font=("微软雅黑", 16, "bold")).pack(pady=(28, 8))
        tk.Label(self.root, text="选择一个 PLY 点云文件，将在浏览器中打开交互式 3D 视图",
                 font=("微软雅黑", 10), fg="#666").pack()

        self.btn = tk.Button(self.root, text="选择 PLY 文件…",
                             font=("微软雅黑", 12), command=self._choose,
                             padx=28, pady=10)
        self.btn.pack(pady=22)

        self.status = tk.StringVar(value="尚未加载文件")
        tk.Label(self.root, textvariable=self.status,
                 font=("微软雅黑", 10), fg="#333").pack()

        tk.Label(self.root, text="提示：three.js 从 CDN 加载，首次打开需联网",
                 font=("微软雅黑", 9), fg="#999").pack(side="bottom", pady=14)

    def _start_server(self):
        self.port = _free_port()
        handler = _make_handler(self.workdir)
        self.server = ThreadingHTTPServer(("127.0.0.1", self.port), handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def _choose(self):
        path = filedialog.askopenfilename(
            title="选择 PLY 点云文件",
            filetypes=[("PLY 点云", "*.ply"), ("所有文件", "*.*")],
        )
        if path:
            self._load(path)

    def _load(self, path):
        if not os.path.isfile(path):
            messagebox.showerror("错误", "文件不存在：\n" + path)
            return
        dst = os.path.join(self.workdir, "pointcloud.ply")
        try:
            shutil.copyfile(path, dst)
        except OSError as e:
            messagebox.showerror("错误", "复制文件失败：\n" + str(e))
            return
        url = f"http://127.0.0.1:{self.port}/"
        webbrowser.open(url, new=2)
        self.status.set(f"已加载：{os.path.basename(path)}")

    def _on_close(self):
        if self.server:
            self.server.shutdown()
            self.server.server_close()
        shutil.rmtree(self.workdir, ignore_errors=True)
        self.root.destroy()

    def run(self):
        self.root.mainloop()


def main():
    try:
        app = ViewerApp()
    except Exception as e:  # --windowed 下无控制台，落盘便于排查
        log = os.path.join(tempfile.gettempdir(), "ply_viewer_error.log")
        try:
            with open(log, "w", encoding="utf-8") as f:
                f.write(repr(e))
        except OSError:
            pass
        try:
            messagebox.showerror("启动失败", str(e) + f"\n详情见 {log}")
        except Exception:
            pass
        return
    if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
        app.root.after(500, lambda: app._load(sys.argv[1]))
    app.run()


if __name__ == "__main__":
    main()
