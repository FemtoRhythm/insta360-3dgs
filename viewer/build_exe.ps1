# 打包点云查看器为单文件 exe（Windows）
# 用法: cd viewer ; .\build_exe.ps1
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

param([string]$Name = "PLYViewer")

Write-Host "==> 安装 PyInstaller"
uv pip install pyinstaller

Write-Host "==> 打包 $Name.exe"
uv run pyinstaller --onefile --windowed --name $Name viewer_app.py

Write-Host "==> 完成: dist\$Name.exe"
