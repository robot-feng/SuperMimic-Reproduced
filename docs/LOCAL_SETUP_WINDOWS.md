# 本机环境（Windows + 项目 `.venv` + 单张 RTX 5060）

## 使用项目环境

本机项目环境是仓库根目录下的 `.venv`（Python 3.12.13）；使用项目自己的 Python 和 `uv`。Conda 的 `mjlab` 环境是另一套环境，当前检查为 Python 3.10.18、torch 2.12.1+cu130，且缺少本仓库的 MuJoCo/Warp 等依赖，因此不适用于 SuperMimic。保留它，不要复制或原地升级。

在 PowerShell 中进入仓库根目录，并先将 uv 下载缓存、Warp kernel 缓存和 TorchInductor 编译缓存指到 D 盘。示例使用 `D:\MinicondaEnv\uv-cache`、`D:\MinicondaEnv\warp-cache` 和 `D:\MinicondaEnv\torchinductor-cache`，也可以改成其他 D 盘目录：

```powershell
cd D:\PycharmProjects\SuperMimic
$env:UV_CACHE_DIR = "D:\MinicondaEnv\uv-cache"
$env:WARP_CACHE_PATH = "D:\MinicondaEnv\warp-cache"
$env:TORCHINDUCTOR_CACHE_DIR = "D:\MinicondaEnv\torchinductor-cache"
New-Item -ItemType Directory -Force -Path $env:UV_CACHE_DIR, $env:WARP_CACHE_PATH, $env:TORCHINDUCTOR_CACHE_DIR | Out-Null
uv cache dir
.\.venv\Scripts\python.exe --version
uv sync --group dev
```

`UV_CACHE_DIR` 控制 uv 下载缓存，`WARP_CACHE_PATH` 控制 Warp kernel 缓存，`TORCHINDUCTOR_CACHE_DIR` 控制 `torch.compile` 的编译缓存（本机默认会写入 C 盘）。`uv sync` 会在当前项目的 `.venv` 中安装 `uv.lock` 锁定的依赖。如果安装中断，设置好 D 盘缓存后重新运行 `uv sync --group dev`。依赖量较大，安装完成后确认 CUDA、MuJoCo 和 Warp 可用：

```powershell
uv run python -c "import torch, mujoco, warp, mjlab; print('Torch/CUDA:', torch.__version__, torch.version.cuda, torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'unavailable'); print('MuJoCo/Warp:', mujoco.__version__, warp.__version__)"
uv run sm-train --help
uv run python scripts/data/csv_to_npz_batch.py --help
```

如果 `PYTHONPATH` 影响导入，可在当前 PowerShell 窗口清除：

```powershell
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
```

如果希望以后新开的 PowerShell 也默认使用 D 盘缓存，可运行一次：

```powershell
[Environment]::SetEnvironmentVariable("UV_CACHE_DIR", "D:\MinicondaEnv\uv-cache", "User")
[Environment]::SetEnvironmentVariable("WARP_CACHE_PATH", "D:\MinicondaEnv\warp-cache", "User")
```

设置持久变量后，新开一个终端；若选用其他缓存位置，把两处路径一并替换。

## 单片段训练冒烟

仓库没有附数据。先按 README 的数据来源说明自行准备 LAFAN1 G1 CSV，并把 1 个片段转换成 npz：

```powershell
uv run python scripts/data/csv_to_npz_batch.py --input-dir data/raw/lafan1/g1 --output-dir data/npz/lafan1 --input-fps 30 --workers 2 --limit 1
```

有了 `data/npz/lafan1` 后，用这组小参数确认仿真采集、回放采样、世界模型更新和策略更新能串起来：

```powershell
uv run sm-train --run-name local-smoke --data.npz-dir data/npz/lafan1 --data.limit 1 `
  --env.num-envs 64 --env.replay-capacity 128 `
  --world-model.hidden 512 512 `
  --wm-trainer.batch-size 32 --wm-trainer.batches-per-update 1 `
  --policy-trainer.batch-size 32 --policy-trainer.horizon 8 `
  --policy-trainer.critic-epochs 1 --policy-trainer.critic-minibatches 1 `
  --schedule.total-iterations 100 --schedule.warmup-iterations 16 `
  --schedule.log-interval 10 --schedule.checkpoint-interval 100
```

这是缩小后的管线检查，不是论文结果。全量 BONES-SEED 紧凑动作库约 6.7 GB 且常驻显存，RTX 5060 的 8 GB 显存不能同时容纳它和训练状态。当前机器已下载并转换完整的 40 条 LAFAN1（2.451 小时），可用下面的显存缩放配置做完整数据集训练；它与论文的 4096 环境正式设置不等价，结果应按本机单卡配置报告：

```powershell
uv run sm-train --run-name lafan1-rtx5060-eager --log-root logs --data.npz-dir data/npz/lafan1 `
  --env.num-envs 512 `
  --wm-trainer.batch-size 1024 --wm-trainer.compile None `
  --policy-trainer.batch-size 1024 --policy-trainer.horizon 32 `
  --policy-trainer.compile None `
  --schedule.max-hours 3 --schedule.log-interval 50 --schedule.checkpoint-interval 500
```

这个配置保留完整 LAFAN1、默认网络宽度和 32 步想象视野，batch 为 1024/1024；不会把这组结果称作论文硬件配置下的复现。策略 batch 2048 曾在 actor 反向传播阶段耗尽 8 GiB 显存，所以当前配方采用此前能完成首个日志窗口的 1024。Windows 环境带有与 PyTorch 2.14 匹配的 `triton-windows`，但本机 RTX 5060 上完整训练图编译超过 20 分钟仍未出首个日志点，因此此配方显式关闭 `torch.compile`。普通训练会避免逐环境步同步 GPU，把 episode 汇总和训练指标留在 GPU 上，按日志窗口一起回传。仅在要精确拆分阶段耗时时追加 `--policy-trainer.profile-timing`；它会强制同步，适合短时诊断，不建议用于正式长跑。

## Linux / WSL 备选

mjlab 官方安装说明更推荐 Linux + NVIDIA GPU 训练；Windows 支持仍属初步测试。如果原生 Windows 的 MuJoCo Warp 工作流遇到平台问题，可改在 WSL 2 Ubuntu 中运行。WSL 2 CUDA 使用 Windows NVIDIA 驱动；参见 [微软 WSL CUDA 指南](https://learn.microsoft.com/windows/ai/directml/gpu-cuda-in-wsl)、[mjlab 安装说明](https://mujocolab.github.io/mjlab/v1.6.0/source/installation.html) 和 [mjlab 平台 FAQ](https://mujocolab.github.io/mjlab/v1.6.0/source/faq.html)。WSL 不是当前首选设置步骤。
