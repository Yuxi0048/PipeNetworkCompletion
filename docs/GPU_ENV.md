# GPU / CUDA Environment

<!-- Workstream: Claude + Codex merge -->

This document covers the optional GPU-accelerated conda environment used for
training the anchor-free road-edge GNN. The repository ships with two parallel
environments:

| Env file              | Conda env name                    | Local prefix                        | PyTorch backend                |
| --------------------- | --------------------------------- | ----------------------------------- | ------------------------------ |
| `environment.yml`     | `pipe-network-completion`         | `.conda/pipe-network-completion`    | CPU only (`cpuonly`)           |
| `environment-cuda.yml`| `pipe-network-completion-cuda`    | `.conda/pipe-network-completion-cuda` | CUDA 12.1 (`pytorch-cuda=12.1`)|

The two envs coexist. CI, smoke tests, and checkpoint evaluation continue to
work against the CPU env; GPU training of the anchor-free GNN uses the CUDA
env. PyTorch (`2.1.0`) and `torch_geometric` (`2.5.3`) versions match across
both envs so saved state dicts are interchangeable.

## Prerequisites

- NVIDIA GPU with compute capability >= 3.5.
- NVIDIA driver new enough for CUDA 12.1 (driver >= 530 on Linux,
  driver >= 531 on Windows). Verify with `nvidia-smi`; the "CUDA Version"
  line shows the maximum CUDA runtime your driver supports; it must be
  >= 12.1.
- A system CUDA toolkit is **not** required. Conda installs its own runtime
  via `pytorch-cuda=12.1`.

This repo has been validated on:
- Windows 11, GeForce GTX 1080 Ti (Pascal, compute 6.1, 11 GB), driver 581.29.

## Install (Windows / PowerShell)

```powershell
.\scripts\create_env_cuda.ps1
.\.conda\pipe-network-completion-cuda\python.exe -m pip install -e .
.\.conda\pipe-network-completion-cuda\python.exe scripts\verify_environment_cuda.py
```

`create_env_cuda.ps1` defaults to:
- conda binary: `$env:USERPROFILE\miniforge3\Scripts\conda.exe`
- env prefix:  `.conda\pipe-network-completion-cuda`
- env file:    `environment-cuda.yml`

Pass `-CondaExe`, `-Prefix`, or `-EnvFile` to override.

## Install (Linux / macOS Linux runners)

```bash
conda env create -f environment-cuda.yml
conda activate pipe-network-completion-cuda
pip install -e .
python scripts/verify_environment_cuda.py
```

## Verifying the GPU is wired up

```bash
python scripts/verify_environment_cuda.py
```

The check:
1. imports every required Python package and prints versions
2. prints `torch.version.cuda`, cuDNN version, and `torch.cuda.is_available()`
3. lists every visible CUDA device (name, capability, free/total memory)
4. runs a tiny matmul on each device
5. runs `torch_geometric.nn.SAGEConv` on GPU
6. runs the anchor-free road-edge GNN through 2 epochs on GPU

If you only need a quick driver sanity check (no anchor-free training), pass:

```bash
python scripts/verify_environment_cuda.py --skip-anchor-free
```

To make missing CUDA a hard failure (useful in CI):

```bash
python scripts/verify_environment_cuda.py --strict
```

## Training on GPU

The anchor-free pipeline already defaults to `device: auto`, which resolves
to CUDA when available and CPU otherwise (`anchor_free.model.resolve_torch_device`).
After activating the CUDA env, training picks up the GPU with no flag changes:

```bash
.\.conda\pipe-network-completion-cuda\python.exe scripts\train_anchor_free.py \
    --config configs\anchor_free_isarc2024.yaml
```

To pin the device explicitly, set it in your config:

```yaml
model:
  type: gnn
  device: cuda          # or cuda:0, cpu, auto
  epochs: 200
```

To watch utilisation during training:

```powershell
nvidia-smi -l 2
```

## Why a second env instead of upgrading the existing one

- The original `environment.yml` is what `scripts/evaluate_checkpoint.py` and
  CI run against. Saved checkpoints (`models/checkpoints/*.pt`) were trained on
  this stack; we keep it unchanged for reproducibility.
- Conda cannot meaningfully ship "CPU and CUDA in the same env"; `cpuonly`
  and `pytorch-cuda` are mutually exclusive metapackages.
- A side-by-side install keeps the existing ~6 GB CPU env intact while you
  experiment with the GPU build (and lets you roll back by just deleting the
  `-cuda` prefix).

## Troubleshooting

**`torch.cuda.is_available()` is `False` after install.** Check the conda
output: if you see "package pytorch-cuda is not installed" it usually means
the `nvidia` channel was missing or the env solved to a CPU build. Re-create
the env instead of updating the CPU env in place.

**PyG wheels fail to install with "no matching distribution".** The wheel
index URL in `environment-cuda.yml` is pinned to `torch-2.1.0+cu121`. If you
change the PyTorch or CUDA version, change that URL too; the wheels are
ABI-bound to both.

**Out-of-memory on the 1080 Ti during training.** The default `hidden_dim=64,
num_layers=3` config is comfortable on 11 GB. If you scale up, lower the
batch size, the hidden dim, or run with `torch.cuda.amp.autocast()` (not yet
wired into the anchor-free trainer).
