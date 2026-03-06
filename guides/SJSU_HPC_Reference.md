# SJSU CoE HPC Reference — Thesis Environment

*Last updated: March 4, 2026*

## Cluster Overview

The SJSU College of Engineering HPC is a Linux cluster managed by SLURM. Login node runs CentOS 7 (GLIBC 2.17). GPU nodes run Rocky 9 (newer GLIBC). This OS mismatch affects binary compatibility — some tools (Playwright, Apptainer) only work on GPU nodes.

- **Login node**: `coe-hpc1.sjsu.edu`. Has internet access. Used for file management, downloads, pip/conda installs, job submission. **Never run heavy compute here.**
- **GPU node g17**: Accessible directly via `ssh -X coe-hpc3` from login node. Has NVIDIA A40 48GB. No internet. This is NOT a SLURM allocation — use for quick tests only. For real workloads, use `sbatch`.
- **GPU nodes** (`g1`–`g19+`): No internet. Allocated via SLURM for batch jobs.
- **CPU compute nodes** (`c1`–`c20`, `cs001`, etc.): No GPU. Do not use for inference.

### How to Tell Where You Are

Look at the shell prompt:
- `[017557527@coe-hpc1 ~]$` → login node (internet, no GPU)
- `[017557527@g17 ~]$` → GPU node (GPU, no internet)
- `[017557527@cs001 ~]$` → CPU node (no GPU, no internet)

### SSH Access

```bash
ssh 017557527@coe-hpc.sjsu.edu   # lands on login node (coe-hpc1)
ssh -X coe-hpc3                   # from login node, direct access to g17 (quick tests)
# Requires SJSU VPN if off-campus
```

## GPU Hardware

| GPU | Nodes | VRAM | Notes |
|-----|-------|------|-------|
| A100 | 18 | 40-80GB | Best option for 7B models |
| H100 | 5 | 80GB | Excellent |
| A40 | 1 (g17) | 48GB | Confirmed working, all 3 thesis models fit |
| V100 | 3 | 16-32GB | Works for 3B model |
| P100 | 17 | 12GB | Too small for 7B models |

Confirmed: NVIDIA A40 with 48GB VRAM, CUDA 12.6 driver. All three thesis models (Llama-3.2-3B ~7GB, Qwen-2.5-7B ~15GB, Mistral-7B ~15GB in fp16) fit comfortably.

## SLURM Partitions

| Partition | Max Time | Use Case |
|-----------|----------|----------|
| `defq` | 4 hours | Default CPU queue |
| `cpuqs` | 5 days | Short CPU |
| `cpuqm` | 14 days | Medium CPU |
| `cpuql` | 21 days | Long CPU |
| `gpuqs` | 2 days | Short GPU — quick tests, pilot runs |
| `gpuqm` | 7 days | Medium GPU — main trace collection |
| `gpuql` | 14 days | Long GPU — extended collection runs |
| `condo` | 30 days | Faculty-owned GPU nodes (may be preempted) |

**IMPORTANT**: The partition name is NOT `gpu`. It's `gpuqs`, `gpuqm`, or `gpuql`. Using `-p gpu` will error.

### Requesting an Interactive GPU Session

```bash
srun -p gpuqs --gres=gpu -n 1 -N 1 -c 4 --mem=64G --pty /bin/bash
```

### Submitting a Batch Job

```bash
#!/bin/bash
#SBATCH --job-name=trace-collect
#SBATCH --output=logs/trace-collect-%j.log
#SBATCH --error=logs/trace-collect-%j.err
#SBATCH --partition=gpuqm
#SBATCH --gres=gpu
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH --mail-user=YOUR_EMAIL@sjsu.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# Environment setup — use activate_thesis alias contents
module load anaconda/3.9
source activate thesis
export PATH="/home/017557527/.conda/envs/thesis/bin:$PATH"

# Required for MiniWoB
export MINIWOB_URL="file:///home/017557527/cmpe299b/miniwob-plusplus/miniwob/html/miniwob/"

# Pre-flight checks
echo "=== Job started at $(date) ==="
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import torch; assert torch.cuda.is_available(), 'No CUDA'"

# Run
cd /home/017557527/agent_xes
python scripts/collect_traces.py "$@"

echo "=== Job finished at $(date) ==="
```

Submit: `sbatch scripts/collect_traces.sh`
Check status: `squeue -u 017557527`
Cancel: `scancel <JOB_ID>`
Post-job stats: `sacct -j <JOB_ID> --format=JobID,State,Elapsed,MaxRSS`

## Thesis Conda Environment

### Python Version

The thesis conda env uses **Python 3.11**. Created with:
```bash
module load anaconda/3.9
conda create -n thesis python=3.11 -y
```

If `conda create` gives Python 3.9 instead, force conda-forge:
```bash
conda create -n thesis python=3.11 --override-channels -c conda-forge -y
```

### Activation (run on ANY node)

```bash
activate_thesis
```

This alias (defined in `~/.bashrc`) runs:
```bash
module load anaconda/3.9 && source activate thesis && export PATH=/home/017557527/.conda/envs/thesis/bin:$PATH
```

The explicit PATH export is required because `source activate thesis` does not properly set PATH on this HPC — `which python` will point to the system Python at `/opt/ohpc/pub/apps/anaconda/3.9/bin/python` instead of the conda env's Python 3.11.

**IMPORTANT**:
- Do NOT use `conda activate thesis` — fails with `CommandNotFoundError`
- Do NOT use `module load python3/3.12.12` or `module load ml/torch/2.6` — these override the conda Python and break package resolution
- Always verify with `which python` → should show `~/.conda/envs/thesis/bin/python`

### Installed Packages

All packages installed via conda or `python -m pip install` into the conda env:
- `numpy` — installed via conda (pip compilation fails due to GCC 7.3 on login node)
- `transformers`, `accelerate`, `huggingface_hub` — via pip
- `torch`, `torchvision`, `torchaudio` — via pip with `--index-url https://download.pytorch.org/whl/cu121`
- `browsergym-miniwob` — via pip
- `playwright` — via pip
- `gymnasium` — installed via conda (`conda install -c conda-forge gymnasium`)

### Installing New Packages

Always install from the **login node** (has internet). Use `python -m pip install` NOT bare `pip install` to avoid leaking to system Python:

```bash
activate_thesis
python -m pip install <package>
```

If pip tries to compile C extensions and fails with GCC errors, install via conda first:
```bash
conda install -c conda-forge <package> -y
```

The `gnu14/14.2.0` module listed in older references does NOT exist on this HPC. Available GCC modules: `gnu/5.4.0`, `gnu7/7.3.0`, `gnu8/8.3.0`. For numpy/scipy, use conda to avoid compilation entirely.

## Playwright / Chromium

Playwright's bundled Node.js requires GLIBC 2.25+ which the login node (CentOS 7, GLIBC 2.17) does not have. Chromium was installed manually.

### How It Was Set Up

1. Downloaded Chromium zip on login node (has internet):
   ```bash
   wget -O /tmp/chromium-linux.zip https://playwright.azureedge.net/builds/chromium/1117/chromium-linux.zip
   ```

2. Extracted to Playwright's expected directory:
   ```bash
   mkdir -p ~/.cache/ms-playwright/chromium-1117
   cd ~/.cache/ms-playwright/chromium-1117
   unzip /tmp/chromium-linux.zip
   ```

3. Browser binary location: `~/.cache/ms-playwright/chromium-1117/chrome-linux/chrome`

Since `~/.cache/` is on shared NFS, this works from any node (login, GPU, compute).

### Verified Working

Chromium headless launches successfully on g17 (Rocky 9). The `playwright install chromium` command does NOT work from either node — login node has old GLIBC, GPU node has no internet. Manual download is the only path.

## MiniWoB++ Benchmark

### Setup

Task HTML files cloned to: `~/cmpe299b/miniwob-plusplus/`

Required environment variable (set before any MiniWoB use):
```bash
export MINIWOB_URL="file:///home/017557527/cmpe299b/miniwob-plusplus/miniwob/html/miniwob/"
```

### Verified Working

MiniWoB tasks are self-contained HTML pages. No external services needed. Tasks load and run correctly on g17 with headless Chromium. ~100 tasks of varying difficulty.

### Why MiniWoB First (Not WebArena)

WebArena requires 7 concurrent Docker containers (shopping, Reddit, GitLab, Wikipedia, map, CMS, homepage) with database seeding. Docker is not available on HPC. Converting to Apptainer and orchestrating without Docker Compose is a multi-day effort. MiniWoB validates the entire trace collection pipeline without infrastructure complexity.

## HuggingFace Model Management

### Authentication

```bash
activate_thesis
python -c "from huggingface_hub import login; login()"
# Paste token when prompted
```

The `huggingface-cli` command is NOT available. Always use the Python API.

### Model Locations

All models stored at `~/cmpe299b/models/`:
- `~/cmpe299b/models/llama-3.2-3b/` — Llama-3.2-3B-Instruct (~6GB) ✅ Downloaded
- `~/cmpe299b/models/qwen-2.5-7b/` — Qwen-2.5-7B-Instruct (~15GB) ✅ Downloaded
- `~/cmpe299b/models/mistral-7b/` — Mistral-7B-Instruct-v0.3 (~15GB) ✅ Downloaded

### Loading Models for Inference (on GPU node)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_path = '/home/017557527/cmpe299b/models/llama-3.2-3b'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float16, device_map='auto')
```

## Directory Layout

```
/home/017557527/
├── agent_xes/              # Project codebase (git repo)
│   ├── src/
│   │   ├── data_collection/
│   │   │   ├── trace_schema.py
│   │   │   ├── trace_logger.py
│   │   │   └── agent_runner.py    # Uses relative imports (.trace_schema, etc.)
│   │   ├── config.py
│   │   └── utils/
│   │       └── experiment_logging.py
│   ├── scripts/
│   │   ├── collect_traces.py      # Entry point for batch collection (TODO)
│   │   └── collect_traces.sh      # SLURM batch script
│   └── data/
│       └── raw_traces/            # Output directory for traces
├── cmpe299b/
│   ├── models/
│   │   ├── llama-3.2-3b/
│   │   ├── qwen-2.5-7b/
│   │   └── mistral-7b/
│   ├── miniwob-plusplus/           # MiniWoB++ task HTML files
│   └── containers/                 # Apptainer SIF files (future WebArena)
├── .conda/envs/thesis/            # Conda environment (Python 3.11)
└── .cache/ms-playwright/
    └── chromium-1117/chrome-linux/ # Manually installed Chromium
```

## Codebase Import Notes

Files in `src/data_collection/` use **relative imports**:
```python
from .trace_schema import AgentTrace, TraceStep  # NOT `from trace_schema import ...`
from .trace_logger import TraceLogger
```

To import from outside the package:
```python
import sys
sys.path.insert(0, 'src')
from data_collection.trace_schema import AgentTrace
```

## Apptainer / Singularity

Not available as system binary on login node. Available as modules on both nodes:
```bash
module avail apptainer    # apptainer/1.1.7, apptainer/1.3.2
module avail singularity  # singularity/3.4.1, singularity/3.10.3
```

Available as system binary on GPU nodes (g17):
```bash
which apptainer      # /usr/bin/apptainer
apptainer --version  # apptainer version 1.4.5-2.el9
```

For pulling Docker images (needs internet), use `module load apptainer` on login node.

## Storage

- **Home directory**: `/home/017557527/` — NFS, 105TB volume shared across all users. Accessible from all nodes.
- **Data directory**: `/data/` — shared across all nodes via InfiniBand.
- **Scratch directory**: `/scratch/` — 524TB Lustre parallel filesystem. High-throughput I/O.

## Network

- **Login node** (`coe-hpc1`): HAS outbound internet ✅
- **GPU/compute nodes**: NO internet ❌
- All downloads (models, pip packages, container images, browser binaries) must happen on login node.

## Common Pitfalls

1. **Wrong partition name**: Use `gpuqs`/`gpuqm`/`gpuql`, NOT `gpu`
2. **`conda activate` fails**: Use `source activate thesis` (or `activate_thesis` alias)
3. **`which python` shows system Python after activation**: Must `export PATH=~/.conda/envs/thesis/bin:$PATH`
4. **`module load ml/torch/2.6` breaks conda**: It loads system Python 3.12, overriding conda Python 3.11. Don't use it — PyTorch is installed in conda env.
5. **`pip install` leaks to system Python**: Always use `python -m pip install`
6. **numpy/scipy compilation fails**: Install via `conda install -c conda-forge <pkg>` instead of pip
7. **`gnu14/14.2.0` doesn't exist**: Available GCC modules are `gnu/5.4.0`, `gnu7/7.3.0`, `gnu8/8.3.0`
8. **`playwright install chromium` fails**: Login node GLIBC too old, GPU node no internet. Must download zip manually on login node and extract to `~/.cache/ms-playwright/`
9. **`huggingface-cli` not found**: Use `python -c "from huggingface_hub import ..."` instead
10. **No internet on GPU node**: Downloads fail silently or timeout. Always download on login node first.

## Validated Pipeline (as of March 4, 2026)

1. ✅ SSH access to HPC working
2. ✅ Direct GPU access via `ssh -X coe-hpc3` → g17
3. ✅ SLURM partitions confirmed (`gpuqs`/`gpuqm`/`gpuql`)
4. ✅ Conda `thesis` env — Python 3.11, PyTorch with CUDA, transformers, BrowserGym
5. ✅ All three models downloaded (Llama-3.2-3B, Qwen-2.5-7B, Mistral-7B)
6. ✅ LLM inference on A40 GPU — Llama-3.2-3B confirmed generating text
7. ✅ Playwright Chromium headless launching on g17
8. ✅ MiniWoB environment loading tasks, returning observations
9. ✅ End-to-end test: model + browser + inference loop working together
10. ✅ Codebase imports resolving (relative imports fixed)
11. ⬜ `collect_traces.py` entry point (next step)
12. ⬜ Batch SLURM job submitted and completing
13. ⬜ WebArena setup (deferred — needs Docker alternative or external machine)
