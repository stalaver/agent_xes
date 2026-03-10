# agent_xes

**Early Failure Detection in Web Navigation Agents via Closed Sequential Pattern Mining**

Master's thesis implementation that detects web agent failures early by mining closed sequential patterns from execution traces. The system symbolizes agent actions into discrete event sequences, mines recurring failure signatures using the BIDE+ algorithm, and ranks them by cross-site coverage to build a pattern library that can predict failures within the first K steps of execution.

Author: Sergio Talavera

---

## Architecture

The system has two phases: an **offline training pipeline** that builds a failure pattern library from historical traces, and an **online detection pipeline** that matches live traces against the library.

```
Offline Pipeline
================
Raw Traces ─> Symbolize ─> K-Prefix Extract ─> BIDE Mine ─> Rank ─> Pattern Library

Online Pipeline
=========================
Live Trace ─> Symbolize ─> Pattern Match ─> Decision (Terminate / Alert / Continue)
```

### Module-to-Pipeline Mapping

| Pipeline Stage          | Source Module                          |
|-------------------------|----------------------------------------|
| Trace collection        | `src/data_collection/agent_runner.py`  |
| Trace storage           | `src/data_collection/trace_logger.py`  |
| Symbolization           | `src/preprocessing/symbolizer.py`      |
| K-prefix extraction     | `src/preprocessing/k_prefix.py`        |
| BIDE+ pattern mining    | `src/mining/spmf_wrapper.py`           |
| Coverage-based ranking  | `src/mining/pattern_ranker.py`         |
| Pattern library storage | `src/mining/signature_library.py`      |
| Baseline comparison     | `src/baselines/`                       |
| Evaluation & metrics    | `src/evaluation/`                      |

---

## Directory Structure

```
agent_xes/
├── src/                            # Core library code
│   ├── data_collection/
│   │   ├── trace_schema.py         # Data models: AgentTrace, TraceStep, enums
│   │   ├── trace_logger.py         # JSON/JSONL trace capture and storage
│   │   ├── agent_runner.py         # BrowserGym + HuggingFace LLM agent runner
│   │   └── failure_injector.py     # (placeholder)
│   ├── preprocessing/
│   │   ├── symbolizer.py           # Multi-level action symbolization (fine/medium/coarse)
│   │   ├── k_prefix.py             # K-prefix extraction and PrefixDataset
│   │   └── xes_exporter.py         # (placeholder — IEEE 1849-2023 XES export)
│   ├── mining/
│   │   ├── spmf_wrapper.py         # SPMF BIDE+ Java subprocess wrapper
│   │   ├── pattern_ranker.py       # Precision x coverage ranking and filtering
│   │   ├── signature_library.py    # Pattern library persistence and matching
│   │   └── bide_miner.py           # (placeholder)
│   ├── detection/
│   │   ├── failure_predictor.py    # (placeholder — online failure prediction)
│   │   ├── pattern_matcher.py      # (placeholder — online pattern matching)
│   │   └── token_calculator.py     # (placeholder — token cost estimation)
│   ├── baselines/
│   │   ├── __init__.py             # BASELINES registry for all 7 methods
│   │   ├── base.py                 # BaseBaseline interface (fit/predict/predict_at_k)
│   │   ├── frequency_vector.py     # Bag-of-symbols + Logistic Regression
│   │   ├── ngram.py                # N-gram features + Random Forest
│   │   ├── taspm.py                # BIDE+ on failure-only traces
│   │   ├── process_conformance.py  # First-order transition model deviation scoring
│   │   ├── deeplog.py              # LSTM next-symbol anomaly detection
│   │   ├── bilstm.py              # Bi-LSTM classifier with SMOTE
│   │   └── bide_coverage.py        # BIDE+ with coverage-based ranking (main approach)
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── data_split.py           # Stratified train/val/test splits, site holdout
│   │   ├── metrics.py              # F1@K, AUC-PR, optimal threshold search
│   │   └── experiment.py           # ExperimentRunner for full baseline comparison
│   └── utils/
│       ├── config.py               # Central configuration (sites, models, mining params)
│       └── logging_setup.py        # (placeholder)
│
├── scripts/                        # Entry points and CLI tools
│   ├── run_experiment.py           # Full 7-baseline comparison experiment
│   ├── run_offline_pipeline.py     # Offline training pipeline (synthetic or real traces)
│   ├── collect_traces.py           # MiniWoB trace collection via BrowserGym + LLM
│   ├── collect_traces.sh           # SLURM batch wrapper for HPC trace collection
│   ├── reprocess_traces.py         # Re-parse action fields in existing trace JSON
│   ├── validate_pipeline.py        # Import and component sanity check (no SPMF needed)
│   └── test_phase2_pipeline.py     # Integration test for preprocessing + mining
│
├── data/
│   ├── raw_traces/                 # Collected agent execution traces
│   │   ├── miniwob/
│   │   │   ├── llama-3.2-3b/      # UUID-named .json trace files
│   │   │   ├── mistral-7b/
│   │   │   └── qwen-2.5-7b/
│   │   └── webarena/
│   │       └── unknown/
│   ├── pipeline_test/              # SPMF I/O from pipeline test runs
│   └── experiment_results/         # Timestamped experiment output directories
│       └── {timestamp}/
│           ├── config.json         # Experiment configuration snapshot
│           ├── results.json        # Full per-baseline, per-K metrics
│           └── summary_table.txt   # Human-readable + LaTeX summary tables
│
├── lib/
│   └── spmf.jar                    # SPMF Java library (BIDE+ algorithm)
│
├── guides/
│   ├── SJSU_HPC_Reference.md       # SJSU CoE HPC setup: SLURM, conda, Playwright, models
│   └── offline_pipeline_guide.md   # Offline pipeline walkthrough with examples
│
├── webarena/                       # Vendored WebArena fork (browser agent framework)
│   ├── agent/
│   ├── browser_env/
│   ├── evaluation_harness/
│   └── ...
│
├── tests/                          # (empty — tests currently in scripts/)
├── experiments/                    # (empty — experiments run via scripts/)
├── logs/                           # Runtime log output
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

---

## Prerequisites

### Required

| Dependency | Version   | Purpose                                    |
|------------|-----------|--------------------------------------------|
| Python     | >= 3.10   | Runtime                                    |
| Java       | >= 8      | Runs SPMF for BIDE+ sequential mining      |
| SPMF       | latest    | Closed sequential pattern mining library    |

### For Trace Collection Only

| Dependency   | Purpose                                           |
|--------------|---------------------------------------------------|
| NVIDIA GPU   | LLM inference (3B model needs ~7 GB, 7B needs ~15 GB VRAM) |
| Chromium     | Headless browser for BrowserGym environments       |
| Playwright   | Browser automation                                 |
| MiniWoB++    | Self-contained HTML task benchmarks                |

### Python Packages

The full list is in `requirements.txt`. Key categories:

| Category           | Packages                                               |
|--------------------|--------------------------------------------------------|
| Web agent          | `browsergym`                                           |
| LLM inference      | `transformers`, `accelerate`, `bitsandbytes`, `vllm`   |
| Pattern mining     | `mlxtend`, SPMF (Java, external)                       |
| Process mining     | `pm4py`                                                |
| Machine learning   | `scikit-learn`, `torch`, `imbalanced-learn`            |
| Data analysis      | `pandas`, `numpy`, `scipy`, `statsmodels`              |
| Visualization      | `matplotlib`, `seaborn`                                |
| Experiment tracking| `wandb`, `mlflow` (optional)                           |
| Development        | `pytest`, `black`, `jupyter`                           |

---

## Installation and Setup

### 1. Clone and install Python dependencies

```bash
git clone <repository-url> agent_xes
cd agent_xes

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

> **Note:** `torch` with CUDA support may require a platform-specific install command.
> See https://pytorch.org/get-started/locally/ for your platform.

### 2. Download SPMF

SPMF is a Java library required by the BIDE+ mining baselines (`bide_coverage`, `taspm`) and the offline pipeline.

```bash
mkdir -p lib
curl -L -o lib/spmf.jar "https://www.philippe-fournier-viger.com/spmf/SPMF.jar"

# Verify Java is available
java -version
```

### 3. Validate the installation

Run the validation script to check that all Python imports resolve and core components work (does not require SPMF):

```bash
python scripts/validate_pipeline.py
```

### 4. (Optional) MiniWoB++ for trace collection

If you plan to collect traces rather than use existing ones:

```bash
git clone https://github.com/Farama-Foundation/miniwob-plusplus.git
export MINIWOB_URL="file://$(pwd)/miniwob-plusplus/miniwob/html/miniwob/"

# Install Playwright browsers
playwright install chromium
```

### 5. (Optional) SJSU HPC environment

For running on the SJSU College of Engineering HPC cluster, see [`guides/SJSU_HPC_Reference.md`](guides/SJSU_HPC_Reference.md) which covers SLURM partitions, conda setup, model paths, and Playwright installation workarounds.

---

## Usage

All scripts are run from the project root. They add `src/` to the Python path automatically.

### Run a baseline comparison experiment

```bash
# Full 7-baseline experiment with default settings (K=10, abstraction_level=1)
python scripts/run_experiment.py --trace-dir data/raw_traces

# Specific baselines and K values
python scripts/run_experiment.py \
    --trace-dir data/raw_traces \
    --baselines frequency_vector,ngram,bide_coverage \
    --k-values 3,5,8,10

# Custom SPMF jar location
python scripts/run_experiment.py \
    --trace-dir data/raw_traces \
    --spmf-jar lib/spmf.jar
```

Results are saved to `data/experiment_results/{timestamp}/` containing:
- `config.json` — experiment parameters
- `results.json` — full per-baseline, per-K metrics (precision, recall, F1, AUC-PR)
- `summary_table.txt` — human-readable table and LaTeX-formatted tables

### Run the offline training pipeline

```bash
# Test with synthetic traces (no real data needed)
python scripts/run_offline_pipeline.py --mode synthetic --n-traces 100

# Run on real collected traces
python scripts/run_offline_pipeline.py \
    --mode real \
    --trace-dir data/raw_traces \
    --k 10 \
    --min-support 0.03 \
    --output-dir data/trained
```

### Collect new traces

```bash
python scripts/collect_traces.py \
    --model-name llama-3.2-3b \
    --model-path /path/to/models/llama-3.2-3b \
    --output-dir data/raw_traces \
    --benchmark miniwob \
    --max-steps 30 \
    --tasks click-test click-button enter-text
```

On SJSU HPC, submit via SLURM:

```bash
sbatch scripts/collect_traces.sh
```

### Validate pipeline components

```bash
python scripts/validate_pipeline.py
```

Tests imports, trace creation, symbolization, and k-prefix extraction without requiring SPMF or GPU resources.

---

## Baselines

Seven failure detection baselines are implemented, all conforming to the `BaseBaseline` interface (`fit` / `predict` / `predict_at_k`):

| Baseline               | Approach                                                        |
|------------------------|-----------------------------------------------------------------|
| `frequency_vector`     | Bag-of-symbol counts fed to Logistic Regression                 |
| `ngram`                | N-gram features (n=2,3,4) fed to Random Forest                  |
| `taspm`                | BIDE+ mining on failure-only traces; match fraction as score    |
| `process_conformance`  | First-order transition model from success traces; deviation scoring |
| `deeplog`              | LSTM next-symbol prediction on success traces (anomaly detection) |
| `bilstm`               | Bi-LSTM binary classifier with SMOTE for class imbalance        |
| `bide_coverage`        | BIDE+ with coverage-based ranking (primary proposed method)      |

A majority-class baseline (predict all-failure) is included as a reference in experiment output.

All baselines that require SPMF (`bide_coverage`, `taspm`) will fail gracefully with a clear error if `lib/spmf.jar` is missing.

---

## Data Formats

### Agent Trace (JSON)

Each trace is a single JSON file named by UUID, stored at `data/raw_traces/{benchmark}/{model}/{uuid}.json`:

```json
{
  "metadata": {
    "trace_id": "12bcdf63-171d-4ba4-b55e-4eac901ff8d7",
    "task_id": "miniwob.navigate-tree",
    "task_description": "miniwob.navigate-tree",
    "website": "miniwob",
    "model": "llama-3.2-3b",
    "outcome": "success",
    "failure_type": null,
    "start_time": "2026-03-08T23:19:31.419379",
    "end_time": "2026-03-08T23:20:07.095607",
    "duration_seconds": 35.68,
    "benchmark": "miniwob"
  },
  "steps": [
    {
      "step_number": 1,
      "reasoning": { "raw_reasoning": "...", "intent": "...", "keywords": [] },
      "action": { "type": "click", "selector": "33", "selector_type": "bid", "raw_action": "click(\"33\")" },
      "observation": { "element_found": true, "element_state": "visible", "page_changed": false },
      "dom_hash": "bb31829d",
      "timestamp": "2026-03-08T23:19:49.743466",
      "prompt_tokens": 289,
      "completion_tokens": 1024
    }
  ],
  "total_steps": 2,
  "total_tokens": 2626
}
```

Key enums defined in `trace_schema.py`:
- **TaskOutcome:** SUCCESS, FAILURE, TIMEOUT, ERROR, UNKNOWN
- **FailureType:** NAVIGATION, VALIDATION, RECOVERY, CONTEXT, NATURAL, UNKNOWN
- **ActionType:** CLICK, TYPE, SCROLL, NAVIGATE, SELECT, HOVER, WAIT, ...

### Symbolization Levels

Traces are symbolized at three abstraction levels before mining:

| Level | Name   | Format                                      | Example                  |
|-------|--------|---------------------------------------------|--------------------------|
| 0     | Fine   | `{ACTION}_{SELECTOR}_{ELEMENT_STATE}_{HTTP}` | `CLICK_ID_VISIBLE_OK`   |
| 1     | Medium | `{ACTION}_{SELECTOR}_{SUCCESS/FAIL}`         | `CLICK_ID_SUCCESS`       |
| 2     | Coarse | `{CATEGORY}_{SUCCESS/FAIL}`                  | `INTERACTION_SUCCESS`    |

Level 1 (medium) is the default for experiments.

### SPMF Format

The SPMF wrapper converts symbol sequences to integer IDs:

```
1 -1 3 -1 5 -1 2 -1 -2
```

Each integer is a symbol ID, `-1` separates items within a sequence, and `-2` marks the end of a sequence.

### Experiment Results

Experiment output in `data/experiment_results/{timestamp}/`:

- **`config.json`** — parameters used (trace_dir, k, abstraction_level, baselines, seed)
- **`results.json`** — per-baseline, per-K detailed metrics (threshold, precision, recall, F1, AUC-PR)
- **`summary_table.txt`** — F1@K comparison table in both plain text and LaTeX format

---

## Caveats and Known Limitations

### Placeholder Modules

Several modules are stubs awaiting implementation:

- `src/detection/failure_predictor.py` — online failure prediction
- `src/detection/pattern_matcher.py` — online pattern matching
- `src/detection/token_calculator.py` — token cost estimation
- `src/preprocessing/xes_exporter.py` — IEEE 1849-2023 XES export via pm4py
- `src/data_collection/failure_injector.py` — controlled failure injection

### Testing

Pipeline integration tests live in `scripts/test_phase2_pipeline.py` and `scripts/validate_pipeline.py`.

### WebArena

WebArena requires 7 concurrent Docker containers (shopping, Reddit, GitLab, Wikipedia, map, CMS, homepage) with database seeding. **MiniWoB++ is the primary benchmark** for trace collection. The `webarena/` directory contains a vendored fork (not a git submodule) of the WebArena codebase for reference and future use.

### SPMF Jar Path

The default SPMF jar path differs between locations:
- `src/utils/config.py` defaults to `/opt/spmf/spmf.jar`
- Scripts and the offline pipeline default to `lib/spmf.jar`

Pass `--spmf-jar lib/spmf.jar` explicitly if you encounter path issues.

### GPU Requirements

Trace collection requires a GPU for LLM inference. The three thesis models and their approximate VRAM requirements:

| Model                        | VRAM (fp16) |
|------------------------------|-------------|
| Llama-3.2-3B-Instruct       | ~7 GB       |
| Qwen-2.5-7B-Instruct        | ~15 GB      |
| Mistral-7B-Instruct-v0.3    | ~15 GB      |

Pattern mining and baseline evaluation are CPU-only (except `bilstm` and `deeplog` which benefit from GPU but work on CPU).

---

## License

This project is part of a Master's thesis at San Jose State University, Department of Computer Engineering.
