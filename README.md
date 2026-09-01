# RAMP

Risk-Aware Maintenance and Production Scheduling for Stochastic Flexible Job Shops.

This repository provides a compact implementation of the health-aware FJSP
environment, scenario/cost model, RAMP policy, PPO trainer, PDR baselines, and
nominal FJSP parser.

## Install

The recommended environment is Python 3.10 with PyTorch 2.3.0/CUDA 12.1.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For a CUDA installation, install the matching PyTorch wheel from the official
PyTorch index before installing the remaining requirements.

## Quick check

The smoke run uses an embedded 2-job/2-machine instance and needs no data:

```bash
python3 train_ramp.py --smoke --updates 1 --validation-limit 1 \
  --stochastic-eval-samples 1 --log runs/smoke.json
```

It trains one PPO update, evaluates Greedy and one sampled policy, and writes
the same `Cost`, `CVaR`, `Makespan`, `Phi` and validity fields used by the full
runner. `pytest -q` runs this check automatically.

## Prepare data

`train_ramp.py` is the single entry point for training and evaluation. The
two available regimes are `H0` (healthy machines and nominal processing) and
`H1` (stochastic processing, degradation, failures, PM and CM). The default
RAMP setting is `H1`; use `--setting H0` for the healthy control.

Prepare disjoint `train`, `validation` and `test` directories containing the
nominal `.fjs` files. The runner intentionally rejects split overlap. The
health overlay is generated from the nominal instance when the environment is
created; no health data file needs to be committed.

The included helper creates disjoint train/validation/test partitions:

```bash
python3 scripts/split_fjsp.py /path/to/nominal/instances \
  --output data/SD1-10x5 --train 80 --validation 20 --test 100 --seed 100
```

Example training and evaluation command:

```bash
CUDA_VISIBLE_DEVICES=0 python3 train_ramp.py \
  --method ramp --setting H1 \
  --physical-gpu 0 --device cuda:0 \
  --train-dir /path/to/SD1-10x5/train \
  --validation-dir /path/to/SD1-10x5/validation \
  --test-dir /path/to/SD1-10x5/test \
  --updates 500 --batch-size 8 \
  --num-scenarios 32 --reward-scenarios 128 \
  --seed 400 --validation-interval 10 --validation-limit 20 \
  --stochastic-eval-samples 16 --disable-early-stopping \
  --best-checkpoint runs/sd1_10x5_seed400/best.pt \
  --last-checkpoint runs/sd1_10x5_seed400/last.pt \
  --log runs/sd1_10x5_seed400/result.json \
  --raw-results runs/sd1_10x5_seed400/raw_results.json
```

Change `--seed` for independent runs. The output contains deterministic
`greedy_*` metrics and stochastic `sampling_*` metrics. For an already trained
model, use `--evaluate-only --resume runs/.../best.pt` and keep the same test
split and scenario settings.

The main settings are recorded in
[`configs/paper_ramp.json`](configs/paper_ramp.json).
`generate_overlay.py` and `data_utils.py` prepare nominal FJSP files and health
overlays; `pdr_baselines.py` and `pdr_adapter.py` expose FIFO, MOR, MWKR and
SPT comparisons.

For the industrial bundle, place the manifest-validated Steel-FJSP data under
`data/Steel_FJSP_Real_v1` and use `--steel-suite main`; the suite contracts are
included in `configs/`.

## Output and result aggregation

Training writes checkpoints and JSON logs under `runs/`. The JSON output
contains cost, CVaR, makespan, objective value, validity counts and timing
fields. Aggregate raw result files with:

```bash
python3 summarize_results.py runs/seed400/raw_results.json \
  runs/seed401/raw_results.json runs/seed402/raw_results.json
```

## Layout

```text
train_ramp.py       # one train/evaluate entry point
ramp/               # environment, health dynamics, reward, PPO, checkpoints
model/              # RAMP policy core, scenario encoder and baselines
data_utils.py       # strict nominal FJSP parser and small generators
pdr_*.py            # priority-dispatching baselines
scripts/             # deterministic data split helper
configs/            # runtime configuration templates
```
