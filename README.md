# RAMP

Risk-Aware Maintenance and Production Scheduling for Stochastic Flexible Job Shops.

This repository contains only the compact implementation used by the paper:
the health-aware FJSP environment, scenario/cost model, RAMP policy, PPO
trainer, PDR baselines, and the nominal FJSP parser.

## Install

The paper was run with Python 3.10, PyTorch 2.3.0/CUDA 12.1 and RTX 3090.

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

## Reproduce the paper protocol

`train_ramp.py` is the single entry point for training and evaluation. The
paper's two public regimes are `H0` (healthy machines and nominal processing)
and `H1` (stochastic processing, degradation, failures, PM and CM). The default
RAMP setting is `H1`; use `--setting H0` for the healthy control.

Prepare disjoint `train`, `validation` and `test` directories containing the
nominal `.fjs` files. The runner intentionally rejects split overlap. The
health overlay is generated from the nominal instance when the environment is
created; no health data file needs to be committed.

If a source directory contains at least 200 instances, the included helper
creates the paper's 80/20/100 partition (use the paper's frozen split when it
is available):

```bash
python3 scripts/split_fjsp.py /path/to/nominal/instances \
  --output data/SD1-10x5 --train 80 --validation 20 --test 100 --seed 100
```

For one seed, the paper configuration is:

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

Run the same command with seeds `401` and `402`. The output contains both
decoders: `greedy_*` corresponds to the paper's Greedy result and
`sampling_*` with `--stochastic-eval-samples 16` corresponds to Best-of-16.
For an already trained model, use `--evaluate-only --resume
runs/.../best.pt` and keep the same test split and scenario seeds.

Aggregate the three seed-level raw files with:

```bash
python3 summarize_results.py runs/seed400/raw_results.json \
  runs/seed401/raw_results.json runs/seed402/raw_results.json
```

The main settings are also recorded in [`configs/paper_ramp.json`](configs/paper_ramp.json).
`generate_overlay.py` and `data_utils.py` are provided for preparing nominal
FJSP files and health overlays; `pdr_baselines.py` and `pdr_adapter.py` expose
FIFO, MOR, MWKR and SPT comparisons.

For the industrial bundle, place the paper's manifest-validated Steel-FJSP
data under `data/Steel_FJSP_Real_v1` and use `--steel-suite main`; the two
small suite contracts are included in `configs/`.

## Published reference values

These are the RAMP columns reported in `manuscript_TII_final(2).pdf`; they are
included as a compact acceptance target, not as generated result artifacts.

| Dataset / regime | Decoder | Cost | Makespan | CVaR<sub>0.95</sub> |
| --- | --- | ---: | ---: | ---: |
| SD1-10x5 / H0 | Greedy | 1.187 ± 0.004 | 112.048 ± 0.345 | 1.330 ± 0.003 |
| SD1-10x5 / H1 | Greedy | 1.198 ± 0.008 | 113.125 ± 0.823 | 1.347 ± 0.007 |
| SD1-20x5 / H1 | Greedy | 1.143 ± 0.002 | 211.064 ± 0.530 | 1.312 ± 0.012 |
| SD1-20x5 / H1 | Best-of-16 | 1.196 ± 0.010 | 220.763 ± 1.624 | 1.406 ± 0.036 |
| ISD1 / H1 | Greedy | 1.554 ± 0.008 | 343.268 ± 0.823 | 1.829 ± 0.007 |
| ISD2 / H1 | Greedy | 2.313 ± 0.002 | 552.064 ± 0.530 | 2.773 ± 0.011 |

Exact means require the paper's disjoint 80/20/100 splits, seeds, trained
checkpoints and hardware protocol. The repository supplies the executable
code and configuration contract but does not redistribute those data or
weights.

## Layout

```text
train_ramp.py       # one train/evaluate entry point
ramp/               # environment, health dynamics, reward, PPO, checkpoints
model/              # RAMP policy core, scenario encoder and baselines
data_utils.py       # strict nominal FJSP parser and small generators
pdr_*.py            # paper priority-dispatching baselines
scripts/             # deterministic data split helper
configs/            # paper protocol, without datasets
```
