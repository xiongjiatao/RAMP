"""Arguments used only by ``data_utils.py`` when generating FJSP instances."""

from __future__ import annotations

import argparse


parser = argparse.ArgumentParser(description="RAMP nominal FJSP generator")
parser.add_argument("--n-j", "--n_j", dest="n_j", type=int, default=10)
parser.add_argument("--n-m", "--n_m", dest="n_m", type=int, default=5)
parser.add_argument("--op-per-job", "--op_per_job", dest="op_per_job", type=float, default=0)
parser.add_argument("--low", type=int, default=1)
parser.add_argument("--high", type=int, default=99)
parser.add_argument("--data-suffix", "--data_suffix", dest="data_suffix", default="mix")
parser.add_argument("--op-per-mch-min", "--op_per_mch_min", dest="op_per_mch_min", type=int, default=1)
parser.add_argument("--op-per-mch-max", "--op_per_mch_max", dest="op_per_mch_max", type=int, default=5)
parser.add_argument("--data-source", "--data_source", dest="data_source", default="SD3")
parser.add_argument("--data-size", "--data_size", dest="data_size", type=int, default=100)
parser.add_argument("--data-type", "--data_type", dest="data_type", choices=("test", "vali"), default="test")
parser.add_argument("--cover-data", "--cover_data_flag", dest="cover_data_flag", action="store_true")
parser.add_argument("--seed-datagen", "--seed_datagen", dest="seed_datagen", type=int, default=200)
parser.add_argument(
    "--seed-train-vali-datagen",
    "--seed_train_vali_datagen",
    dest="seed_train_vali_datagen",
    type=int,
    default=100,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse generator arguments without consuming the trainer's CLI."""

    return parser.parse_args(argv)
