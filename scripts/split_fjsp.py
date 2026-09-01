"""Create deterministic, disjoint train/validation/test folders of .fjs files."""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path


def split_files(
    sources: list[Path],
    output: Path,
    *,
    train: int,
    validation: int,
    test: int,
    seed: int,
) -> None:
    files = sorted({path.resolve() for source in sources for path in source.rglob("*.fjs")})
    required = train + validation + test
    if len(files) < required:
        raise ValueError(f"found {len(files)} .fjs files, need {required}")
    random.Random(seed).shuffle(files)
    boundaries = (train, train + validation, required)
    for name, start, stop in (
        ("train", 0, boundaries[0]),
        ("validation", boundaries[0], boundaries[1]),
        ("test", boundaries[1], boundaries[2]),
    ):
        destination = output / name
        destination.mkdir(parents=True, exist_ok=True)
        for index, source in enumerate(files[start:stop], start=1):
            shutil.copy2(source, destination / f"{index:03d}_{source.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--train", type=int, default=80)
    parser.add_argument("--validation", type=int, default=20)
    parser.add_argument("--test", type=int, default=100)
    parser.add_argument("--seed", type=int, default=100)
    args = parser.parse_args()
    split_files(
        args.sources,
        args.output,
        train=args.train,
        validation=args.validation,
        test=args.test,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
