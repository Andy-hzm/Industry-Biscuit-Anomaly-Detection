"""Build train/valid/test folders from the Kaggle Industry Biscuit dataset.

The dataset has 1,225 biscuits, each photographed at 4 rotations (4,900 images).
Annotations.csv lists the original captures in rows 1-1225 and the three rotated
copies of biscuit k in rows 1226 + 3*(k-1) ... 1228 + 3*(k-1). Every split here is
made per *biscuit*, so all four rotations of a biscuit land in the same split and
no biscuit leaks from train into valid/test.

Two layouts are supported:

  multiclass  4 classes, stratified 80/10/10 by biscuit.
              -> OUT/train|valid|test/{Defect_No,Defect_Shape,Defect_Object,Defect_Color}
              Used by phase2_fall2025/02_supervised.

  ok_nok      One-class setup: train on OK only; valid/test hold OK + NOK.
              Defaults mirror the original notebooks (1000 OK train images,
              100 OK + 100 NOK in valid and test, NOK mix 40% shape /
              30% object / 30% color), rounded to whole biscuits.
              -> OUT/train/ok, OUT/valid|test/{ok,nok}
              Used by phase2_fall2025/03_unsupervised.

Usage:
  python prepare_data.py --raw data/IndustryBiscuit --out data/DS_multiclass --mode multiclass
  python prepare_data.py --raw data/IndustryBiscuit --out data/DS --mode ok_nok

--raw is the unzipped Kaggle folder containing Images/ and Annotations.csv.
"""

import argparse
import random
import shutil
from collections import defaultdict
from pathlib import Path

import pandas as pd

N_BISCUITS = 1225
CLASSES = ["Defect_No", "Defect_Shape", "Defect_Object", "Defect_Color"]


def load_biscuits(raw: Path) -> dict[str, list[list[str]]]:
    """Return {class: [[4 file names of biscuit 1], [biscuit 2], ...]}."""
    ann = pd.read_csv(raw / "Annotations.csv")
    if len(ann) != 4 * N_BISCUITS:
        raise ValueError(f"Expected {4 * N_BISCUITS} annotation rows, found {len(ann)}")

    by_class = defaultdict(list)
    for k in range(1, N_BISCUITS + 1):
        rows = [k] + [N_BISCUITS + 3 * (k - 1) + j for j in (1, 2, 3)]  # 1-based
        group = ann.iloc[[r - 1 for r in rows]]
        labels = set(group["classDescription"])
        if len(labels) != 1:
            raise ValueError(f"Biscuit {k}: rotations disagree on label {labels}")
        by_class[labels.pop()].append(group["file"].tolist())
    return by_class


def copy_group(files, raw: Path, dest: Path) -> int:
    dest.mkdir(parents=True, exist_ok=True)
    for f in files:
        shutil.copy2(raw / "Images" / f, dest / f)
    return len(files)


def split_multiclass(by_class, rng, ratios=(0.8, 0.1, 0.1)):
    plan = defaultdict(list)  # (split, class) -> [biscuit groups]
    for cls in CLASSES:
        groups = by_class[cls][:]
        rng.shuffle(groups)
        n_train = round(len(groups) * ratios[0])
        n_valid = round(len(groups) * ratios[1])
        plan[("train", cls)] = groups[:n_train]
        plan[("valid", cls)] = groups[n_train:n_train + n_valid]
        plan[("test", cls)] = groups[n_train + n_valid:]
    return plan


def split_ok_nok(by_class, rng, train_ok=1000, eval_ok=100, eval_nok=100,
                 nok_mix=(("Defect_Shape", 0.4), ("Defect_Object", 0.3), ("Defect_Color", 0.3))):
    to_biscuits = lambda images: images // 4
    plan = defaultdict(list)

    ok = by_class["Defect_No"][:]
    rng.shuffle(ok)
    n_tr, n_ev = to_biscuits(train_ok), to_biscuits(eval_ok)
    if n_tr + 2 * n_ev > len(ok):
        raise ValueError("Not enough OK biscuits for the requested sizes")
    plan[("train", "ok")] = ok[:n_tr]
    plan[("valid", "ok")] = ok[n_tr:n_tr + n_ev]
    plan[("test", "ok")] = ok[n_tr + n_ev:n_tr + 2 * n_ev]

    nok_biscuits = to_biscuits(eval_nok)
    counts = [round(nok_biscuits * share) for _, share in nok_mix]
    counts[0] += nok_biscuits - sum(counts)  # make the mix sum exactly
    for (cls, _), n in zip(nok_mix, counts):
        groups = by_class[cls][:]
        rng.shuffle(groups)
        plan[("valid", "nok")] += groups[:n]
        plan[("test", "nok")] += groups[n:2 * n]
    return plan


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw", type=Path, required=True, help="Folder with Images/ and Annotations.csv")
    p.add_argument("--out", type=Path, required=True, help="Output folder (must not exist)")
    p.add_argument("--mode", choices=["multiclass", "ok_nok"], required=True)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if args.out.exists():
        raise SystemExit(f"{args.out} already exists; remove it or choose another --out")

    rng = random.Random(args.seed)
    by_class = load_biscuits(args.raw)
    plan = split_multiclass(by_class, rng) if args.mode == "multiclass" else split_ok_nok(by_class, rng)

    print(f"{'split':<6} {'class':<14} {'biscuits':>8} {'images':>7}")
    for (split, cls), groups in sorted(plan.items()):
        n = sum(copy_group(g, args.raw, args.out / split / cls) for g in groups)
        print(f"{split:<6} {cls:<14} {len(groups):>8} {n:>7}")
    print(f"Done -> {args.out}")


if __name__ == "__main__":
    main()
