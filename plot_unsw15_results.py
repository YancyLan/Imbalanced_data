
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def plot_distribution_shift(csv_path: Path, out_png: Path):
    df = pd.read_csv(csv_path)
    if "synth/real_ratio" not in df.columns:
        raise ValueError(f"'synth/real_ratio' not in columns of {csv_path}")
    df = df.sort_values("synth/real_ratio", ascending=True).reset_index(drop=True)

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.bar(df["class"].astype(str), df["synth/real_ratio"].astype(float))
    ax.axhline(1.0, linestyle="--")  # ideal line
    ax.set_xlabel("Class")
    ax.set_ylabel("synth/real_ratio  (1 = matched prior)")
    ax.set_title("Distribution Shift: Synthetic vs Real (Classifier-Predicted)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)

def plot_relative_coverage(csv_path: Path, out_png: Path):
    df = pd.read_csv(csv_path)
    col = "relative_coverage"
    if col not in df.columns:
        # fallback: older file naming
        if "zspace_nn_median" in df.columns:
            raise ValueError(f"Expected 'relative_coverage' in {csv_path}, found legacy coverage file.")
        else:
            raise ValueError(f"'{col}' not in columns of {csv_path}")
    df = df.sort_values(col, ascending=True).reset_index(drop=True)

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.bar(df["class"].astype(str), df[col].astype(float))
    ax.axhline(1.0, linestyle="--")  # roughly indicates parity (values <1 worse coverage)
    ax.set_xlabel("Class")
    ax.set_ylabel("Relative coverage = median(real→real) / median(real→synth)")
    ax.set_title("Per-Class Relative Coverage (smaller = worse)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="indir", type=str, required=True, help="Directory containing result CSVs")
    ap.add_argument("--dist_csv", type=str, default="distribution_shift_summary.csv")
    ap.add_argument("--cov_csv", type=str, default="relative_coverage.csv")
    ap.add_argument("--out_prefix", type=str, default="unsw15_uncond")
    args = ap.parse_args()

    indir = Path(args.indir)
    out_prefix = args.out_prefix

    dist_csv = indir / args.dist_csv
    cov_csv = indir / args.cov_csv

    dist_png = indir / f"{out_prefix}_distribution_shift.png"
    cov_png  = indir / f"{out_prefix}_relative_coverage.png"

    plot_distribution_shift(dist_csv, dist_png)
    plot_relative_coverage(cov_csv, cov_png)

    print(f"Saved: {dist_png}")
    print(f"Saved: {cov_png}")

if __name__ == "__main__":
    main()
