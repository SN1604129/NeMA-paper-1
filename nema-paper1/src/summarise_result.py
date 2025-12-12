import os
import glob
import argparse
import pandas as pd

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", type=str, default="results",
                    help="Directory containing per-run CSV files.")
    ap.add_argument("--out_dir", type=str, default="tables",
                    help="Where to save summary tables.")
    return ap.parse_args()

def main():
    args = parse_args()

    pattern = os.path.join(args.results_dir, "*.csv")
    files = glob.glob(pattern)

    if not files:
        raise RuntimeError(
            f"No CSV files found using pattern: {pattern}\n"
            f"Run this to locate them:\n"
            f"  find . -name \"*.csv\" | head -n 50\n"
        )

    dfs = [pd.read_csv(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)

    # final epoch per run
    final_df = df.sort_values("epoch").groupby("run_id").tail(1)

    summary = final_df.groupby(["mem_lambda", "write_threshold"]).agg(
        runs=("run_id", "count"),
        write_ratio_mean=("avg_write_ratio", "mean"),
        write_ratio_std=("avg_write_ratio", "std"),
        acc_mem_mean=("val_acc_mem", "mean"),
        acc_mem_std=("val_acc_mem", "std"),
        acc_nomem_mean=("val_acc_nomem", "mean"),
        acc_nomem_std=("val_acc_nomem", "std"),
    ).reset_index().sort_values(["mem_lambda", "write_threshold"])

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "summary_final_epoch.csv")
    summary.to_csv(out_path, index=False)

    print(summary)
    print(f"\nSaved to {out_path}")

if __name__ == "__main__":
    main()
