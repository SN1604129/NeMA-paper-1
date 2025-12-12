# src/plot_results.py
import os
import glob
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", type=str, default="results",
                    help="Directory containing per-run CSV files.")
    ap.add_argument("--plots_dir", type=str, default="plots",
                    help="Directory to save generated plots.")
    ap.add_argument("--run_prefix", type=str, default="",
                    help="Only include runs whose run_id starts with this prefix (e.g., 'final_').")
    return ap.parse_args()


def load_all_results(results_dir: str) -> pd.DataFrame:
    files = glob.glob(os.path.join(results_dir, "*.csv"))
    if not files:
        raise RuntimeError(
            f"No CSV files found in '{results_dir}'.\n"
            f"Fix options:\n"
            f"  1) Run from repo root, so 'results/' exists there.\n"
            f"  2) Pass the correct folder: python -m src.plot_results --results_dir <path>\n"
            f"  3) Find where CSVs are: find . -name \"*.csv\" | head\n"
        )

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            df["_source_file"] = os.path.basename(f)
            dfs.append(df)
        except Exception as e:
            print(f"[WARN] Failed reading {f}: {e}")

    if not dfs:
        raise RuntimeError(f"Found CSVs in '{results_dir}' but none could be read.")

    all_df = pd.concat(dfs, ignore_index=True)
    return all_df


def ensure_types(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure columns exist
    required = [
        "run_id", "epoch", "mem_lambda", "write_threshold",
        "avg_write_ratio", "val_acc_mem", "val_acc_nomem"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(
            "Missing required columns in results CSV(s): "
            + ", ".join(missing)
            + "\nCheck that train_delayed_qa.py is writing the expected header."
        )

    # Types
    df = df.copy()
    df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")
    df["mem_lambda"] = pd.to_numeric(df["mem_lambda"], errors="coerce")
    df["write_threshold"] = pd.to_numeric(df["write_threshold"], errors="coerce")
    df["avg_write_ratio"] = pd.to_numeric(df["avg_write_ratio"], errors="coerce")
    df["val_acc_mem"] = pd.to_numeric(df["val_acc_mem"], errors="coerce")
    df["val_acc_nomem"] = pd.to_numeric(df["val_acc_nomem"], errors="coerce")

    df = df.dropna(subset=["run_id", "epoch", "mem_lambda", "write_threshold"])
    return df


def filter_by_prefix(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    if not prefix:
        return df
    df2 = df[df["run_id"].astype(str).str.startswith(prefix)].copy()
    if df2.empty:
        raise RuntimeError(
            f"No rows matched run_prefix='{prefix}'.\n"
            f"Tip: print unique run_ids via:\n"
            f"  python -c \"import pandas as pd,glob; "
            f"import os; "
            f"dfs=[pd.read_csv(f) for f in glob.glob('results/*.csv')]; "
            f"df=pd.concat(dfs); print(df['run_id'].unique()[:20])\""
        )
    return df2


def final_epoch_per_run(df: pd.DataFrame) -> pd.DataFrame:
    # final epoch row per run
    return df.sort_values("epoch").groupby("run_id", as_index=False).tail(1)


# ------------------ PLOTS ------------------

def plot_acc_vs_write_ratio(df_final: pd.DataFrame, outdir: str):
    """
    Scatter: final accuracy (with memory) vs final write_ratio.
    Color by mem_lambda.
    """
    plt.figure()
    sc = plt.scatter(
        df_final["avg_write_ratio"],
        df_final["val_acc_mem"],
        c=df_final["mem_lambda"],
        cmap="viridis",
        alpha=0.85,
    )
    plt.colorbar(sc, label="mem_lambda")
    plt.xlabel("Average write ratio (final epoch)")
    plt.ylabel("Validation accuracy with memory (final epoch)")
    plt.title("Accuracy vs Write Ratio")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "acc_vs_write_ratio.png"), dpi=300)
    plt.close()


def plot_memory_on_vs_off_over_epochs(df: pd.DataFrame, outdir: str):
    """
    Line plot: mean acc over epochs, memory vs no-memory (averaged over all runs).
    """
    grouped = df.groupby("epoch").agg(
        val_acc_mem_mean=("val_acc_mem", "mean"),
        val_acc_nomem_mean=("val_acc_nomem", "mean"),
    ).reset_index()

    plt.figure()
    plt.plot(grouped["epoch"], grouped["val_acc_mem_mean"], label="With memory")
    plt.plot(grouped["epoch"], grouped["val_acc_nomem_mean"], label="Without memory")
    plt.xlabel("Epoch")
    plt.ylabel("Validation accuracy")
    plt.title("Memory On vs Off (mean across runs)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "memory_on_vs_off.png"), dpi=300)
    plt.close()


def plot_write_ratio_vs_mem_lambda(df_final: pd.DataFrame, outdir: str):
    """
    Errorbar: final write_ratio vs mem_lambda (mean ± std over runs).
    """
    grouped = df_final.groupby("mem_lambda").agg(
        write_ratio_mean=("avg_write_ratio", "mean"),
        write_ratio_std=("avg_write_ratio", "std"),
        n=("avg_write_ratio", "count"),
    ).reset_index().sort_values("mem_lambda")

    plt.figure()
    plt.errorbar(
        grouped["mem_lambda"],
        grouped["write_ratio_mean"],
        yerr=grouped["write_ratio_std"],
        fmt="-o",
    )
    plt.xlabel("mem_lambda")
    plt.ylabel("Average write ratio (final epoch)")
    plt.title("Write Ratio vs mem_lambda (mean ± std)")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "write_ratio_vs_mem_lambda.png"), dpi=300)
    plt.close()


def plot_acc_vs_mem_lambda(df_final: pd.DataFrame, outdir: str):
    """
    Errorbar: final accuracy (with memory) vs mem_lambda.
    """
    grouped = df_final.groupby("mem_lambda").agg(
        acc_mean=("val_acc_mem", "mean"),
        acc_std=("val_acc_mem", "std"),
        n=("val_acc_mem", "count"),
    ).reset_index().sort_values("mem_lambda")

    plt.figure()
    plt.errorbar(
        grouped["mem_lambda"],
        grouped["acc_mean"],
        yerr=grouped["acc_std"],
        fmt="-o",
    )
    plt.xlabel("mem_lambda")
    plt.ylabel("Validation accuracy with memory (final epoch)")
    plt.title("Accuracy vs mem_lambda (mean ± std)")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "acc_vs_mem_lambda.png"), dpi=300)
    plt.close()


def plot_tradeoff_by_threshold(df_final: pd.DataFrame, outdir: str):
    """
    Scatter: accuracy vs write_ratio, separate color by write_threshold.
    This helps show how threshold affects the tradeoff.
    """
    plt.figure()
    sc = plt.scatter(
        df_final["avg_write_ratio"],
        df_final["val_acc_mem"],
        c=df_final["write_threshold"],
        cmap="plasma",
        alpha=0.85,
    )
    plt.colorbar(sc, label="write_threshold")
    plt.xlabel("Average write ratio (final epoch)")
    plt.ylabel("Validation accuracy with memory (final epoch)")
    plt.title("Tradeoff by write_threshold")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "tradeoff_by_threshold.png"), dpi=300)
    plt.close()


# ------------------ MAIN ------------------

def main():
    args = parse_args()
    os.makedirs(args.plots_dir, exist_ok=True)

    df = load_all_results(args.results_dir)
    df = ensure_types(df)
    df = filter_by_prefix(df, args.run_prefix)

    df_final = final_epoch_per_run(df)

    # Basic required plots
    plot_acc_vs_write_ratio(df_final, args.plots_dir)
    plot_memory_on_vs_off_over_epochs(df, args.plots_dir)
    plot_write_ratio_vs_mem_lambda(df_final, args.plots_dir)

    # Extra helpful plots for paper
    plot_acc_vs_mem_lambda(df_final, args.plots_dir)
    plot_tradeoff_by_threshold(df_final, args.plots_dir)

    print(f"Loaded {len(df)} rows from results_dir='{args.results_dir}'")
    print(f"Unique runs: {df['run_id'].nunique()}")
    print(f"Plots saved to: {args.plots_dir}/")


if __name__ == "__main__":
    main()
