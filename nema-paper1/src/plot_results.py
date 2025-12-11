# src/plot_results.py
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_DIR = "results"
PLOTS_DIR = "plots"

os.makedirs(PLOTS_DIR, exist_ok=True)

def load_all_results(results_dir=RESULTS_DIR):
    files = glob.glob(os.path.join(results_dir, "*.csv"))
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        dfs.append(df)
    if not dfs:
        raise RuntimeError(f"No CSV files found in {results_dir}")
    all_df = pd.concat(dfs, ignore_index=True)
    return all_df

def extract_config_cols(df):
    # in case float columns are read as object
    df["mem_lambda"] = df["mem_lambda"].astype(float)
    df["write_threshold"] = df["write_threshold"].astype(float)
    return df

def plot_acc_vs_write_ratio(all_df):
    """
    Scatter plot: final-epoch accuracy with memory vs write_ratio,
    colored by mem_lambda or grouped by write_threshold.
    """
    df = all_df.copy()
    df = extract_config_cols(df)

    # take final epoch per run_id
    df_final = df.sort_values("epoch").groupby("run_id").tail(1)

    plt.figure()
    scatter = plt.scatter(
        df_final["avg_write_ratio"],
        df_final["val_acc_mem"],
        c=df_final["mem_lambda"],
        cmap="viridis",
        alpha=0.8,
    )
    plt.colorbar(scatter, label="mem_lambda")
    plt.xlabel("Average write ratio")
    plt.ylabel("Validation accuracy (with memory)")
    plt.title("Accuracy vs Write Ratio (final epoch)")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "acc_vs_write_ratio.png"), dpi=300)

def plot_memory_on_vs_off(all_df):
    """
    Line plot of val_acc_mem vs epoch and val_acc_nomem vs epoch,
    averaged over runs for a given configuration.
    For simplicity, here we aggregate over all runs.
    You can also condition on e.g. mem_lambda or write_threshold.
    """
    df = all_df.copy()
    df = extract_config_cols(df)

    # Example: average over all runs, all configs
    grouped = df.groupby("epoch").agg(
        val_acc_mem_mean=("val_acc_mem", "mean"),
        val_acc_nomem_mean=("val_acc_nomem", "mean"),
    )

    plt.figure()
    plt.plot(grouped.index, grouped["val_acc_mem_mean"], label="With memory")
    plt.plot(grouped.index, grouped["val_acc_nomem_mean"], label="Without memory")
    plt.xlabel("Epoch")
    plt.ylabel("Validation accuracy")
    plt.title("Memory On vs Off (averaged over runs)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "memory_on_vs_off.png"), dpi=300)

def plot_write_ratio_vs_mem_lambda(all_df):
    """
    Bar/line plot: final write_ratio as a function of mem_lambda
    (aggregated over runs and thresholds).
    """
    df = all_df.copy()
    df = extract_config_cols(df)

    df_final = df.sort_values("epoch").groupby("run_id").tail(1)

    grouped = df_final.groupby("mem_lambda").agg(
        write_ratio_mean=("avg_write_ratio", "mean"),
        write_ratio_std=("avg_write_ratio", "std"),
    ).reset_index()

    plt.figure()
    plt.errorbar(
        grouped["mem_lambda"],
        grouped["write_ratio_mean"],
        yerr=grouped["write_ratio_std"],
        fmt="-o",
    )
    plt.xlabel("mem_lambda")
    plt.ylabel("Average write ratio (final epoch)")
    plt.title("Write Ratio vs mem_lambda")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "write_ratio_vs_mem_lambda.png"), dpi=300)

def main():
    all_df = load_all_results()
    plot_acc_vs_write_ratio(all_df)
    plot_memory_on_vs_off(all_df)
    plot_write_ratio_vs_mem_lambda(all_df)
    print(f"Plots saved to {PLOTS_DIR}/")

if __name__ == "__main__":
    main()
