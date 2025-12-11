# sweep_delayed_qa.py
import itertools
import subprocess
import os
import uuid

def main():
    # grids to sweep
    mem_lambdas = [0.0, 0.05, 0.1, 0.2]
    write_thresholds = [0.3, 0.5, 0.7]

    num_epochs = 20
    batch_size = 64
    lr = 1e-3
    log_dir = "results"

    os.makedirs(log_dir, exist_ok=True)

    # (optional) multiple seeds per config for robustness
    seeds = [0, 1, 2]

    for mem_lambda, write_threshold in itertools.product(mem_lambdas, write_thresholds):
        for seed in seeds:
            run_id = f"ml{mem_lambda}_wt{write_threshold}_s{seed}_{uuid.uuid4().hex[:8]}"
            print(f"\n=== Running {run_id} ===")

            cmd = [
                "python",
                "src/train_delayed_qa.py",   # <---- key change
                f"--mem_lambda={mem_lambda}",
                f"--write_threshold={write_threshold}",
                f"--num_epochs={num_epochs}",
                f"--batch_size={batch_size}",
                f"--lr={lr}",
                f"--log_dir={log_dir}",
                f"--run_id={run_id}",
            ]

            print("Command:", " ".join(cmd))
            subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
