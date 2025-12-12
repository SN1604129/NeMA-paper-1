NeMA-Lite

Learning Selective Memory Writing in Memory-Augmented Transformers

📄 Paper 1 – Selective Memory Writing
🧠 Research Codebase

📌 Overview

NeMA-Lite is a lightweight memory-augmented Transformer that learns when to write information into external memory.
Unlike prior memory-augmented models that store token representations indiscriminately, NeMA-Lite introduces a learned write gate and an explicit memory usage regulariser, enabling selective, sparse memory storage under a controllable budget.

This repository contains the full implementation, experiments, sweeps, and analysis code used for Paper 1, which focuses exclusively on the problem of selective memory writing.

🎯 Research Motivation

Memory-augmented Transformers are widely used to handle long-range dependencies. However:

Most existing approaches store all token states or rely on heuristics

Memory usage grows uncontrollably

The question of when to write to memory is underexplored

Key insight:

In many long-range tasks, only a small subset of tokens are actually relevant for future decisions.

NeMA-Lite addresses this by learning task-aware, selective memory storage.

✨ Key Contributions (Paper 1)

Learned Write Gate
A neural gating mechanism decides whether each token should be written to memory.

Explicit Memory Budget Control
Memory usage is regularised via a differentiable penalty on write probabilities.

Selective Storage Emergence
The model achieves high accuracy while writing only a small fraction of tokens.

Memory–Performance Tradeoff Analysis
Systematic sweeps reveal how memory usage and accuracy trade off under different budgets.

⚠️ Scope note:
This paper focuses only on memory writing. Forgetting, updating, and hierarchical memory are intentionally left for future work.

🏗️ Architecture Summary

NeMA-Lite consists of:

A standard Transformer encoder

An external episodic memory

A learned write gate

A simple read mechanism using the CLS token

Write Gate


For each token hidden state \( h_t \):

\[
g_t = \sigma(W_2 \, \text{ReLU}(W_1 h_t))
\]

- \( g_t \in [0,1] \) represents the probability of writing to memory
- A threshold converts probabilities into hard write decisions during training
A threshold converts probabilities into hard write decisions during training

Training Objective
\[
\mathcal{L} = \mathcal{L}_{task} + \lambda \cdot \mathbb{E}[g_t]
\]

- \( \mathcal{L}_{task} \): classification loss
- \( \lambda \): memory penalty controlling write sparsity​

: classification loss

𝜆
λ: memory penalty controlling write sparsity

🧪 Experimental Setup
Task: Synthetic Delayed Question Answering

Input: sequence of digits

Target: digit appearing at a random early position

Requires remembering a specific earlier token

Metrics

Validation accuracy

Average memory write ratio

Accuracy with vs without memory

Swept Hyperparameters

mem_lambda ∈ {0.0, 0.05, 0.1, 0.2}

write_threshold ∈ {0.3, 0.5, 0.7}

Multiple random seeds

📁 Repository Structure

nema-paper1/
├── src/
│   ├── models/
│   │   ├── memory_store.py        # External memory implementation
│   │   ├── write_gate.py          # Neural write gate
│   │   └── transformer_wrapper.py # NeMA-Lite model
│   ├── tasks/
│   │   └── synthetic_delayed_qa.py
│   ├── train_delayed_qa.py        # Training + logging
│   ├── sweep_delayed_qa.py        # Hyperparameter sweeps
│   ├── plot_results.py            # Plot generation
│   └── summarise_result.py        # Final-epoch aggregation
├── notebooks/
│   └── 01_NeMA_Lite_Results_Summary.ipynb
├── results/                       # Per-run CSV logs
├── plots/                         # Generated figures
├── tables/                        # Summary tables
└── README.md

🚀 How to Run
1️⃣ Install dependencies
pip install -r requirements.txt

2️⃣ Train a single run
python src/train_delayed_qa.py \
  --mem_lambda 0.05 \
  --write_threshold 0.7 \
  --num_epochs 20

3️⃣ Run hyperparameter sweep
python src/sweep_delayed_qa.py

4️⃣ Generate plots
python -m src.plot_results --results_dir results --plots_dir plots

5️⃣ Summarise results (Table 1)
python src/summarise_result.py --results_dir results

📊 Results Highlights

Selective storage emerges naturally under memory regularisation

High accuracy is achieved with very low write ratios

Writing all tokens is unnecessary and often suboptimal

Memory usage can be smoothly controlled via mem_lambda

See:

plots/acc_vs_write_ratio.png

plots/write_ratio_vs_mem_lambda.png

tables/summary_final_epoch.csv

📓 Notebook

The notebook
notebooks/01_NeMA_Lite_Results_Summary.ipynb
provides a clean, reproducible summary of:

Sweep results

Final-epoch aggregation

Best configurations

Plot inspection

This notebook is intended for analysis and presentation, not training.

⚠️ Limitations (Explicit)

Synthetic task only

No memory forgetting or updating

Small-scale experiments

These limitations are intentional and define the scope of Paper 1.
