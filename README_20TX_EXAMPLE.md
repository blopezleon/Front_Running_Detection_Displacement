# 20-Transaction Example: Model vs FlashBoys (Illustrative)

This is a short, self-contained example that shows — step by step — how the predictive ML model and the FlashBoys heuristic label the same stream of transactions, and how to compute the "lead" (how many transactions earlier the model signaled MEV before the first true MEV occurred).

IMPORTANT: this file contains an illustrative example to explain the logic and metrics. Run the provided 20-tx notebook cell on your dataset to get real values.

Columns in the table below:
- IDX: local index inside the 20-transaction window (0..19)
- BLOCK: block number where the tx was seen (may repeat for many txs)
- GAS (Gwei): tx gas price in Gwei
- TRUE: ground-truth label (from FlashBoys auction analysis) — MEV or NORMAL
- MODEL_PROB: ML model predicted MEV probability (0..100%)
- MODEL_PRED: binary model prediction at 0.50 threshold (MEV/NORM)
- FLASHBOYS: simplified FlashBoys heuristic label at time of inspection (MEV/NORM)

------------------------------------------------------------------------------

| IDX | BLOCK   | GAS (Gwei) | TRUE   | MODEL_PROB | MODEL_PRED | FLASHBOYS |
|-----:|--------:|-----------:|:-------|----------:|:----------:|:---------:|
|   0 | 1001000 |       1.2  | NORMAL |      0.2%  | NORM       | NORM      |
|   1 | 1001000 |       0.3  | NORMAL |      0.1%  | NORM       | NORM      |
|   2 | 1001000 |       0.4  | NORMAL |      0.3%  | NORM       | NORM      |
|   3 | 1001000 |       1.1  | NORMAL |      0.4%  | NORM       | NORM      |
|   4 | 1001000 |       0.2  | NORMAL |      0.1%  | NORM       | NORM      |
|   5 | 1001001 |       1.5  | NORMAL |      0.5%  | NORM       | NORM      |
|   6 | 1001001 |       1.8  | NORMAL |      0.6%  | NORM       | NORM      |
|   7 | 1001001 |       2.2  | NORMAL |      1.2%  | NORM       | NORM      |
|   8 | 1001001 |       3.5  | NORMAL |      4.2%  | NORM       | NORM      |
|   9 | 1001002 |       5.1  | NORMAL |     12.3%  | NORM       | NORM      |
|  10 | 1001002 |       8.0  | NORMAL |     65.4%  | MEV        | NORM      |
|  11 | 1001002 |       9.5  | NORMAL |     72.1%  | MEV        | NORM      |
|  12 | 1001002 |      10.2  | MEV    |     88.9%  | MEV        | MEV       |
|  13 | 1001002 |       9.8  | MEV    |     83.0%  | MEV        | MEV       |
|  14 | 1001003 |       9.0  | MEV    |     79.5%  | MEV        | MEV       |
|  15 | 1001003 |       3.1  | NORMAL |      2.0%  | NORM       | NORM      |
|  16 | 1001003 |       1.2  | NORMAL |      0.5%  | NORM       | NORM      |
|  17 | 1001004 |       0.3  | NORMAL |      0.2%  | NORM       | NORM      |
|  18 | 1001004 |       0.2  | NORMAL |      0.1%  | NORM       | NORM      |
|  19 | 1001004 |       0.4  | NORMAL |      0.1%  | NORM       | NORM      |

------------------------------------------------------------------------------

Explanation of this illustrative run

- Ground truth: transactions 12, 13, 14 are labeled MEV by the FlashBoys auction analysis (these are the "true" MEV front-running auction transactions).
- FlashBoys heuristic (the one implemented in the notebook as a comparison) also labels 12..14 as MEV, because it sees high gas prices combined with competition in recent gas prices.
- The ML model assigned high probabilities starting at indices 10 and 11 (65% and 72%) and produced MEV predictions at those indices before any transaction was labeled MEV by the ground truth (first true MEV is at index 12).

How to compute "lead" (transactions earlier):

1. Find first_true = index of the first transaction with TRUE == MEV.
   - In this example, first_true = 12.

2. Find first_model = index of the first transaction where MODEL_PRED == MEV and first_model < first_true.
   - Here, first_model = 10.

3. Lead (in transactions) = first_true - first_model = 12 - 10 = 2 transactions earlier.

4. Lead (in blocks): if you want block-level lead, compute block_first_true - block_first_model.
   - block_first_true = BLOCK at index 12 = 1001002
   - block_first_model = BLOCK at index 10 = 1001002
   - Block lead = 1001002 - 1001002 = 0 blocks (in this example the model predicted within the same block, 2 txs before the first true MEV)

Why the model can detect earlier

- The ML model uses temporal features (recent_mev_rate, recent_avg_gas, recent_max_gas, gas_vs_recent_avg, etc.). These features capture the start of an escalation pattern: a few transactions raising gas aggressively.
- FlashBoys (hardcoded) needs a clearer signal (e.g., explicit competition or a gas range + above-average gas) that often only becomes visible at the first MEV-labeled txs or once a full mini-auction is visible.
- The ML model can pick up subtler precursors: e.g., a sequence of slowly increasing gas prices and a rising recent_mev_rate can push the model probability above threshold before the auction's labeled winner(s) appear.

Interpretation and practical notes

- Lead in transactions is what matters for a predictive system: catching MEV even 1-3 txs earlier allows downstream actions (alerting, filtering, inserting protective txs).
- Lead in blocks is coarser; in many cases the entire mini-auction happens within the same block, so transaction-level lead is more informative.
- Lowering the model threshold (e.g., 0.3 instead of 0.5) will increase early detection (more lead) but also increase false positives. Tune threshold for your risk tolerance.

What to check when you run this on real data

1. Run the 20-tx example cell included in the notebook using a known MEV auction (the code picks the largest MEV auction by default). Compare the printed table to this illustrative one.
2. Note first_true and first_model, compute lead (txs and blocks). Repeat across multiple auctions to compute median lead and distribution (how many txs earlier, how often model led vs lagged vs missed).
3. If the model often lags FlashBoys (i.e., first_model >= first_true), examine feature distributions around those auctions — perhaps the model needs more examples of low-gas MEV events or feature scaling adjusted.

Quick formula / pseudocode to compute lead programmatically

```
# sample_df: 20-row DataFrame window in temporal order
# pred_labels: binary array of model predictions for the window
# true_labels: array of ground-truth labels in the window

first_true = next(i for i, v in enumerate(true_labels) if v == 1)
first_model = next((i for i, v in enumerate(pred_labels) if v == 1 and i < first_true), None)
if first_model is None:
    lead_tx = 0  # model did not detect earlier
else:
    lead_tx = first_true - first_model
```

Closing notes

This file is intentionally short and concrete. If you want, I can:
- Run the 20-tx example on actual data and save real output to `examples/20tx_actual_output.md` (requires running the notebook kernel), or
- Add a command-line script that computes lead statistics across all MEV auctions and prints a summary table (median lead, mean lead, % of auctions where model led, etc.).

Tell me which of the two you'd like and I'll implement it next.
