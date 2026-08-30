"""
results/new/build_cl_table.py

Pure pandas over the VALIDATED_*.csv files - no training. Computes standard
Average Accuracy (ACC) and Backward Transfer (BWT), Lopez-Paz & Ranzato 2017:

    ACC = (1/T) * sum_k R(T-1, k)
    BWT = (1/(T-1)) * sum_{k=0..T-2} [ R(T-1, k) - R(k, k) ]

R(t, k) = the metric at trained_up_to_task=t, eval_task=k. T=5, final index 4.

The stored "avg" row at trained_up_to_task=4 already equals ACC (this is
asserted below, not assumed) - the repo's existing forgetting_ndcg column is
a *different* metric (Chaudhry et al., best-prior-checkpoint, opposite sign)
and is kept alongside BWT rather than conflated with it.

Joint has no diagonal R(k,k) (it never trained on a single task in
isolation), so its BWT is reported as "-", not computed as 0.
"""

import glob
import math
import os
import sys

import pandas as pd

# Windows console defaults to cp1252, which can't print the "λ"/"—"
# characters below - force utf-8 on stdout, best-effort.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

RESULTS_DIR = os.path.dirname(os.path.abspath(__file__))
FINAL_TASK = 4  # T-1, with T=5

# display name, sort position, "stored samples" label
ROW_ORDER = [
    ("baseline", None, "Baseline (sequential)", 0),
    ("replay", None, "Replay", 1),
    ("ewc", 10.0, "EWC (λ=10)", 2),
    ("ewc", 100.0, "EWC (λ=100)", 3),
    ("hybrid", 10.0, "Hybrid (λ=10)", 4),
    ("hybrid", 100.0, "Hybrid (λ=100)", 5),
    ("joint", None, "Joint (upper bound)", 6),
]


def load_all():
    frames = []
    for path in sorted(glob.glob(os.path.join(RESULTS_DIR, "VALIDATED_*.csv"))):
        df = pd.read_csv(path)
        df["_file"] = os.path.basename(path)
        frames.append(df)
    if not frames:
        raise SystemExit(f"No VALIDATED_*.csv files found in {RESULTS_DIR}")
    return pd.concat(frames, ignore_index=True)


def r(df, model, lam, t, k):
    """R(t, k) for a given model/lambda: metric row at trained_up_to_task=t, eval_task=k."""
    sel = (df["model"] == model) & (df["trained_up_to_task"] == t) & (df["eval_task"].astype(str) == str(k))
    if lam is not None:
        sel &= df.get("lambda_ewc", pd.Series(index=df.index, dtype=float)) == lam
    rows = df[sel]
    if len(rows) != 1:
        raise ValueError(f"expected 1 row for model={model} lambda={lam} t={t} k={k}, got {len(rows)}")
    return rows.iloc[0]


def stored_samples(model, lam, sub):
    if model == "joint":
        return "all (~4.96M)"
    if "buffer_size" in sub.columns and sub["buffer_size"].notna().any():
        return f"{int(sub['buffer_size'].dropna().iloc[0]):,}"
    return "0"


def build_row(df, model, lam, label):
    sel = (df["model"] == model)
    if lam is not None:
        sel &= df.get("lambda_ewc", pd.Series(index=df.index, dtype=float)) == lam
    sub = df[sel]
    if sub.empty:
        raise ValueError(f"no rows found for model={model} lambda={lam} ({label})")

    avg_row = sub[(sub["trained_up_to_task"] == FINAL_TASK) & (sub["eval_task"].astype(str) == "avg")].iloc[0]

    # ACC self-check: recompute from the per-task rows and assert it matches
    # the stored avg row, rather than trusting the stored value blindly.
    per_task_recall = [r(df, model, lam, FINAL_TASK, k)["recall@10"] for k in range(FINAL_TASK + 1)]
    per_task_ndcg   = [r(df, model, lam, FINAL_TASK, k)["ndcg@10"] for k in range(FINAL_TASK + 1)]
    acc_recall = sum(per_task_recall) / len(per_task_recall)
    acc_ndcg   = sum(per_task_ndcg) / len(per_task_ndcg)
    assert math.isclose(acc_recall, avg_row["recall@10"], abs_tol=1e-9), \
        f"{label}: recomputed ACC recall {acc_recall} != stored avg row {avg_row['recall@10']}"
    assert math.isclose(acc_ndcg, avg_row["ndcg@10"], abs_tol=1e-9), \
        f"{label}: recomputed ACC ndcg {acc_ndcg} != stored avg row {avg_row['ndcg@10']}"

    if model == "joint":
        bwt_recall = bwt_ndcg = None  # no diagonal R(k,k) exists for a pooled model
    else:
        diffs_recall, diffs_ndcg = [], []
        for k in range(FINAL_TASK):  # k = 0..3, i.e. T-1 terms
            final = r(df, model, lam, FINAL_TASK, k)
            diag  = r(df, model, lam, k, k)
            diffs_recall.append(final["recall@10"] - diag["recall@10"])
            diffs_ndcg.append(final["ndcg@10"] - diag["ndcg@10"])
        bwt_recall = sum(diffs_recall) / len(diffs_recall)
        bwt_ndcg   = sum(diffs_ndcg) / len(diffs_ndcg)

    return {
        "Method": label,
        "Stored samples": stored_samples(model, lam, sub),
        "ACC Recall@10": acc_recall,
        "ACC NDCG@10": acc_ndcg,
        "BWT Recall@10": bwt_recall,
        "BWT NDCG@10": bwt_ndcg,
        "Forgetting NDCG (Chaudhry)": avg_row.get("forgetting_ndcg", float("nan")),
    }


def main():
    df = load_all()

    rows = [build_row(df, model, lam, label) for model, lam, label, _ in ROW_ORDER]
    table = pd.DataFrame(rows)

    joint_ndcg = table.loc[table["Method"] == "Joint (upper bound)", "ACC NDCG@10"].iloc[0]
    table["% of joint NDCG"] = (table["ACC NDCG@10"] / joint_ndcg * 100).round(1)

    out_path = os.path.join(RESULTS_DIR, "cl_summary_table.csv")
    table.to_csv(out_path, index=False)

    # Markdown for pasting into README/slide, "-" for joint's undefined BWT
    md = table.copy()
    for col in ("BWT Recall@10", "BWT NDCG@10"):
        md[col] = md[col].map(lambda v: "—" if v is None or pd.isna(v) else f"{v:+.4f}")
    for col in ("ACC Recall@10", "ACC NDCG@10", "Forgetting NDCG (Chaudhry)"):
        md[col] = md[col].map(lambda v: f"{v:.4f}")
    md["% of joint NDCG"] = md["% of joint NDCG"].map(lambda v: f"{v:.1f}%")

    print(md.to_markdown(index=False))
    print(f"\nT=5 tasks, seed 42, single run, 5 epochs/task (~96.9k optimizer steps for every row including joint).")
    print(f"BWT: negative = forgetting (Lopez-Paz & Ranzato 2017). "
          f"Forgetting NDCG: positive = forgetting (Chaudhry et al. 2018), from the trained_up_to_task=4 avg row.")
    print(f"\nwritten to {out_path}")


if __name__ == "__main__":
    main()
