from __future__ import annotations

import glob
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


INPUT_DIR = os.path.join("results", "new")
OUTPUT_DIR = os.path.join(INPUT_DIR, "plots")


def _model_display(model: str, lam: float | None) -> str:
    m = model.lower()
    if m == "baseline":
        return "Baseline"
    if m == "replay":
        return "Replay"
    if m == "ewc":
        return f"EWC ({int(lam)})" if lam is not None else "EWC"
    if m == "hybrid":
        return f"Hybrid ({int(lam)})" if lam is not None else "Hybrid"
    if m == "joint":
        return "Joint (upper bound)"
    return model


def _sort_key(name: str) -> int:
    order = [
        "Baseline",
        "Replay",
        "EWC (10)",
        "EWC (100)",
        "Hybrid (10)",
        "Hybrid (100)",
        "Joint (upper bound)",
    ]
    return order.index(name) if name in order else 999


def _load_final_avg_rows(csv_paths: List[str]) -> pd.DataFrame:
    rows: List[Dict[str, float | str | None]] = []
    for path in csv_paths:
        df = pd.read_csv(path)
        if "eval_task" not in df.columns or "model" not in df.columns:
            continue

        df = df.copy()
        df["eval_task"] = df["eval_task"].astype(str).str.lower()
        avg_rows = df[df["eval_task"] == "avg"].copy()
        if avg_rows.empty:
            continue

        avg_rows["trained_up_to_task"] = pd.to_numeric(
            avg_rows["trained_up_to_task"], errors="coerce"
        )
        final_task = int(avg_rows["trained_up_to_task"].max())
        final_row = avg_rows[avg_rows["trained_up_to_task"] == final_task].iloc[-1]

        lam = None
        if "lambda_ewc" in df.columns and pd.notna(final_row.get("lambda_ewc", np.nan)):
            lam = float(final_row["lambda_ewc"])

        model = str(final_row["model"]).lower()
        rows.append(
            {
                "source_file": os.path.basename(path),
                "model": model,
                "display": _model_display(model, lam),
                "lambda": lam,
                "recall@10": float(pd.to_numeric(final_row.get("recall@10"), errors="coerce")),
                "ndcg@10": float(pd.to_numeric(final_row.get("ndcg@10"), errors="coerce")),
                "forgetting_recall": float(
                    pd.to_numeric(final_row.get("forgetting_recall"), errors="coerce")
                ),
                "forgetting_ndcg": float(
                    pd.to_numeric(final_row.get("forgetting_ndcg"), errors="coerce")
                ),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        raise SystemExit("No valid final avg rows found in results/new CSV files.")

    out = out.sort_values(by="display", key=lambda s: s.map(_sort_key)).reset_index(drop=True)
    return out


def _plot_comparison_slide(summary: pd.DataFrame, out_path: str) -> None:
    labels = summary["display"].tolist()
    ndcg = summary["ndcg@10"].to_numpy()
    recall = summary["recall@10"].to_numpy()

    x = np.arange(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(13, 7))
    ndcg_colors = ["#4c78a8" for _ in labels]
    recall_colors = ["#54a24b" for _ in labels]
    if "EWC (100)" in labels:
        ndcg_colors[labels.index("EWC (100)")] = "#1f4e8c"
        recall_colors[labels.index("EWC (100)")] = "#1f7a3a"

    b1 = ax.bar(x - width / 2, ndcg, width, label="NDCG@10", color=ndcg_colors)
    b2 = ax.bar(x + width / 2, recall, width, label="Recall@10", color=recall_colors)

    ax.set_title(
        "Model Comparison: Baseline vs Replay vs EWC(10,100) vs Hybrid(10,100)\n"
        "EWC (100) highlighted as best overall final metric"
    )
    ax.set_ylabel("Final Average Metric")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    metric_min = float(min(np.min(ndcg), np.min(recall)))
    metric_max = float(max(np.max(ndcg), np.max(recall)))
    span = max(metric_max - metric_min, 1e-4)
    pad = max(0.0015, span * 0.40)
    ax.set_ylim(metric_min - pad, metric_max + pad)
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.2)

    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(
                f"{h:.4f}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    if "EWC (100)" in labels:
        idx = labels.index("EWC (100)")
        top_y = max(ndcg[idx], recall[idx])
        ax.annotate(
            "Best overall",
            xy=(x[idx], top_y),
            xytext=(0, 20),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            arrowprops={"arrowstyle": "->", "lw": 1.2},
        )

    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close(fig)


def _plot_forgetting_slide(summary: pd.DataFrame, out_path: str) -> None:
    labels = summary["display"].tolist()
    forget = summary["forgetting_ndcg"].to_numpy()

    fig, ax = plt.subplots(figsize=(13, 7))
    colors = ["#4c78a8" if v <= 0 else "#f58518" for v in forget]
    if "Hybrid (100)" in labels:
        colors[labels.index("Hybrid (100)")] = "#1f4e8c"
    bars = ax.bar(labels, forget, color=colors)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_title(
        "Forgetting-Focused Slide (Final Avg Forgetting NDCG)\n"
        "Lower is better (negative = improvement over past)"
    )
    ax.set_ylabel("Forgetting NDCG (lower is better)")
    lim = max(0.01, float(np.nanmax(np.abs(forget))) * 1.35)
    ax.set_ylim(-lim, lim)
    ax.grid(axis="y", alpha=0.2)

    for bar in bars:
        h = bar.get_height()
        ax.annotate(
            f"{h:.4f}",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 4 if h >= 0 else -10),
            textcoords="offset points",
            ha="center",
            va="bottom" if h >= 0 else "top",
            fontsize=8,
        )

    if "Hybrid (100)" in labels:
        idx = labels.index("Hybrid (100)")
        best_val = forget[idx]
        ax.text(
            0.98,
            0.96,
            f"Best forgetting: Hybrid (100) = {best_val:.4f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            fontweight="bold",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.9, "edgecolor": "#1f4e8c"},
        )

    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close(fig)


def _plot_final_metric_slide(summary: pd.DataFrame, out_path: str) -> None:
    ranked = summary.sort_values("ndcg@10", ascending=False).reset_index(drop=True)
    labels = ranked["display"].tolist()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))

    axes[0].barh(labels, ranked["ndcg@10"], color="#4c78a8")
    axes[0].invert_yaxis()
    axes[0].set_title("Final Average NDCG@10")
    axes[0].set_xlabel("NDCG@10")
    axes[0].grid(axis="x", alpha=0.2)

    axes[1].barh(labels, ranked["recall@10"], color="#54a24b")
    axes[1].invert_yaxis()
    axes[1].set_title("Final Average Recall@10")
    axes[1].set_xlabel("Recall@10")
    axes[1].grid(axis="x", alpha=0.2)

    fig.suptitle("Final-Metric Slide (Ranked by NDCG@10)", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    csv_paths = sorted(glob.glob(os.path.join(INPUT_DIR, "VALIDATED_*.csv")))
    if not csv_paths:
        raise SystemExit("No VALIDATED_*.csv files found in results/new.")

    summary = _load_final_avg_rows(csv_paths)
    summary_out = os.path.join(OUTPUT_DIR, "presentation_summary.csv")
    summary.to_csv(summary_out, index=False)

    comparison_out = os.path.join(OUTPUT_DIR, "slide_model_comparison.png")
    forgetting_out = os.path.join(OUTPUT_DIR, "slide_forgetting_focus.png")
    final_metric_out = os.path.join(OUTPUT_DIR, "slide_final_metrics.png")

    _plot_comparison_slide(summary, comparison_out)
    _plot_forgetting_slide(summary, forgetting_out)
    _plot_final_metric_slide(summary, final_metric_out)

    print("SUMMARY", summary_out)
    print("COMPARISON_SLIDE", comparison_out)
    print("FORGETTING_SLIDE", forgetting_out)
    print("FINAL_METRIC_SLIDE", final_metric_out)


if __name__ == "__main__":
    main()
