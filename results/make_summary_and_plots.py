import glob
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def padded_limits(values: pd.Series, min_pad: float = 0.0002, rel_pad: float = 0.15) -> tuple[float, float]:
    vmin = float(values.min())
    vmax = float(values.max())
    span = vmax - vmin
    pad = max(min_pad, span * rel_pad)
    return vmin - pad, vmax + pad


def main() -> None:
    results_dir = "results"
    files = sorted(glob.glob(os.path.join(results_dir, "*.csv")))
    if not files:
        raise SystemExit("No result CSV files found in results/")

    rows = []
    for path in files:
        name = os.path.basename(path)
        df = pd.read_csv(path)

        if "eval_task" not in df.columns:
            continue

        avg_rows = df[df["eval_task"].astype(str) == "avg"].copy()
        if avg_rows.empty:
            continue

        avg_rows["ndcg@10"] = pd.to_numeric(avg_rows["ndcg@10"], errors="coerce")
        avg_rows["forgetting_ndcg"] = pd.to_numeric(avg_rows.get("forgetting_ndcg"), errors="coerce")
        avg_rows["recall@10"] = pd.to_numeric(avg_rows["recall@10"], errors="coerce")

        model = str(avg_rows["model"].iloc[0]).lower()
        model_disp = {
            "baseline": "Baseline",
            "replay": "Replay",
            "ewc": "EWC",
            "hybrid": "Hybrid",
        }.get(model, model.capitalize())

        lam = "-"
        if "lambda_ewc" in avg_rows.columns and avg_rows["lambda_ewc"].notna().any():
            lam = str(float(avg_rows["lambda_ewc"].dropna().iloc[0])).rstrip("0").rstrip(".")

        # Validate numeric conversions have valid data
        f_ndcg_vals = avg_rows["forgetting_ndcg"]
        avg_f_ndcg = f_ndcg_vals.mean(skipna=True) if f_ndcg_vals.notna().any() else 0.0
        
        rows.append(
            {
                "file": name,
                "model": model_disp,
                "lambda": lam,
                "avg_recall": avg_rows["recall@10"].mean(),
                "avg_ndcg": avg_rows["ndcg@10"].mean(),
                "avg_forgetting_ndcg": avg_f_ndcg,
            }
        )

    summary = pd.DataFrame(rows)
    if summary.empty:
        raise SystemExit("No avg rows found in result CSVs")

    order = {"Baseline": 0, "Replay": 1, "EWC": 2, "Hybrid": 3}
    summary["sort"] = summary["model"].map(order).fillna(99)
    summary["lambda_num"] = pd.to_numeric(summary["lambda"], errors="coerce")
    summary = summary.sort_values(["sort", "lambda_num", "file"]).drop(columns=["sort"])

    summary_out = os.path.join(results_dir, "summary_table.csv")
    summary.to_csv(summary_out, index=False)

    # Dedicated forgetting comparison table for report/slide use.
    forgetting_table = summary[["model", "lambda", "avg_forgetting_ndcg", "avg_ndcg"]].copy()
    forgetting_table = forgetting_table.sort_values("avg_forgetting_ndcg")
    forgetting_out = os.path.join(results_dir, "forgetting_comparison.csv")
    forgetting_table.to_csv(forgetting_out, index=False)

    sns.set_theme(style="whitegrid")

    lam_df = summary[summary["model"].isin(["EWC", "Hybrid"])].copy()
    lam_df = lam_df[pd.to_numeric(lam_df["lambda"], errors="coerce").notna()].copy()
    lam_df["lambda_val"] = lam_df["lambda"].astype(float)

    # Keep NDCG y-scaling consistent across NDCG plots.
    ndcg_ymin, ndcg_ymax = padded_limits(summary["avg_ndcg"], min_pad=0.00015, rel_pad=0.20)

    plt.figure(figsize=(8, 5))
    for m in ["EWC", "Hybrid"]:
        d = lam_df[lam_df["model"] == m].sort_values("lambda_val")
        if not d.empty:
            plt.plot(d["lambda_val"], d["avg_ndcg"], marker="o", label=m)
    plt.xscale("log")
    plt.xlabel("Lambda (log scale)")
    plt.ylabel("Avg NDCG@10")
    plt.title("Lambda vs Avg NDCG@10")
    plt.ylim(ndcg_ymin, ndcg_ymax)
    plt.legend()
    plt.tight_layout()
    plot1 = os.path.join(results_dir, "plot1_lambda_vs_ndcg.png")
    plt.savefig(plot1, dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    for m in ["EWC", "Hybrid"]:
        d = lam_df[lam_df["model"] == m].sort_values("lambda_val")
        if not d.empty:
            plt.plot(d["lambda_val"], d["avg_forgetting_ndcg"], marker="o", label=m)
    plt.xscale("log")
    plt.xlabel("Lambda (log scale)")
    plt.ylabel("Avg Forgetting (NDCG)")
    plt.title("Lambda vs Avg Forgetting (NDCG)")
    f_max = float(lam_df["avg_forgetting_ndcg"].abs().max()) if not lam_df.empty else 0.001
    f_lim = max(0.0003, f_max * 1.25)
    plt.ylim(-f_lim, f_lim)
    plt.axhline(0, color="gray", linestyle="--", linewidth=1)
    plt.legend()
    plt.tight_layout()
    plot2 = os.path.join(results_dir, "plot2_lambda_vs_forgetting.png")
    plt.savefig(plot2, dpi=180)
    plt.close()

    baseline_row = summary[summary["model"] == "Baseline"].head(1)
    replay_row = summary[summary["model"] == "Replay"].head(1)
    best_ewc = summary[summary["model"] == "EWC"].sort_values("avg_ndcg", ascending=False).head(1)
    best_hybrid = summary[summary["model"] == "Hybrid"].sort_values("avg_ndcg", ascending=False).head(1)
    
    # Only concat non-empty rows
    rows_to_concat = [r for r in [baseline_row, replay_row, best_ewc, best_hybrid] if not r.empty]
    if not rows_to_concat:
        raise SystemExit("No model data available for comparison plot. Check results directory.")
    comp = pd.concat(rows_to_concat, ignore_index=True)

    def fmt_label(row):
        if row["model"] in ["EWC", "Hybrid"]:
            return f"{row['model']} (lambda={row['lambda']})"
        return str(row["model"])

    comp["label"] = comp.apply(fmt_label, axis=1)

    # Validate that we have data to plot
    if comp.empty:
        raise SystemExit("No comparison data after filtering. Check results.")
    
    plt.figure(figsize=(9, 5))
    ax = sns.barplot(data=comp, x="label", y="avg_ndcg", hue="label", palette="deep", legend=False)
    ax.set_xlabel("Model")
    ax.set_ylabel("Avg NDCG@10")
    ax.set_title(f"Model Comparison (Best lambda for EWC/Hybrid) - {len(comp)} models")
    ax.set_ylim(ndcg_ymin, ndcg_ymax)
    for i, v in enumerate(comp["avg_ndcg"]):
        ax.text(i, v + 0.0003, f"{v:.4f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plot3 = os.path.join(results_dir, "plot3_model_comparison.png")
    plt.savefig(plot3, dpi=180)
    plt.close()

    # Small, slide-ready forgetting table for core model comparison.
    comp_table = comp[["model", "lambda", "avg_forgetting_ndcg", "avg_ndcg"]].copy()
    comp_table["lambda"] = comp_table["lambda"].astype(str)
    comp_table["avg_forgetting_ndcg"] = comp_table["avg_forgetting_ndcg"].map(lambda x: f"{x:.6f}")
    comp_table["avg_ndcg"] = comp_table["avg_ndcg"].map(lambda x: f"{x:.6f}")
    slide_table_out = os.path.join(results_dir, "forgetting_comparison_slide.png")

    fig, ax = plt.subplots(figsize=(8.5, 2.4))
    ax.axis("off")
    table = ax.table(
        cellText=comp_table.values,
        colLabels=["Model", "Lambda", "Avg Forgetting (NDCG)", "Avg NDCG@10"],
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.4)
    plt.title("Forgetting Comparison (Baseline vs Replay vs Best EWC/Hybrid)", pad=8)
    plt.tight_layout()
    plt.savefig(slide_table_out, dpi=180, bbox_inches="tight")
    plt.close()

    print("SUMMARY_TABLE", summary_out)
    print("FORGETTING_TABLE", forgetting_out)
    print("PLOT1", plot1)
    print("PLOT2", plot2)
    print("PLOT3", plot3)
    print("SLIDE_TABLE", slide_table_out)
    print("\nTop summary:")
    print(summary[["model", "lambda", "avg_ndcg", "avg_forgetting_ndcg"]].to_string(index=False))

    best_ewc_row = summary[summary["model"] == "EWC"].sort_values("avg_ndcg", ascending=False).head(1)
    best_hybrid_row = summary[summary["model"] == "Hybrid"].sort_values("avg_ndcg", ascending=False).head(1)
    if not best_ewc_row.empty:
        print(
            "\nBEST_EWC_LAMBDA",
            best_ewc_row["lambda"].iloc[0],
            "NDCG",
            round(float(best_ewc_row["avg_ndcg"].iloc[0]), 6),
            "F_NDCG",
            round(float(best_ewc_row["avg_forgetting_ndcg"].iloc[0]), 6),
        )
    if not best_hybrid_row.empty:
        print(
            "BEST_HYBRID_LAMBDA",
            best_hybrid_row["lambda"].iloc[0],
            "NDCG",
            round(float(best_hybrid_row["avg_ndcg"].iloc[0]), 6),
            "F_NDCG",
            round(float(best_hybrid_row["avg_forgetting_ndcg"].iloc[0]), 6),
        )


if __name__ == "__main__":
    main()
