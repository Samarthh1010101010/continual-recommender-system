"""
Diagnostic script to investigate the 5 suspected issues in the capstone project.
Run this BEFORE running experiments to understand the results.
"""

import os
import torch
import pandas as pd
import numpy as np
from collections import defaultdict

def check_issue_1_perfect_recall():
    """Issue #1: Are all models achieving Recall@10 = 1.0?"""
    print("\n" + "="*70)
    print("ISSUE #1: Perfect Recall@10 = 1.0 for ALL Models?")
    print("="*70)
    
    summary = pd.read_csv("results/summary_table.csv")
    
    recall_values = summary["avg_recall"].unique()
    print(f"\nRecall@10 values in results: {recall_values}")
    print(f"All recalls are 1.0: {all(r == 1.0 for r in recall_values)}")
    
    if all(r == 1.0 for r in recall_values):
        print("\n⚠️ CONFIRMED: All models achieve 100% Recall@10")
        print("\nPossible causes:")
        print("  1. EXPECTED: Test set has 1 positive + 99 negatives = 100 candidates")
        print("              → With decent model: hitting 1/100 in top-10 is easy")
        print("  2. CONCERNING: Test positives are in training set (data leakage)")
        print("  3. CONCERNING: Evaluation bug (positives excluded from negatives)")
    
    return all(r == 1.0 for r in recall_values)


def check_issue_2_baseline_beats_cl():
    """Issue #2: Does baseline beat continual learning methods?"""
    print("\n" + "="*70)
    print("ISSUE #2: Baseline BEATS Continual Learning Methods?")
    print("="*70)
    
    summary = pd.read_csv("results/summary_table.csv")
    
    baseline_ndcg = summary[summary["model"] == "Baseline"]["avg_ndcg"].values[0]
    best_elinear_cl = summary[summary["model"].isin(["EWC", "Replay", "Hybrid"])]["avg_ndcg"].max()
    best_cl_row = summary[summary["model"].isin(["EWC", "Replay", "Hybrid"])].loc[
        summary["avg_ndcg"].idxmax()
    ]
    
    print(f"\nBaseline NDCG:        {baseline_ndcg:.10f}")
    print(f"Best CL NDCG:         {best_elinear_cl:.10f}")
    print(f"Best CL method:       {best_cl_row['model']} (λ={best_cl_row['lambda']})")
    print(f"Difference:           {baseline_ndcg - best_elinear_cl:.10f}")
    print(f"Baseline advantage:   {((baseline_ndcg - best_elinear_cl) / best_elinear_cl * 100):.4f}%")
    
    if baseline_ndcg > best_elinear_cl:
        print("\n⚠️ CONFIRMED: Baseline is BETTER than all CL methods!")
        print("\nThis is VERY SUSPICIOUS because:")
        print("  • CL methods should PREVENT catastrophic forgetting")
        print("  • If baseline wins, either:")
        print("    - There IS no catastrophic forgetting (tasks are too similar)")
        print("    - CL hyperparameters are poorly tuned (λ too high)")
        print("    - CL methods are hurting performance")
    else:
        print("\n✓ Good: Best CL method beats baseline")
    
    return baseline_ndcg > best_elinear_cl


def check_issue_3_low_variance():
    """Issue #3: Is variance extremely low (all methods identical)?"""
    print("\n" + "="*70)
    print("ISSUE #3: Extremely Low Variance Across Models?")
    print("="*70)
    
    summary = pd.read_csv("results/summary_table.csv")
    ndcg_values = summary["avg_ndcg"].values
    
    ndcg_min = ndcg_values.min()
    ndcg_max = ndcg_values.max()
    ndcg_mean = ndcg_values.mean()
    ndcg_std = ndcg_values.std()
    ndcg_range = ndcg_max - ndcg_min
    cv = (ndcg_std / ndcg_mean) * 100  # Coefficient of variation
    
    print(f"\nNDCG Statistics:")
    print(f"  Min:                  {ndcg_min:.10f}")
    print(f"  Max:                  {ndcg_max:.10f}")
    print(f"  Mean:                 {ndcg_mean:.10f}")
    print(f"  Std Dev:              {ndcg_std:.10f}")
    print(f"  Range:                {ndcg_range:.10f}")
    print(f"  Coefficient of Var:   {cv:.4f}%")
    print(f"  Range as % of mean:   {(ndcg_range/ndcg_mean)*100:.4f}%")
    
    if ndcg_range < 0.001:  # Less than 0.1% range
        print(f"\n⚠️ CONFIRMED: Variance is EXTREMELY low!")
        print(f"  All methods differ by less than {(ndcg_range/ndcg_mean)*100:.3f}% NDCG")
        print("\nThis suggests:")
        print("  • Methods are not differentiated (all equally effective/ineffective)")
        print("  • Task may be too easy (ceiling effect)")
        print("  • Task may be too similar (no real catastrophic forgetting)")
    
    return ndcg_range < 0.001


def check_issue_4_negative_forgetting():
    """Issue #4: Do any models show negative forgetting?"""
    print("\n" + "="*70)
    print("ISSUE #4: Negative Forgetting (Models Improving on Old Tasks)?")
    print("="*70)
    
    summary = pd.read_csv("results/summary_table.csv")
    
    negative_forgetting = summary[summary["avg_forgetting_ndcg"] < 0]
    
    print(f"\nModels with negative forgetting:")
    if len(negative_forgetting) == 0:
        print("  None detected ✓")
    else:
        for idx, row in negative_forgetting.iterrows():
            print(f"  • {row['model']} (λ={row['lambda']}): {row['avg_forgetting_ndcg']:.10f}")
    
    if len(negative_forgetting) > 0:
        print(f"\n⚠️ CONFIRMED: {len(negative_forgetting)} models show negative forgetting!")
        print("\nThis MIGHT be legitimate (positive transfer) BUT check:")
        print("  1. Is there test set leakage? (old test ≠ new train)")
        print("  2. Is the forgetting calculation correct?")
        print("  3. Are different random seeds used for evaluation?")
    
    return len(negative_forgetting) > 0


def check_issue_5_non_monotonic_forgetting():
    """Issue #5: Does forgetting monotonically increase?"""
    print("\n" + "="*70)
    print("ISSUE #5: Forgetting Non-Monotonically Increasing?")
    print("="*70)
    
    # Load baseline data
    baseline_df = pd.read_csv("results/baseline_seed42.csv")
    
    # Get average forgetting per task
    task_forgetting = {}
    for _, row in baseline_df.iterrows():
        if row["eval_task"] == "avg":
            task = int(row["trained_up_to_task"])
            forgetting = row["forgetting_ndcg"]
            task_forgetting[task] = forgetting
    
    print(f"\nBaseline forgetting by task:")
    for task in sorted(task_forgetting.keys()):
        print(f"  Task {task}: {task_forgetting[task]:.10f}")
    
    # Check if monotonically increasing
    is_monotonic = True
    for i in range(len(task_forgetting) - 1):
        if task_forgetting[i] > task_forgetting[i+1]:
            is_monotonic = False
            print(f"\n⚠️ Non-monotonic at Task {i} → {i+1}: {task_forgetting[i]:.10f} → {task_forgetting[i+1]:.10f}")
    
    if is_monotonic:
        print("\n✓ Forgetting is monotonically increasing (good)")
    else:
        print("\n⚠️ CONFIRMED: Forgetting is NOT monotonically increasing!")
        print("\nThis suggests:")
        print("  • Task N might help previous tasks (positive transfer)")
        print("  • Bug in forgetting calculation")
        print("  • Statistical noise (try multiple seeds)")
    
    return not is_monotonic


def check_catastrophic_forgetting():
    """Check if baseline actually exhibits catastrophic forgetting"""
    print("\n" + "="*70)
    print("DIAGNOSTIC: Does Baseline Have Catastrophic Forgetting?")
    print("="*70)
    
    baseline_df = pd.read_csv("results/baseline_seed42.csv")
    
    # Get NDCG for task 0 after each training step
    task_0_performance = []
    for task_trained in range(5):
        row = baseline_df[(baseline_df["trained_up_to_task"] == task_trained) & 
                          (baseline_df["eval_task"] == 0)]
        if not row.empty:
            ndcg = row["ndcg@10"].values[0]
            task_0_performance.append((task_trained, ndcg))
    
    print("\nTask 0 NDCG after each training step:")
    for task_trained, ndcg in task_0_performance:
        print(f"  After training task {task_trained}: {ndcg:.10f}")
    
    if len(task_0_performance) > 1:
        initial_ndcg = task_0_performance[0][1]
        final_ndcg = task_0_performance[-1][1]
        forgetting_amount = initial_ndcg - final_ndcg
        forgetting_pct = (forgetting_amount / initial_ndcg) * 100
        
        print(f"\nTask 0 degradation:")
        print(f"  Initial (task 0): {initial_ndcg:.10f}")
        print(f"  Final (after all): {final_ndcg:.10f}")
        print(f"  Absolute loss:    {forgetting_amount:.10f}")
        print(f"  Relative loss:    {forgetting_pct:.4f}%")
        
        if forgetting_pct < 0.5:
            print(f"\n⚠️ MAJOR CONCERN: Only {forgetting_pct:.3f}% forgetting on task 0!")
            print("   This is NEGLIGIBLE. Expected: 3-10% forgetting for sequential learning")
            print("   This explains why CL methods can't help - there's nothing to prevent!")
        elif forgetting_pct > 5:
            print(f"\n✓ GOOD: {forgetting_pct:.3f}% forgetting detected")
            print("   CL methods should be able to reduce this")


def check_test_set_structure():
    """Verify test set has expected structure"""
    print("\n" + "="*70)
    print("DIAGNOSTIC: Test Set Structure Verification")
    print("="*70)
    
    # Load one task to inspect
    try:
        task_data = torch.load("fixed_tasks/task_0.pt", weights_only=True)
        test_data = task_data["test"]
        
        users = test_data["user"].tolist()
        items = test_data["item"].tolist()
        labels = test_data["label"].tolist()
        
        unique_users = set(users)
        pos_count = sum(1 for l in labels if l > 0.5)
        neg_count = sum(1 for l in labels if l < 0.5)
        
        candidates_per_user = len(test_data["user"]) / len(unique_users)
        
        print(f"\nTask 0 test set:")
        print(f"  Total interactions: {len(users)}")
        print(f"  Unique users:       {len(unique_users)}")
        print(f"  Candidates/user:    {candidates_per_user:.1f}")
        print(f"  Positives:          {pos_count}")
        print(f"  Negatives:          {neg_count}")
        print(f"  Positive ratio:     {pos_count/(pos_count+neg_count)*100:.2f}%")
        
        if abs(candidates_per_user - 100) < 1:
            print(f"\n✓ Candidates per user ≈ 100 (1 pos + 99 neg)")
            print("  This explains 100% Recall@10 - it's by design")
        else:
            print(f"\n⚠️ Candidates per user = {candidates_per_user:.1f}, NOT 100")
            print("  This is unexpected for 1+99 setup")
            
    except Exception as e:
        print(f"\n⚠️ Could not load task data: {e}")


def main():
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + "CAPSTONE PROJECT - DIAGNOSTIC ANALYZER".center(68) + "█")
    print("█" + " "*68 + "█")
    print("█"*70)
    
    # Run all diagnostics
    issue1 = check_issue_1_perfect_recall()
    issue2 = check_issue_2_baseline_beats_cl()
    issue3 = check_issue_3_low_variance()
    issue4 = check_issue_4_negative_forgetting()
    issue5 = check_issue_5_non_monotonic_forgetting()
    
    check_catastrophic_forgetting()
    check_test_set_structure()
    
    # Summary report
    print("\n" + "="*70)
    print("SUMMARY OF FINDINGS")
    print("="*70)
    
    issues_found = sum([issue1, issue2, issue3, issue4, issue5])
    
    print(f"\nIssues confirmed: {issues_found}/5")
    print(f"  Issue #1 (Perfect Recall 1.0):           {'⚠️ YES' if issue1 else '✓ NO'}")
    print(f"  Issue #2 (Baseline beats CL):            {'⚠️ YES' if issue2 else '✓ NO'}")
    print(f"  Issue #3 (Low variance):                 {'⚠️ YES' if issue3 else '✓ NO'}")
    print(f"  Issue #4 (Negative forgetting):          {'⚠️ YES' if issue4 else '✓ NO'}")
    print(f"  Issue #5 (Non-monotonic forgetting):     {'⚠️ YES' if issue5 else '✓ NO'}")
    
    print("\n" + "="*70)
    print("ACTION ITEMS")
    print("="*70)
    
    if issue1:
        print("\n1. Perfect Recall@10 = 1.0")
        print("   ACTION: This is EXPECTED for 1+99 candidate setup")
        print("   VERIFY: Check if test_set really has 100 candidates per user")
    
    if issue2:
        print("\n2. Baseline beats CL methods")
        print("   ACTION: Investigate if there's actual catastrophic forgetting")
        print("   OPTION A: Lower lambda values (try 0.001, 0.01, 0.05)")
        print("   OPTION B: Make tasks more distinct (increase task separation)")
        print("   OPTION C: More tasks (try 10-20 instead of 5)")
    
    if issue3:
        print("\n3. Extremely low variance")
        print("   ACTION: Make the task harder to differentiate methods")
        print("   - Use harder negative sampling")
        print("   - Reduce epochs per task (force faster learning)")
        print("   - Increase task difficulty")
    
    if issue4:
        print("\n4. Negative forgetting")
        print("   ACTION: Verify forgetting calculation is correct")
        print("   CHECK: best_previous_score - current_score formula")
    
    if issue5:
        print("\n5. Non-monotonic forgetting")
        print("   ACTION: Run multiple seeds to check if it's noise")
        print("   DEBUG: Verify ForgettingTracker.compute_forgetting() logic")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
