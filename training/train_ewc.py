"""
training/train_ewc.py

EWC only: NCF trained on current chunk with Elastic Weight Consolidation.
No replay buffer.

Fisher computation:   Mini-batch, normalized by N (scale-invariant).
Fisher accumulation:  Summed across tasks (protects ALL prior tasks,
                      not just the most recent one).
Anchor parameters:    Updated after each task.
Evaluation protocol:  1-positive-vs-99-negatives per user (NCF benchmark).

Lambda sweep usage:
    python training/train_ewc.py --lambda_ewc 0.1
    python training/train_ewc.py --lambda_ewc 1
    python training/train_ewc.py --lambda_ewc 10
    python training/train_ewc.py --lambda_ewc 100
"""

import argparse
import os
import sys

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.ncf import NCF
from evaluation.metrics import evaluate_model
from training.utils import (
    set_seed,
    compute_fisher,
    ewc_loss,
    verify_label_distribution,
    ForgettingTracker,
    assert_prediction_label_shape,
)


# CLI

def parse_args():
    p = argparse.ArgumentParser(description="EWC-only NCF")
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--num_tasks",   type=int,   default=5)
    p.add_argument("--num_epochs",  type=int,   default=10)
    p.add_argument("--batch_size",  type=int,   default=256)
    p.add_argument("--lr",          type=float, default=0.001)
    p.add_argument("--lambda_ewc",  type=float, default=10.0,
                   help="EWC regularization coefficient")
    p.add_argument("--data_dir",    type=str,   default="fixed_tasks")
    p.add_argument("--results_dir", type=str,   default="results")
    return p.parse_args()


# Main

def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Config: seed={args.seed}, epochs={args.num_epochs}, "
          f"batch={args.batch_size}, lr={args.lr}, lambda={args.lambda_ewc}")

    base_dir    = os.path.dirname(os.path.abspath(__file__))
    data_dir    = os.path.join(base_dir, "..", args.data_dir)
    lam_str     = str(args.lambda_ewc).replace(".", "p")
    model_dir   = os.path.join(base_dir, "..", "saved_models",
                               f"ewc_lambda{lam_str}")
    results_dir = os.path.join(base_dir, "..", args.results_dir)
    os.makedirs(model_dir,   exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    meta      = torch.load(os.path.join(data_dir, "meta.pt"), weights_only=True)
    num_users = meta["num_users"]
    num_items = meta["num_items"]
    print(f"Dataset: {num_users} users, {num_items} items")

    model     = NCF(num_users, num_items).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCELoss()

    # EWC state - accumulated across tasks
    fisher_dict = {}   # param_name -> accumulated Fisher diagonal (CPU)
    optpar_dict = {}   # param_name -> anchor parameter values (CPU)

    tracker  = ForgettingTracker()
    all_rows = []

    for t in range(args.num_tasks):
        print(f"\n{'='*50}")
        print(f"TASK {t}  (EWC only, lambda={args.lambda_ewc})")
        print(f"{'='*50}")

        # Load current task's training data
        task_data  = torch.load(os.path.join(data_dir, f"task_{t}.pt"), weights_only=True)
        train_data = task_data["train"]
        verify_label_distribution(train_data, t)

        user  = train_data["user"]
        item  = train_data["item"]
        label = train_data["label"]

        dataset = TensorDataset(user, item, label)
        loader_gen = torch.Generator().manual_seed(args.seed + t)
        loader  = DataLoader(dataset,
                             batch_size=args.batch_size,
                             shuffle=True,
                     drop_last=False,
                     generator=loader_gen)

        # Train
        model.train()
        for epoch in range(args.num_epochs):
            epoch_loss     = 0.0
            epoch_bce_loss = 0.0
            epoch_ewc_loss = 0.0

            for b_u, b_i, b_l in loader:
                b_u = b_u.to(device)
                b_i = b_i.to(device)
                b_l = b_l.to(device)

                preds    = model(b_u, b_i).squeeze(-1)
                assert_prediction_label_shape(preds, b_l, context=f"ewc task={t} epoch={epoch+1}")
                bce      = criterion(preds, b_l)

                # EWC penalty only applies after the first task when lambda > 0
                if t > 0 and args.lambda_ewc > 0:
                    ewc_pen = args.lambda_ewc * ewc_loss(model, fisher_dict, optpar_dict)
                else:
                    ewc_pen = torch.tensor(0.0, device=device)

                loss = bce + ewc_pen

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                n = len(b_u)
                epoch_loss     += loss.item()     * n
                epoch_bce_loss += bce.item()      * n
                epoch_ewc_loss += ewc_pen.item()  * n

            N = len(dataset)
            print(f"  Epoch {epoch+1}/{args.num_epochs}  "
                  f"total={epoch_loss/N:.4f}  "
                  f"bce={epoch_bce_loss/N:.4f}  "
                  f"ewc={epoch_ewc_loss/N:.4f}")

        # Compute Fisher on current task's data
        # Design choice: Fisher computed on current task only (not mixed).
        # Rationale: EWC anchors parameters to their state after the current
        # task; the Fisher measures how sensitive those parameters are to the
        # current task's distribution. Future tasks will further accumulate
        # Fisher for parameters important to their respective distributions.
        if args.lambda_ewc > 0:
            new_fisher = compute_fisher(model, train_data, criterion, device,
                                        batch_size=args.batch_size)

            # Accumulate Fisher across tasks
            # Summing (not replacing) ensures parameters important to ANY prior
            # task remain protected, not just the most recent one.
            for name, f in new_fisher.items():
                if name in fisher_dict:
                    fisher_dict[name] = fisher_dict[name] + f.cpu()
                else:
                    fisher_dict[name] = f.cpu().clone()

            # Update parameter anchors
            for name, param in model.named_parameters():
                optpar_dict[name] = param.data.cpu().clone()

        # Evaluate on all tasks seen so far
        model.eval()
        current_results = {}

        with torch.no_grad():
            for k in range(t + 1):
                test_data = torch.load(os.path.join(data_dir, f"task_{k}.pt"), weights_only=True)["test"]
                recall, ndcg = evaluate_model(model, test_data, device)
                current_results[k] = (recall, ndcg)
                print(f"  Eval task {k}: Recall@10={recall:.4f}  NDCG@10={ndcg:.4f}")

                all_rows.append({
                    "trained_up_to_task": t,
                    "eval_task":          k,
                    "model":              "ewc",
                    "lambda_ewc":         args.lambda_ewc,
                    "seed":               args.seed,
                    "recall@10":          recall,
                    "ndcg@10":            ndcg,
                })

        # Forgetting
        avg_f_recall, avg_f_ndcg = tracker.compute_forgetting(current_results)
        print(f"  Avg forgetting — Recall: {avg_f_recall:.4f}  NDCG: {avg_f_ndcg:.4f}")

        for k, (recall, ndcg) in current_results.items():
            tracker.update(k, recall, ndcg)

        all_rows.append({
            "trained_up_to_task": t,
            "eval_task":          "avg",
            "model":              "ewc",
            "lambda_ewc":         args.lambda_ewc,
            "seed":               args.seed,
            "recall@10":          sum(r for r, n in current_results.values()) / len(current_results),
            "ndcg@10":            sum(n for r, n in current_results.values()) / len(current_results),
            "forgetting_recall":  avg_f_recall,
            "forgetting_ndcg":    avg_f_ndcg,
        })

        torch.save(model.state_dict(),
                   os.path.join(model_dir, f"ewc_task_{t}.pt"))

    out_path = os.path.join(results_dir,
                            f"ewc_lambda{lam_str}_seed{args.seed}.csv")
    pd.DataFrame(all_rows).to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")
    print("EWC training complete.")


if __name__ == "__main__":
    main()
