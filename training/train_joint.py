"""
training/train_joint.py

Joint upper bound: one NCF trained on all 5 tasks pooled and shuffled,
instead of sequentially. This is what "no forgetting" looks like, since a
model that never leaves a task behind can't forget it.

Compute-matched, not epoch-count-matched: sequential does 5 epochs over each
~993k-row task chunk (5 tasks x 5 epochs = ~96,920 optimizer steps at
batch=256). Joint at 5 epochs over the ~4.96M pooled rows is ~96,905 steps -
same gradient budget, same number of passes over every interaction. Same
architecture/optimizer/lr/seed as every other config, untuned by design, so
any gap is attributable to data ordering, not hyperparameters.

Output rows use trained_up_to_task=4 as a sentinel (joint has no task
sequence to be "up to") so this drops into the same comparison table as the
other six VALIDATED_*.csv files. forgetting_recall/forgetting_ndcg are
written as 0.0 here (a model that never left a task has zero forgetting by
construction) - render that as "-" in the human-facing table instead, since
standard BWT needs a diagonal R(i,i) that joint doesn't have.
"""

import argparse
import math
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
    verify_label_distribution,
    assert_prediction_label_shape,
)


def parse_args():
    p = argparse.ArgumentParser(description="Joint upper bound (all tasks pooled)")
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--num_tasks",   type=int,   default=5)
    p.add_argument("--num_epochs",  type=int,   default=5)  # compute-matched default
    p.add_argument("--batch_size",  type=int,   default=256)
    p.add_argument("--lr",          type=float, default=0.001)
    p.add_argument("--data_dir",    type=str,   default="fixed_tasks")
    p.add_argument("--results_dir", type=str,   default="results")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Config: seed={args.seed}, epochs={args.num_epochs}, "
          f"batch={args.batch_size}, lr={args.lr}")

    base_dir    = os.path.dirname(os.path.abspath(__file__))
    data_dir    = os.path.join(base_dir, "..", args.data_dir)
    model_dir   = os.path.join(base_dir, "..", "saved_models", "joint")
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

    # Pool every task's TRAINING split only. Never touch ["test"] here -
    # evaluation still happens per-task, after training, same as every
    # other config.
    users, items, labels = [], [], []
    seq_steps = 0
    for t in range(args.num_tasks):
        task_data  = torch.load(os.path.join(data_dir, f"task_{t}.pt"), weights_only=True)
        train_data = task_data["train"]
        verify_label_distribution(train_data, t)
        users.append(train_data["user"])
        items.append(train_data["item"])
        labels.append(train_data["label"])
        seq_steps += math.ceil(len(train_data["user"]) / args.batch_size)

    user  = torch.cat(users)
    item  = torch.cat(items)
    label = torch.cat(labels)
    assert len(user) == sum(len(u) for u in users), "pooled size mismatch"
    print(f"Pooled: {len(user)} rows across {args.num_tasks} tasks")

    # Fairness self-check: compute-matched, not epoch-count-matched.
    joint_steps_per_epoch = math.ceil(len(user) / args.batch_size)
    seq_total   = seq_steps * args.num_epochs
    joint_total = joint_steps_per_epoch * args.num_epochs
    drift = abs(joint_total - seq_total) / seq_total
    print(f"Compute-match audit: sequential={seq_total} steps, "
          f"joint={joint_total} steps, drift={drift:.4%}")
    assert drift < 0.01, (
        f"Joint training compute diverged >1% from sequential total "
        f"({seq_total} vs {joint_total}) - not a fair upper-bound comparison."
    )

    dataset = TensorDataset(user, item, label)
    loader_gen = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                         drop_last=False, generator=loader_gen)

    model.train()
    for epoch in range(args.num_epochs):
        epoch_loss = 0.0
        for b_u, b_i, b_l in loader:
            b_u, b_i, b_l = b_u.to(device), b_i.to(device), b_l.to(device)

            preds = model(b_u, b_i).squeeze(-1)
            assert_prediction_label_shape(preds, b_l, context=f"joint epoch={epoch+1}")
            loss = criterion(preds, b_l)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(b_u)

        avg_loss = epoch_loss / len(dataset)
        print(f"  Epoch {epoch+1}/{args.num_epochs}  loss={avg_loss:.4f}")

    # Evaluate on all 5 tasks, same evaluate_model() every other config uses.
    model.eval()
    per_task = {}
    all_rows = []
    with torch.no_grad():
        for k in range(args.num_tasks):
            test_data = torch.load(os.path.join(data_dir, f"task_{k}.pt"), weights_only=True)["test"]
            recall, ndcg = evaluate_model(model, test_data, device)
            per_task[k] = (recall, ndcg)
            print(f"  Eval task {k}: Recall@10={recall:.4f}  NDCG@10={ndcg:.4f}")
            all_rows.append({
                "trained_up_to_task": args.num_tasks - 1,
                "eval_task":          k,
                "model":              "joint",
                "seed":               args.seed,
                "recall@10":          recall,
                "ndcg@10":            ndcg,
            })

    all_rows.append({
        "trained_up_to_task": args.num_tasks - 1,
        "eval_task":          "avg",
        "model":              "joint",
        "seed":               args.seed,
        "recall@10":          sum(r for r, n in per_task.values()) / len(per_task),
        "ndcg@10":            sum(n for r, n in per_task.values()) / len(per_task),
        "forgetting_recall":  0.0,
        "forgetting_ndcg":    0.0,
    })

    torch.save(model.state_dict(), os.path.join(model_dir, "joint_final.pt"))

    out_path = os.path.join(results_dir, f"joint_{args.num_epochs}epoch_seed{args.seed}.csv")
    pd.DataFrame(all_rows).to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")
    print("Joint training complete.")


if __name__ == "__main__":
    main()
