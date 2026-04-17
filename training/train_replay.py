"""
training/train_replay.py

Replay buffer only: NCF trained on current chunk + reservoir-sampled
past interactions. No EWC penalty.

Training data scope:  current chunk + replay samples.
Replay eviction:      Reservoir sampling (uniform across all past tasks).
Evaluation protocol:  1-positive-vs-99-negatives per user (NCF benchmark).
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
    verify_label_distribution,
    ForgettingTracker,
    ReplayBuffer,
    assert_prediction_label_shape,
)


# CLI

def parse_args():
    p = argparse.ArgumentParser(description="Replay-only NCF")
    p.add_argument("--seed",               type=int,   default=42)
    p.add_argument("--num_tasks",          type=int,   default=5)
    p.add_argument("--num_epochs",         type=int,   default=10)
    p.add_argument("--batch_size",         type=int,   default=256)
    p.add_argument("--lr",                 type=float, default=0.001)
    p.add_argument("--buffer_size",        type=int,   default=5000,
                   help="Max interactions stored in replay buffer")
    p.add_argument("--replay_sample_size", type=int,   default=1000,
                   help="Interactions sampled from buffer per task")
    p.add_argument("--data_dir",           type=str,   default="fixed_tasks")
    p.add_argument("--results_dir",        type=str,   default="results")
    return p.parse_args()


# Main

def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Config: seed={args.seed}, epochs={args.num_epochs}, "
          f"batch={args.batch_size}, lr={args.lr}, "
          f"buffer={args.buffer_size}, replay_sample={args.replay_sample_size}")

    base_dir    = os.path.dirname(os.path.abspath(__file__))
    data_dir    = os.path.join(base_dir, "..", args.data_dir)
    model_dir   = os.path.join(base_dir, "..", "saved_models", "replay")
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

    # Reservoir-sampled replay buffer
    replay_buffer = ReplayBuffer(max_size=args.buffer_size)

    tracker  = ForgettingTracker()
    all_rows = []

    for t in range(args.num_tasks):
        print(f"\n{'='*50}")
        print(f"TASK {t}  (replay only, buffer size={len(replay_buffer)})")
        print(f"{'='*50}")

        # Load current task's training data
        task_data  = torch.load(os.path.join(data_dir, f"task_{t}.pt"), weights_only=True)
        train_data = task_data["train"]
        verify_label_distribution(train_data, t)

        cur_user  = train_data["user"]
        cur_item  = train_data["item"]
        cur_label = train_data["label"]

        # Mix in replay samples
        if len(replay_buffer) > 0:
            rep_u, rep_i, rep_l = replay_buffer.sample(args.replay_sample_size)
            # Verify replayed labels also contain both classes
            n_rep_pos = (rep_l > 0.5).sum().item()
            n_rep_neg = (rep_l < 0.5).sum().item()
            if n_rep_pos == 0:
                raise ValueError(
                    f"Replay sample at task {t} contains ONLY negatives (no positives). "
                    "Buffer is corrupted or all stored positives were purged."
                )
            if n_rep_neg == 0:
                raise ValueError(
                    f"Replay sample at task {t} contains ONLY positives (no negatives). "
                    "Buffer may be all-positive — check data pipeline."
                )
            user  = torch.cat([cur_user,  rep_u])
            item  = torch.cat([cur_item,  rep_i])
            label = torch.cat([cur_label, rep_l])
            print(f"  Training on {len(cur_user)} current + "
                  f"{len(rep_u)} replayed = {len(user)} total samples")
        else:
            user, item, label = cur_user, cur_item, cur_label
            print(f"  Training on {len(user)} samples (no replay yet)")

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
            epoch_loss = 0.0
            for b_u, b_i, b_l in loader:
                b_u = b_u.to(device)
                b_i = b_i.to(device)
                b_l = b_l.to(device)

                preds = model(b_u, b_i).squeeze(-1)
                assert_prediction_label_shape(preds, b_l, context=f"replay task={t} epoch={epoch+1}")
                loss  = criterion(preds, b_l)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * len(b_u)

            avg_loss = epoch_loss / len(dataset)
            print(f"  Epoch {epoch+1}/{args.num_epochs}  loss={avg_loss:.4f}")

        # Update replay buffer with current task's interactions
        # Add AFTER training so current task is eligible for future replay
        replay_buffer.add(cur_user, cur_item, cur_label)
        print(f"  Replay buffer updated: {len(replay_buffer)} stored")

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
                    "model":              "replay",
                    "seed":               args.seed,
                    "buffer_size":        args.buffer_size,
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
            "model":              "replay",
            "seed":               args.seed,
            "buffer_size":        args.buffer_size,
            "recall@10":          sum(r for r, n in current_results.values()) / len(current_results),
            "ndcg@10":            sum(n for r, n in current_results.values()) / len(current_results),
            "forgetting_recall":  avg_f_recall,
            "forgetting_ndcg":    avg_f_ndcg,
        })

        torch.save(model.state_dict(),
                   os.path.join(model_dir, f"replay_task_{t}.pt"))

    out_path = os.path.join(results_dir,
                            f"replay_buf{args.buffer_size}_seed{args.seed}.csv")
    pd.DataFrame(all_rows).to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")
    print("Replay training complete.")


if __name__ == "__main__":
    main()
