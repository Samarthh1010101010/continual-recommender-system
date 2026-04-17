"""
training/train_hybrid.py

Hybrid: NCF + EWC + Replay buffer.

Loss at each step:    BCE(current + replay data) + lambda * EWC_penalty
Training data:        current chunk + reservoir-sampled replay
Replay eviction:      Reservoir sampling (uniform across all tasks)
Fisher computation:   Mini-batch, normalized by N, computed on full training
                      mix (current + replay) so Fisher reflects the actual
                      distribution the model was trained on.
Fisher accumulation:  Summed across tasks (protects ALL prior tasks).
Evaluation protocol:  1-positive-vs-99-negatives per user (NCF benchmark).

Lambda sweep usage:
    python training/train_hybrid.py --lambda_ewc 0
    python training/train_hybrid.py --lambda_ewc 0.1
    python training/train_hybrid.py --lambda_ewc 1
    python training/train_hybrid.py --lambda_ewc 10
    python training/train_hybrid.py --lambda_ewc 100
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
    ReplayBuffer,
    assert_prediction_label_shape,
)


# CLI

def parse_args():
    p = argparse.ArgumentParser(description="Hybrid EWC + Replay NCF")
    p.add_argument("--seed",               type=int,   default=42)
    p.add_argument("--num_tasks",          type=int,   default=5)
    p.add_argument("--num_epochs",         type=int,   default=10)
    p.add_argument("--batch_size",         type=int,   default=256)
    p.add_argument("--lr",                 type=float, default=0.001)
    p.add_argument("--lambda_ewc",         type=float, default=10.0,
                   help="EWC regularization coefficient (0 = replay only)")
    p.add_argument("--buffer_size",        type=int,   default=5000,
                   help="Max interactions in replay buffer")
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
          f"lambda={args.lambda_ewc}, "
          f"buffer={args.buffer_size}, replay_sample={args.replay_sample_size}")

    base_dir    = os.path.dirname(os.path.abspath(__file__))
    data_dir    = os.path.join(base_dir, "..", args.data_dir)
    lam_str     = str(args.lambda_ewc).replace(".", "p")
    model_dir   = os.path.join(base_dir, "..", "saved_models",
                               f"hybrid_lambda{lam_str}")
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

    # EWC state
    fisher_dict = {}
    optpar_dict = {}

    # Replay buffer with reservoir sampling
    replay_buffer = ReplayBuffer(max_size=args.buffer_size)

    tracker  = ForgettingTracker()
    all_rows = []

    for t in range(args.num_tasks):
        print(f"\n{'='*50}")
        print(f"TASK {t}  (hybrid, lambda={args.lambda_ewc}, "
              f"buffer={len(replay_buffer)})")
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

            # Sanity check replayed labels
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
            print(f"  Training mix: {len(cur_user)} current + "
                  f"{len(rep_u)} replayed = {len(user)} total")
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
            epoch_loss     = 0.0
            epoch_bce_loss = 0.0
            epoch_ewc_loss = 0.0

            for b_u, b_i, b_l in loader:
                b_u = b_u.to(device)
                b_i = b_i.to(device)
                b_l = b_l.to(device)

                preds = model(b_u, b_i).squeeze(-1)
                assert_prediction_label_shape(preds, b_l, context=f"hybrid task={t} epoch={epoch+1}")
                bce   = criterion(preds, b_l)

                if t > 0 and args.lambda_ewc > 0:
                    ewc_pen = args.lambda_ewc * ewc_loss(
                        model, fisher_dict, optpar_dict
                    )
                else:
                    ewc_pen = torch.tensor(0.0, device=device)

                loss = bce + ewc_pen

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                n = len(b_u)
                epoch_loss     += loss.item()    * n
                epoch_bce_loss += bce.item()     * n
                epoch_ewc_loss += ewc_pen.item() * n

            N = len(dataset)
            print(f"  Epoch {epoch+1}/{args.num_epochs}  "
                  f"total={epoch_loss/N:.4f}  "
                  f"bce={epoch_bce_loss/N:.4f}  "
                  f"ewc={epoch_ewc_loss/N:.4f}")

        # Update replay buffer with current task's interactions
        # Add AFTER training, so current task is eligible for future replay.
        replay_buffer.add(cur_user, cur_item, cur_label)
        print(f"  Replay buffer updated: {len(replay_buffer)} stored")

        # Compute Fisher on full training mix
        # Design choice: Fisher is computed on the FULL training mix
        # (current + replay), not just the current chunk. Rationale:
        # the model was trained on this mixed distribution, so Fisher
        # should reflect the parameter sensitivity to that distribution.
        # This ensures EWC anchoring is consistent with what was trained.
        if args.lambda_ewc > 0:
            mixed_data = {
                "user":  user.cpu(),
                "item":  item.cpu(),
                "label": label.cpu(),
            }
            new_fisher = compute_fisher(model, mixed_data, criterion, device,
                                        batch_size=args.batch_size)

            # Accumulate Fisher across tasks
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
                    "model":              "hybrid",
                    "lambda_ewc":         args.lambda_ewc,
                    "buffer_size":        args.buffer_size,
                    "seed":               args.seed,
                    "recall@10":          recall,
                    "ndcg@10":            ndcg,
                })

        # Forgetting
        avg_f_recall, avg_f_ndcg = tracker.compute_forgetting(current_results)
        print(f"  Avg forgetting — Recall: {avg_f_recall:.4f}  NDCG: {avg_f_ndcg:.4f}")

        for k, (recall, ndcg) in current_results.items():
            tracker.update(k, recall, ndcg)

        avg_recall = sum(r for r, n in current_results.values()) / len(current_results)
        avg_ndcg   = sum(n for r, n in current_results.values()) / len(current_results)

        all_rows.append({
            "trained_up_to_task": t,
            "eval_task":          "avg",
            "model":              "hybrid",
            "lambda_ewc":         args.lambda_ewc,
            "buffer_size":        args.buffer_size,
            "seed":               args.seed,
            "recall@10":          avg_recall,
            "ndcg@10":            avg_ndcg,
            "forgetting_recall":  avg_f_recall,
            "forgetting_ndcg":    avg_f_ndcg,
        })

        torch.save(model.state_dict(),
                   os.path.join(model_dir, f"hybrid_task_{t}.pt"))

    out_path = os.path.join(
        results_dir,
        f"hybrid_lambda{lam_str}_buf{args.buffer_size}_seed{args.seed}.csv"
    )
    pd.DataFrame(all_rows).to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")
    print("Hybrid training complete.")


if __name__ == "__main__":
    main()
