"""
training/rebuild_fixed_tasks.py

Rebuild the saved task files with a proper evaluation candidate pool.

The existing task artifacts contain only one positive and one negative in each
test split, which makes Recall@10 trivially equal to 1.0. This script rewrites
the task files so that each user is evaluated against the positive item plus a
larger negative candidate pool sampled from the full item universe.

The training split is preserved as-is.

Usage:
    python training/rebuild_fixed_tasks.py --input_dir fixed_tasks --output_dir fixed_tasks --num_negatives 99
"""

import argparse
import os
import random
import shutil
import tempfile

import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Rebuild fixed task files")
    parser.add_argument("--input_dir", type=str, default="fixed_tasks")
    parser.add_argument("--output_dir", type=str, default="fixed_tasks")
    parser.add_argument("--num_tasks", type=int, default=5)
    parser.add_argument("--num_negatives", type=int, default=99)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_user_history(split: dict) -> dict:
    history = {}
    for user, item, label in zip(split["user"].tolist(), split["item"].tolist(), split["label"].tolist()):
        if float(label) > 0.5:
            history.setdefault(user, set()).add(item)
    return history


def merge_histories(base_history: dict, extra_history: dict) -> dict:
    merged = {user: set(items) for user, items in base_history.items()}
    for user, items in extra_history.items():
        merged.setdefault(user, set()).update(items)
    return merged


def rebuild_test_split(test_split: dict,
                      excluded_history: dict,
                      num_items: int,
                      num_negatives: int,
                      rng: random.Random) -> dict:
    users = []
    items = []
    labels = []
    dropped_users = []  # Track users with no positives

    test_users = test_split["user"].tolist()
    test_items = test_split["item"].tolist()
    test_labels = test_split["label"].tolist()

    user_rows = {}
    for user, item, label in zip(test_users, test_items, test_labels):
        user_rows.setdefault(user, []).append((item, label))

    all_items = list(range(num_items))

    for user in sorted(user_rows):
        rows = user_rows[user]
        positives = [item for item, label in rows if float(label) > 0.5]
        if not positives:
            dropped_users.append(user)
            continue

        excluded = set(excluded_history.get(user, set()))
        excluded.update(item for item, _ in rows)

        candidate_pool = [item for item in all_items if item not in excluded]
        if len(candidate_pool) < num_negatives:
            raise ValueError(
                f"User {user} only has {len(candidate_pool)} negatives available, "
                f"but {num_negatives} were requested."
            )

        neg_items = rng.sample(candidate_pool, num_negatives)
        
        # Use ALL positives, not just first
        for pos_item in positives:
            assert pos_item not in neg_items
            users.append(user)
            items.append(pos_item)
            labels.append(1.0)
        
        # One set of negatives per user (shared across all their positives)
        users.extend([user] * len(neg_items))
        items.extend(neg_items)
        labels.extend([0.0] * len(neg_items))
    
    if dropped_users:
        import warnings
        warnings.warn(
            f"Dropped {len(dropped_users)} users with no positive labels in test split. "
            f"Output test size will be smaller (affected user count: {len(dropped_users)})"
        )

    return {
        "user": torch.tensor(users, dtype=torch.long),
        "item": torch.tensor(items, dtype=torch.long),
        "label": torch.tensor(labels, dtype=torch.float32),
    }


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir)
    staging_dir = output_dir
    if os.path.normcase(input_dir) == os.path.normcase(output_dir):
        staging_dir = tempfile.mkdtemp(prefix="rebuild_fixed_tasks_", dir=os.path.dirname(output_dir))
    os.makedirs(staging_dir, exist_ok=True)

    meta = torch.load(os.path.join(input_dir, "meta.pt"), map_location="cpu", weights_only=True)
    torch.save(meta, os.path.join(staging_dir, "meta.pt"))

    cumulative_train_history = {}
    source_tasks = []

    for task_id in range(args.num_tasks):
        task_path = os.path.join(input_dir, f"task_{task_id}.pt")
        source_tasks.append(torch.load(task_path, map_location="cpu", weights_only=True))

    for task_id in range(args.num_tasks):
        task_data = source_tasks[task_id]

        current_train_history = build_user_history(task_data["train"])
        excluded_history = merge_histories(cumulative_train_history, current_train_history)

        rebuilt_test = rebuild_test_split(
            task_data["test"],
            excluded_history,
            meta["num_items"],
            args.num_negatives,
            rng,
        )

        rebuilt_task = {
            "train": task_data["train"],
            "test": rebuilt_test,
        }

        out_path = os.path.join(staging_dir, f"task_{task_id}.pt")
        torch.save(rebuilt_task, out_path)
        print(f"Wrote {out_path}")

        cumulative_train_history = merge_histories(cumulative_train_history,
                                                   current_train_history)

    if staging_dir != output_dir:
        os.makedirs(output_dir, exist_ok=True)
        for name in ["meta.pt"] + [f"task_{task_id}.pt" for task_id in range(args.num_tasks)]:
            src_path = os.path.join(staging_dir, name)
            dst_path = os.path.join(output_dir, name)
            os.replace(src_path, dst_path)
        shutil.rmtree(staging_dir, ignore_errors=True)


if __name__ == "__main__":
    main()