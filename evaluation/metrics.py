import os

import torch

from training.utils import assert_prediction_label_shape


def evaluate_model(model, data, device, k=10, debug=False, random_scores=False, print_rank=False):
    model.eval()

    debug = debug or os.getenv("EVAL_DEBUG", "0") == "1"
    random_scores = random_scores or os.getenv("EVAL_RANDOM_SCORES", "0") == "1"
    print_rank = print_rank or os.getenv("EVAL_PRINT_RANK", "0") == "1"

    user_dict = {}

    for u, i, l in zip(data["user"], data["item"], data["label"]):
        u = u.item()
        if u not in user_dict:
            user_dict[u] = {"items": [], "labels": []}
        user_dict[u]["items"].append(i.item())
        user_dict[u]["labels"].append(l.item())

    recalls = []
    ndcgs = []

    for u in user_dict:
        interactions = user_dict[u]
        candidate_items = torch.tensor(interactions["items"], dtype=torch.long, device=device)
        labels = torch.tensor(interactions["labels"], dtype=torch.float32, device=device)

        with torch.no_grad():
            scores = model(
                torch.full((len(candidate_items),), u, dtype=torch.long, device=device),
                candidate_items,
            )

        assert_prediction_label_shape(scores, labels, context=f"eval user={u}")

        if debug:
            print("Candidates:", len(candidate_items))

        if scores.numel() == 0:
            continue

        if labels.sum() == 0:
            continue

        if random_scores:
            scores = torch.rand(len(candidate_items), dtype=torch.float32, device=device)

        # Some users can have fewer than k candidates.
        # Clamp k to avoid out-of-range errors in torch.topk.
        k_eff = min(k, scores.numel())

        ranked_indices = torch.argsort(scores, descending=True)
        top_k_indices = ranked_indices[:k_eff]
        top_k_labels = labels[top_k_indices]

        if print_rank:
            positive_indices = torch.nonzero(labels > 0.5, as_tuple=False).flatten()
            if len(positive_indices) == 1:
                pos_item = positive_indices.item()
                rank = (ranked_indices == pos_item).nonzero(as_tuple=False).item()
                print("Rank:", rank)

        hit = (top_k_labels.sum() > 0).float().item()
        recalls.append(hit)

        gains = top_k_labels
        discounts = torch.log2(torch.arange(len(gains), dtype=torch.float32, device=gains.device) + 2)
        dcg = (gains / discounts).sum().item()

        ndcgs.append(dcg)  # idcg = 1

    if not recalls:
        import warnings
        warnings.warn(
            f"No valid evaluations computed: {len(user_dict)} users had zero candidates "
            "or zero positive labels. Check test data validity."
        )

    recall = sum(recalls) / len(recalls) if recalls else 0
    ndcg = sum(ndcgs) / len(ndcgs) if ndcgs else 0

    return recall, ndcg
