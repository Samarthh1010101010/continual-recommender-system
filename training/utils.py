"""
Shared utilities for all training scripts.
Covers: reproducibility seeding, Fisher computation (mini-batch, normalized),
EWC penalty, replay buffer with reservoir sampling, and evaluation helpers.
"""

import random
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


DEBUG = os.getenv("DEBUG", "0") == "1"


# Reproducibility

def set_seed(seed: int) -> None:
    """Fix all random sources for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Makes CUDA ops deterministic (slight perf cost, worth it for research)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def assert_prediction_label_shape(pred: torch.Tensor, label: torch.Tensor, context: str = "") -> None:
    """Fail fast when model predictions and labels do not align."""
    if not DEBUG:
        return

    assert pred.shape == label.shape, (
        f"{context} prediction/label shape mismatch: "
        f"pred={tuple(pred.shape)} label={tuple(label.shape)}"
    )

    assert pred.ndim == 1, (
        f"{context} prediction should be 1D after squeeze(-1), got {pred.ndim}D"
    )

    assert pred.dtype == torch.float32, (
        f"{context} prediction dtype must be float32, got {pred.dtype}"
    )

    assert label.dtype == torch.float32, (
        f"{context} label dtype must be float32, got {label.dtype}"
    )


# Fisher computation

def compute_fisher(model: nn.Module,
                   data: dict,
                   criterion: nn.Module,
                   device: torch.device,
                   batch_size: int = 256) -> dict:
    """
    Compute the diagonal Fisher Information Matrix via mini-batch SGD.

    The Fisher diagonal F_i for parameter theta_i is estimated as:
        F_i = (1/N) * sum_n [ (d loss_n / d theta_i)^2 ]

    Critically:
      - We loop over mini-batches so large datasets don't OOM.
      - We divide by total N so the scale is independent of dataset size.
        This means your lambda is comparable across tasks of different sizes
        and your lambda-tuning experiments are valid.
      - model is restored to train() mode before returning.

    Args:
        model:      The NCF model (already trained on current task).
        data:       Dict with keys "user", "item", "label" (CPU tensors).
        criterion:  BCELoss instance.
        device:     torch.device.
        batch_size: Mini-batch size for Fisher accumulation.

    Returns:
        fisher: Dict[param_name -> Tensor], same shape as model params.
    """
    fisher = {name: torch.zeros_like(param)
              for name, param in model.named_parameters()}

    model.eval()
    model.zero_grad()

    user = data["user"]
    item = data["item"]
    label = data["label"]
    N = len(user)

    if N == 0:
        model.train()
        return fisher

    dataset = TensorDataset(user, item, label)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    for b_u, b_i, b_l in loader:
        b_u = b_u.to(device)
        b_i = b_i.to(device)
        b_l = b_l.to(device)

        model.zero_grad()
        preds = model(b_u, b_i).squeeze(-1)
        loss = criterion(preds, b_l)
        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is not None:
                # Accumulate squared gradients; normalize by N at the end
                fisher[name] += param.grad.data.pow(2) * len(b_u)

    # Normalize by total dataset size -> scale-invariant Fisher
    for name in fisher:
        fisher[name] /= N

    # Always restore training mode
    model.train()
    return fisher


# EWC penalty

def ewc_loss(model: nn.Module,
             fisher_dict: dict,
             optpar_dict: dict) -> torch.Tensor:
    """
    Compute the EWC regularization penalty:
        L_ewc = sum_i [ F_i * (theta_i - theta*_i)^2 ]

    where theta*_i are the parameters after the previous task and
    F_i is the corresponding Fisher diagonal entry.

    Note on Fisher accumulation strategy (online EWC):
        We accumulate Fisher across tasks (sum). This means parameters
        important to ANY prior task are protected, not just the last one.
        The alternative (replace each task) only protects the most recent
        task's parameters, which is incorrect for multi-task continual
        learning. See: Schwarz et al. 2018 "Progress & Compress".

    Args:
        model:       Current model.
        fisher_dict: Accumulated Fisher diagonals {name -> Tensor}.
        optpar_dict: Anchored parameters from after last task {name -> Tensor}.

    Returns:
        Scalar EWC loss tensor.
    """
    loss = torch.tensor(0.0, device=next(model.parameters()).device)

    for name, param in model.named_parameters():
        if name in fisher_dict:
            fisher = fisher_dict[name].to(param.device)
            optpar = optpar_dict[name].to(param.device)
            loss += (fisher * (param - optpar).pow(2)).sum()

    return loss


# Replay buffer - reservoir sampling

class ReplayBuffer:
    """
    Fixed-size replay buffer using reservoir sampling.

    Reservoir sampling guarantees that every seen interaction has an
    equal probability of being in the buffer, regardless of which task
    it came from. This prevents FIFO truncation from silently biasing
    the buffer toward the most recent task.

    Reference: Vitter, J.S. (1985). "Random sampling with a reservoir."
    """

    def __init__(self, max_size: int):
        self.max_size = max_size
        self.buffer = []  # List of (user, item, label) scalar tensors
        self.n_seen = 0   # Total interactions seen so far

    def add(self, users: torch.Tensor,
                  items: torch.Tensor,
                  labels: torch.Tensor) -> None:
        """
        Add a batch of interactions using reservoir sampling.

        Args:
            users:  1-D LongTensor of user IDs.
            items:  1-D LongTensor of item IDs.
            labels: 1-D FloatTensor of labels (0 or 1).
        """
        for u, i, l in zip(users.tolist(), items.tolist(), labels.tolist()):
            self.n_seen += 1
            if len(self.buffer) < self.max_size:
                self.buffer.append((u, i, l))
            else:
                # Replace a random existing entry with probability max_size/n_seen
                j = random.randint(0, self.n_seen - 1)
                if j < self.max_size:
                    self.buffer[j] = (u, i, l)

    def sample(self, n: int) -> tuple:
        """
        Sample n interactions from the buffer.

        Returns:
            (user_tensor, item_tensor, label_tensor) - all 1-D, CPU.
        """
        n = min(n, len(self.buffer))
        samples = random.sample(self.buffer, n)
        users = torch.tensor([s[0] for s in samples], dtype=torch.long)
        items = torch.tensor([s[1] for s in samples], dtype=torch.long)
        labels = torch.tensor([s[2] for s in samples], dtype=torch.float)
        return users, items, labels

    def __len__(self):
        return len(self.buffer)


# Data label verification

def verify_label_distribution(data: dict, task_id: int) -> None:
    """
    Sanity-check that the saved task data contains both positives and
    negatives. Crashes early with a clear message if only positives
    are present - this would cause the replay buffer to be all-positive
    and training to collapse.
    """
    labels = data["label"].float()
    n_pos = (labels > 0.5).sum().item()
    n_neg = (labels < 0.5).sum().item()
    ratio = n_pos / max(n_pos + n_neg, 1)

    print(f"  Task {task_id} label check: {n_pos} pos, {n_neg} neg "
          f"(pos ratio={ratio:.2f})")

    if n_neg == 0:
        raise ValueError(
            f"Task {task_id} training data contains ONLY positive labels. "
            "Negatives must be pre-sampled into task_t.pt by your data pipeline, "
            "or add negative sampling here before storing to replay buffer."
        )
    if n_pos == 0:
        raise ValueError(
            f"Task {task_id} training data contains ONLY negative labels (no positives). "
            "This is invalid for training. Check your data pipeline."
        )
    if ratio > 0.5:
        import warnings
        warnings.warn(
            f"Task {task_id}: positive ratio={ratio:.2f} is unusually high. "
            "Expected ~0.2 (1 pos : 4 neg). Check your data pipeline."
        )


# Forgetting tracker

class ForgettingTracker:
    """
    Tracks best Recall@10 and NDCG@10 per task across training steps,
    and computes average forgetting after each new task is trained.

    Forgetting for task k after training task t (t > k) is:
        F_k = best_metric_on_task_k_before_t  -  metric_on_task_k_after_t

    Both Recall and NDCG are tracked independently.
    """

    def __init__(self):
        # task_id -> list of (recall, ndcg) tuples, one per evaluation round
        self.history: dict = {}

    def update(self, task_id: int, recall: float, ndcg: float) -> None:
        if task_id not in self.history:
            self.history[task_id] = []
        self.history[task_id].append((recall, ndcg))

    def compute_forgetting(self, current_results: dict) -> tuple:
        """
        Returns (avg_forgetting_recall, avg_forgetting_ndcg).
        Only tasks seen more than once contribute (i.e., tasks before the current one).
        """
        f_recall_list, f_ndcg_list = [], []

        for task_id, (recall, ndcg) in current_results.items():
            if task_id in self.history and len(self.history[task_id]) > 0:
                best_recall = max(r for r, n in self.history[task_id])
                best_ndcg = max(n for r, n in self.history[task_id])
                f_recall_list.append(best_recall - recall)
                f_ndcg_list.append(best_ndcg - ndcg)

        avg_f_recall = sum(f_recall_list) / len(f_recall_list) if f_recall_list else 0.0
        avg_f_ndcg = sum(f_ndcg_list) / len(f_ndcg_list) if f_ndcg_list else 0.0
        if not f_recall_list:
            return float("nan"), float("nan")
        return avg_f_recall, avg_f_ndcg
