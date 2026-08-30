# Continual Recommender — Catastrophic Forgetting Study

Neural collaborative filtering (NCF) trained on MovieLens, split into 5
sequential tasks. Compares naive fine-tuning against EWC, a replay buffer,
and a hybrid of both — against a **compute-matched joint-training upper
bound** built specifically so the comparison can't be won by giving one
config more optimizer steps than another.

## Results

7-way comparison, seed 42, 5 epochs/task (~96.9K optimizer steps per config,
joint included — drift 0.0155% from the sequential total). Table generated
by [`results/new/build_cl_table.py`](results/new/build_cl_table.py), which
self-checks Average Accuracy against each config's own per-task CSV before
printing anything.

| Method | Stored samples | ACC Recall@10 | ACC NDCG@10 | BWT NDCG@10 | % of joint NDCG |
|---|---|---|---|---|---|
| Baseline (sequential) | 0 | 0.4872 | 0.2634 | -0.0092 | 98.7% |
| Replay | 5,000 | 0.4867 | 0.2608 | -0.0093 | 97.7% |
| EWC (λ=10) | 0 | 0.4927 | 0.2647 | -0.0062 | 99.1% |
| **EWC (λ=100)** | 0 | 0.4974 | **0.2709** | **+0.0033** | **101.5%** |
| Hybrid (λ=10) | 5,000 | 0.4870 | 0.2623 | -0.0078 | 98.3% |
| Hybrid (λ=100) | 5,000 | 0.4958 | 0.2700 | +0.0037 | 101.2% |
| Joint (upper bound, 5 epochs) | all (~4.96M) | 0.4958 | 0.2669 | — | 100.0% |

**ACC** = Average Accuracy (Lopez-Paz & Ranzato 2017): mean per-task metric
after training on the final task. **BWT** = Backward Transfer, same paper:
negative means forgetting. Both computed from the standard formula, not
approximated — see the script for the exact per-task lookups.

### The headline finding, with the qualifier that makes it true

At matched compute, **EWC (λ=100) beats the joint-training upper bound**
(101.5% of its NDCG, positive BWT) — cheap regularization is compute-
efficient enough to fully offset forgetting on this benchmark, at equal
gradient budget.

That claim only survives with the compute-matched qualifier attached. A
second joint run at 3x the epoch budget (15 epochs, uncapped —
[`results/joint_15epoch_seed42.csv`](results/joint_15epoch_seed42.csv), not
folded into the table above so it can't be double-counted) reaches
**ACC NDCG@10 = 0.3114** — 15% above EWC(λ=100). The 5-epoch joint number was
undertrained, not a weaker method in principle. So the correct two-layer
statement is:

- **At equal compute**, sequential fine-tuning with cheap regularization
  matches or beats a pooled model — this is the resume-worthy, defensible
  finding.
- **Given more budget**, the pooled model's true ceiling is meaningfully
  higher, meaning real forgetting is happening — it's just masked at equal
  compute because the sequential methods make efficient use of their smaller
  per-config budget.

Never state "EWC beats joint training" without the first qualifier.

## Why the joint upper bound is a fair comparison, not a strawman

[`training/train_joint.py`](training/train_joint.py) pools all 5 tasks'
training data and shuffles it, rather than presenting tasks in sequence. To
keep this an honest upper bound rather than a comparison rigged either way:

- **Same architecture, optimizer, learning rate, and seed** as every
  sequential config — untuned by design, so any gap is attributable to data
  ordering, not hyperparameters.
- **Compute-matched by default**: 5 epochs over the pooled ~4.96M rows is
  ~96,905 optimizer steps at batch=256, against ~96,920 for 5 sequential
  tasks × 5 epochs each — a coincidence of near-equal task sizes that the
  script verifies and asserts (`drift < 1%`) before training even starts.
- **Evaluated identically** — the same `evaluate_model()` call, same 5 held-
  out test sets, same metric definitions as every other config.

## Holes worth volunteering before an interviewer finds them

1. **Task splits are near-i.i.d. MovieLens chunks, not a real distribution
   shift.** Forgetting is mild everywhere in this study (the entire
   comparison band sits within ~4 points of NDCG) — that's why a
   regularizer can out-noise an untuned pooled model at equal compute.
2. **13.3% of task 0's test positives also appear in task 0's own training
   set** — pre-existing leakage in the fixed task splits. It hits every
   config equally, so the relative comparisons above hold, but every
   absolute Recall@10/NDCG@10 number is inflated by it.
3. **Joint's advantage is partly cold-start coverage, not just memory
   retention.** It sees every user/item embedding from step one; the
   sequential model at task 0 has untrained embeddings for entities that
   only appear in task 3.
4. **Single seed (42), n=1**, no error bars, no validation-set early
   stopping — every reported number is the final epoch.

## Repo layout

```
models/ncf.py              NCF: 64-dim embeddings, MLP 128->64->1, sigmoid
training/train_*.py        baseline / ewc / replay / hybrid / joint
training/utils.py          Fisher computation, EWC penalty, reservoir-
                            sampling replay buffer, forgetting tracker
evaluation/metrics.py      Recall@10 / NDCG@10, shared by every config
fixed_tasks/                5 pre-split task chunks + meta (fixed across
                            all configs, for a fair comparison)
results/new/VALIDATED_*.csv per-task, per-config evaluation rows
results/new/build_cl_table.py   ACC + BWT table, self-checked
results/new/generate_presentation_plots.py  slide-ready PNGs
```

## Reproduce

```bash
python training/train_baseline.py --seed 42 --num_epochs 5
python training/train_ewc.py --seed 42 --num_epochs 5 --lambda_ewc 100
python training/train_replay.py --seed 42 --num_epochs 5 --buffer_size 5000
python training/train_hybrid.py --seed 42 --num_epochs 5 --lambda_ewc 100 --buffer_size 5000
python training/train_joint.py --seed 42 --num_epochs 5
python results/new/build_cl_table.py
```
