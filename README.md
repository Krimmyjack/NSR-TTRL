## Project Overview

This repository is a **lightly modified fork** of [PRIME-RL/TTRL](https://github.com/PRIME-RL/TTRL). Instead of reintroducing TTRL from scratch, we focus on documenting the **additional engineering pieces** that were built on top:

- Token-level scoring & reward adjustments used during PPO training.
- GRPO-style advantage estimators that support NSR-style reweighting.
- Majority-vote powered TTRL utilities for data selection and evaluation.
- Example Hydra overrides for running the new pipeline on Qwen2.5-Math-1.5B.

If you need the canonical description of TTRL itself (motivation, benchmarks, paper links, etc.), please refer to the upstream README and paper. The rest of this document only highlights what changed in this derivative work.

## What’s New Compared to Upstream TTRL

- **Token-level score pipeline** (`verl/trainer/ppo/ray_trainer.py`, `verl/trainer/ppo/core_algos.py`)
  - Reward model outputs are expanded to `token_level_scores` (non-zero on the last valid response token).
  - Optional KL-in-reward penalty converts scores to `token_level_rewards` before advantage estimation.
  - Metrics log the score distribution as well as the NSR positive-token ratios.

- **GRPO + NSR advantage estimators** (`verl/trainer/ppo/core_algos.py`)
  - Added `compute_grpo_nsr_outcome_advantage` with configurable positive/negative weights and correctness threshold.
  - Supports switching between std-normalized or mean-only baselines per prompt group.
  - Shared helpers (`compute_rewards`, `agg_loss`) are wired so other estimators can reuse the same token-level tensors.

- **TTRL data utilities** (`verl/trainer/ppo/ttrl_utils.py`)
  - Majority-vote pseudo labels can be injected via `apply_ttrl_gt`, then reverted by `apply_original_gt`.
  - `compute_ttrl_metrics` now records majority vs. original rewards and label accuracy, enabling quick monitoring of the pseudo-label quality.

- **Training loop integration** (`verl/trainer/ppo/ray_trainer.py`)
  - Reward computation, KL penalty, advantage estimation, and metric logging happen on the driver before the Ray workers update actor/critic models.
  - Added hooks to dump rollout generations with their aggregated scores or to trigger validation under the same scoring pipeline.

## Key Files at a Glance

- `verl/trainer/ppo/ray_trainer.py`
  - `apply_kl_penalty`: subtracts `beta * KL` from `token_level_scores` → `token_level_rewards`.
  - `compute_advantage`: routes to the requested estimator (GAE / GRPO / GRPO_NSR / etc.) and feeds in `token_level_scores` when NSR correctness needs it.
  - Training loop block (around lines 1200+) shows the end-to-end order: reward → KL → advantage → actor/critic updates → optional TTRL metrics.

- `verl/trainer/ppo/core_algos.py`
  - Houses every registered advantage estimator plus shared helpers like `compute_rewards` and `agg_loss`.
  - `compute_grpo_nsr_outcome_advantage` demonstrates how we reuse `token_level_scores` for correctness masks while still consuming `token_level_rewards` for gradients.

- `verl/trainer/ppo/ttrl_utils.py`
  - Implements majority-vote logic (`apply_ttrl_gt`, `_batch_majority_vote`) and evaluation metrics (`compute_ttrl_metrics`).
  - Stores both original and majority-based scores so that experiments can compare “pseudo” vs. “ground-truth” feedback even when statistics disagree.

## Running the Extended Pipeline

1. **Environment**: follow the upstream instructions (`conda create -n ttrl python=3.10`, `bash scripts/install_ttrl_deps.sh`, `pip install -e .`).
2. **Hydra overrides**: the following command shows a concrete run we used for Qwen2.5-Math-1.5B with GRPO + TTRL enabled. Adjust file paths to your environment.

```bash
python verl/trainer/main_ppo.py \
  data.train_files=[/root/autodl-tmp/TTRL/verl/data/MATH-TTT/answer.jsonl_0.4_pi1_r32_repeatto32.parquet] \
  data.val_files=[/root/autodl-tmp/TTRL/verl/data/MATH-TTT/test.parquet] \
  data.max_prompt_length=1024 \
  data.max_response_length=3072 \
  data.train_batch_size=32 \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  actor_rollout_ref.model.path=/root/autodl-tmp/model/Qwen2.5-Math-1.5B \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=1 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.optim.lr=5e-7 \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.03 \
  actor_rollout_ref.actor.optim.warmup_style=cosine \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4096 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.n=32 \
  actor_rollout_ref.rollout.max_model_len=4096 \
  actor_rollout_ref.rollout.max_num_batched_tokens=4096 \
  critic.optim.lr=9e-6 \
  critic.model.use_remove_padding=True \
  critic.model.path=/root/autodl-tmp/model/Qwen2.5-Math-1.5B \
  critic.model.enable_gradient_checkpointing=True \
  critic.ppo_micro_batch_size_per_gpu=2 \
  algorithm.kl_ctrl.kl_coef=0.0 \
  algorithm.adv_estimator=grpo_nsr \
  custom_reward_function.path=./verl/utils/reward_score/ttrl_math/__init__.py \
  custom_reward_function.name=reward_func \
  ttrl.enable=True \
  ttrl.n_votes_per_prompt=64 \
  ttrl.n_samples_per_prompt=32 \
  trainer.logger=[console,wandb] \
  trainer.project_name=one-shot-TTRL-verl \
  trainer.experiment_name=test \
  trainer.n_gpus_per_node=8 \
  trainer.nnodes=1 \
  trainer.save_freq=40 \
  trainer.test_freq=5 \
  trainer.default_local_dir=checkpoints/one-shot-TTRL-verl/MATH-TTT-Qwen2.5-Math-1.5B/1126/TTRL-Len@3k-grpo-115533 \
  trainer.total_epochs=1000
```

The command looks long, but it mainly reflects the knobs exposed by the new components (GRPO-NSR, KL penalty, majority-vote settings). Smaller experiments can drop most of these overrides.

## Token-Level Scores vs. Rewards

- **token_level_scores**
  - Produced by the reward manager (either rule-based majority voting or a neural reward model).
  - Stored as a tensor shaped like the response, but only the last valid token carries the non-zero score.
  - Used for logging, for NSR correctness checks, and for pseudo-label bookkeeping in TTRL utilities.

- **token_level_rewards**
  - Derived from `token_level_scores` after subtracting the KL penalty (when `algorithm.use_kl_in_reward=True`).
  - Consumed by every advantage estimator and becomes the signal that backpropagates through PPO.

Visually, the flow per batch is:

```
reward_manager → token_level_scores
apply_kl_penalty (optional) → token_level_rewards
compute_advantage → advantages, returns
policy/value losses → actor/critic updates
```

## Majority-Vote Utilities

- `apply_ttrl_gt` decodes the `n` rollouts per prompt, performs majority voting, and writes pseudo labels + ratios into `DataProto.non_tensor_batch`.
- `apply_original_gt` reverts pseudo labels so that evaluations against the real ground truth remain available.
- `compute_ttrl_metrics` aggregates:
  - Label accuracy between majority vote and original GT.
  - Reward accuracy (how many sequences keep the same score).
  - Average majority ratio and per-prompt pass@k derived from the recomputed rewards.

These helpers allow you to audit whether pseudo labels drift too much when scaling to large vote counts (`ttrl.n_votes_per_prompt`).

## Debug & Logging Tips

- Enable `trainer.balance_batch=True` if sequence-length imbalance hurts throughput; metrics for token distribution are emitted automatically.
- When GRPO-NSR is active, extra metrics (`training/grpo_nsr_pos_ratio`, `training/grpo_nsr_pos_token_ratio`) confirm whether the correctness threshold is reasonable.
- Use `rollout_data_dir` or `validation_data_dir` in the config to dump decoded prompts/outputs with `token_level_scores.sum(-1)` for manual inspection.

## Citation

All conceptual contributions originate from the original TTRL work. Please cite it when using this repo:

```bibtex
@article{zuo2025ttrl,
  title={Ttrl: Test-time reinforcement learning},
  author={Zuo, Yuxin and Zhang, Kaiyan and Sheng, Li and Qu, Shang and Cui, Ganqu and Zhu, Xuekai and Li, Haozhan and Zhang, Yuchen and Long, Xinwei and Hua, Ermo and others},
  journal={arXiv preprint arXiv:2504.16084},
  year={2025}
}
```
