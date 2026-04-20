# Qwen3-Omni Communication Optimization Dev Plan

This file is local-only and should remain untracked.

## Goal

Improve 32-GPU Qwen3-Omni thinker training efficiency, with communication optimization as the primary focus.

Current baseline:

- repo: `/home/luban/liuwei/omni/megatron-bridge/Megatron-Bridge-omni3-train`
- branch: `omni3-train`
- parallel: `TP=2`, `PP=2`, `CP=1`, `EP=8`, `ETP=1`, `SP=True`
- cluster: `4 nodes x 8 GPUs`
- sequence length: `16384`
- current MFU after the first round of tuning: about `6.8%` to `7.0%`

## Working Principle

Start from the main local worktree and optimize code / launch / runtime behavior first.

Default assumption:

- do not modify environment packages initially
- do not rebuild NCCL initially
- do not introduce custom system libraries until the current bottleneck analysis justifies it

Reason:

- current profile already shows large optimization headroom in communication overlap and MoE dispatch path
- these changes are cheaper, safer, and easier to validate than swapping low-level comm libraries too early

## Current Recommendation

Short answer:

- yes, we should start by modifying only the main local worktree
- no environment package changes are required for the first phase

Main worktree:

- `/home/luban/liuwei/omni/megatron-bridge/Megatron-Bridge-omni3-train`

## When To Consider NCCL / Environment Changes

Only escalate to custom NCCL or environment-level changes if at least one of the following becomes true:

1. overlap tuning and MoE-path tuning plateau, but communication still dominates step time
2. profiler keeps showing NCCL kernels as the main bottleneck after software-level improvements
3. we want to test a specific NCCL build or plugin with a concrete hypothesis

If that happens:

- prepare a separate local NCCL directory
- keep the experiment isolated
- switch by launcher env vars rather than mutating the base environment

Preferred approach if we need it later:

- copy custom NCCL to a dedicated directory
- point `LD_LIBRARY_PATH` and related NCCL env vars at it from the training launcher
- compare against the current baseline with one controlled A/B at a time

## Phase 1: No Env Changes

Focus on:

- communication overlap
- MoE dispatcher / combine behavior
- launch/runtime settings that reduce communication-visible idle time

Candidate items:

- keep validating:
  - `overlap_grad_reduce=True`
  - `overlap_param_gather=True`
  - `align_param_gather=True`
- inspect whether the current MoE token routing / dispatch path can be made cheaper through runtime knobs or shape changes
- keep profile-based comparisons in the same launcher family

## Phase 2: Escalate Only If Needed

Possible later experiments:

- custom NCCL build
- NCCL plugin tuning
- network / topology specific env tuning
- alternate comm library versions

These should be treated as separate experiments, not part of the default baseline.

## What To Do Next

1. Continue from the main local worktree.
2. Record all communication-side findings in:
   - `examples/models/vlm/qwen3_omni/local_comm_optimization_log.md`
3. Keep general development notes in:
   - `examples/models/vlm/qwen3_omni/local_training_notes.md`
4. If we later decide to test custom NCCL:
   - create a separate local directory for it
   - update the launcher to point NCCL there
   - run a controlled A/B against the current baseline
