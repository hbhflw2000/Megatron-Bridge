# Qwen3-Omni Communication Optimization Log

This file is local-only and should remain untracked.

## Scope

Track communication-side profiling results and optimization notes separately from the general training/dev notes.

Primary focus:

- NCCL overhead
- MoE token dispatcher communication/reordering cost
- overlap opportunities
- communication-related idle time

## Baseline Context

Current stable baseline:

- model: Qwen3-Omni thinker-side training
- parallel: `TP=2`, `PP=2`, `CP=1`, `EP=8`, `ETP=1`, `SP=True`
- cluster shape: `4 nodes x 8 GPUs`
- sequence length: `16384`
- global batch size: `16`
- micro batch size: `1`

Observed MFU before communication-specific follow-up:

- baseline estimate: `5.6%` to `5.9%`
- after disabling optimizer offload: about `6.8%` to `7.0%`

## 2026-04-15: 32-GPU PyTorch Profiler Readout

Profiler artifact:

- `/nfs/ofs-llab-hdd/users/liuwei/omni/qwen3_omni_train/results/qwen3_omni_sft32_tp2_pp2_ep8_sp_profile/torch_profile/rank-0.json.gz`

High-level conclusion:

- main bottleneck is communication, not pure math kernels
- secondary hotspot is MoE token dispatch/combine reorder work
- FlashAttention is visible but not the dominant bottleneck

### Key GPU annotation totals

- `nccl:reduce_scatter_tensor_coalesced`: about `6492 ms`
- `nccl:all_gather_into_tensor_coalesced`: about `5958 ms`
- `nccl:coalesced`: about `5368 ms`
- `nccl:all_to_all`: about `3388 ms`
- `nccl:all_reduce`: about `1125 ms`
- `nccl:_all_gather_base`: about `725 ms`
- `nccl:_reduce_scatter_base`: about `97 ms`

Approximate communication total from GPU annotations:

- about `23153 ms`

Two profiled step windows:

- `ProfilerStep#10`: about `23618 ms`
- `ProfilerStep#11`: about `23754 ms`

Combined step-window total:

- about `47372 ms`

Estimated direct communication share:

- about `49%`

Interpretation:

- nearly half of step time is directly visible as NCCL communication
- real communication impact is higher once waiting/synchronization overhead is included

### Key kernel hotspots

- `ncclDevKernel_SendRecv`: about `8756 ms`
- `ncclDevKernel_AllGather_RING_LL`: about `6684 ms`
- `ncclDevKernel_ReduceScatter_Sum_f32_RING_LL`: about `6492 ms`
- `_sort_chunks_by_idxs_kernel`: about `697 ms`
- `_permute_kernel`: about `184 ms`
- `_unpermute_kernel`: about `184 ms`
- flash attention backward kernel: about `1347 ms`
- flash attention forward kernel: about `1014 ms`

Interpretation:

- communication kernels dominate
- MoE dispatcher sorting/reordering is the largest non-NCCL hotspot
- attention optimization is lower priority than communication optimization

### Synchronization / idle evidence

- `cudaStreamSynchronize`: about `15190 ms`
- `cudaDeviceSynchronize`: about `5481 ms`
- `cudaEventSynchronize`: about `4244 ms`

Interpretation:

- GPU compute is not continuously saturated
- there is substantial waiting for communication or dependent work
- timeline inspection in Perfetto/Chrome trace should show compute gaps around communication-heavy regions

## Working Hypothesis

Current priority order:

1. reduce or hide NCCL communication cost
2. reduce MoE token dispatch/combine overhead
3. only then optimize attention/math kernels further

## Profiling-Driven Optimization Checklist

Use this checklist before making another round of performance changes.

### A. Verify forward/backward balance

- measure forward vs backward time within one complete profiled step
- expected rough health range:
  - backward is typically heavier than forward
  - `fwd:bwd ~= 1:1.5` to `1:2` is not surprising
- if backward is much larger than `2x` forward:
  - check recompute overhead first
  - check MoE dispatcher/combine backward cost
  - check whether backward-side communication dominates

What to inspect in trace:

- `ProfilerStep#N` total forward window
- `ProfilerStep#N` total backward window
- backward-side `all_to_all`, `reduce_scatter`, `all_gather`
- MoE reorder kernels during backward:
  - `_sort_chunks_by_idxs_kernel`
  - `_permute_kernel`
  - `_unpermute_kernel`

### B. Check whether communication is intrinsically slow or rank-wait dominated

- compare the same collective across ranks:
  - `all_to_all`
  - `all_gather`
  - `reduce_scatter`
  - `SendRecv`
- determine whether ranks enter the collective at nearly the same time

Interpretation:

- if ranks enter NCCL at almost the same time and the kernel is still long:
  - communication itself is a primary bottleneck
- if some ranks arrive much earlier and then wait:
  - compute imbalance / pipeline skew / MoE skew is the real upstream cause

### C. Check rank synchronization and stage balance

- compare step timelines across representative ranks
- compare pipeline stages separately
- look for:
  - one stage consistently finishing later than the others
  - one rank entering `all_to_all` much later than peers
  - visible idle gaps before collective launch

Likely causes if imbalance is observed:

- MoE token skew across experts / ranks
- multimodal sample-size variance
- pipeline stage load imbalance
- uneven visual/audio tower work before the decoder

### D. Check MoE-specific communication pressure

- inspect token dispatcher / combine path around:
  - `all_to_all`
  - `_sort_chunks_by_idxs_kernel`
  - `_permute_kernel`
  - `_unpermute_kernel`
- estimate whether the dominant issue is:
  - token movement volume
  - token skew
  - dispatcher kernel overhead

Practical follow-up knobs:

- `MOE_FLEX_DISPATCHER_BACKEND`
- alternate `TP/EP` balance after low-risk runtime knobs are exhausted

### E. Keep attention optimization lower priority unless trace disproves current hypothesis

- FlashAttention is visible, but not the first optimization target right now
- only promote attention-side work if:
  - communication share drops materially
  - or updated profiling shows attention overtaking NCCL + MoE reorder work

## Immediate Next Checks

1. quantify `forward` vs `backward` time ratio on one stable profiled step
2. compare collective-entry timing across ranks to identify wait-heavy collectives
3. determine whether current communication time is mostly:
   - real NCCL time
   - or rank skew waiting at communication boundaries
4. if skew is confirmed, inspect:
   - pipeline stage imbalance
   - MoE token skew
   - multimodal sample variance
5. only after that, continue runtime A/B on:
   - `MOE_FLEX_DISPATCHER_BACKEND=deepep`
   - `MOE_FLEX_DISPATCHER_BACKEND=hybridep`

## Candidate Follow-up Directions

### DDP / parameter communication overlap

- validate and keep:
  - `overlap_grad_reduce=True`
  - `overlap_param_gather=True`
  - `align_param_gather=True`
- goal:
  - hide part of gradient / param communication under compute

## 2026-04-15: DDP overlap revisit

Historical A/B results found in local logs:

- overlap off:
  - stable step time about `14.55s` to `14.76s`
  - throughput about `22.1` to `22.2 TFLOP/s/GPU`
  - MFU about `7.0%` to `7.1%`
- overlap on:
  - stable step time about `14.88s` to `15.23s`
  - throughput about `21.2` to `21.7 TFLOP/s/GPU`
  - MFU about `6.8%` to `6.9%`

Conclusion:

- DDP overlap was already tested locally
- for the current Qwen3-Omni 4x8 recipe, overlap is slightly slower
- do not make overlap the default baseline

Evidence:

- `/nfs/ofs-llab-hdd/users/liuwei/omni/qwen3_omni_train/logs/qwen3_omni_sft32_tp2_pp2_ep8_sp_ab_overlap_off_full.log`
- `/nfs/ofs-llab-hdd/users/liuwei/omni/qwen3_omni_train/logs/qwen3_omni_sft32_tp2_pp2_ep8_sp_ab_overlap_on_full.log`

## 2026-04-15: First Local Comm Tuning Action

Launcher updated in local worktree:

- `local_train_thinker_full.sh`

Changes:

- add passthrough knob:
  - `MOE_FLEX_DISPATCHER_BACKEND`

Rationale:

- profiler shows direct NCCL cost is already about half of step time
- historical overlap A/B already shows a small regression, so overlap remains opt-in only
- MoE flex dispatcher should be tested as a runtime A/B knob instead of becoming a recipe default immediately

## 2026-04-17: MoE flex dispatcher A/B on 4x8 baseline

Experiment shape:

- cluster: `4 nodes x 8 GPUs`
- parallel: `TP=2`, `PP=2`, `EP=8`, `ETP=1`, `SP=True`
- sequence length: `16384`
- global batch size: `16`
- micro batch size: `1`
- recompute: `full / uniform / 12`
- optimizer offload: `False`
- overlap knobs: all `False`

Logs:

- `flex_off`:
  - `/nfs/ofs-llab-hdd/users/liuwei/omni/qwen3_omni_train/logs/qwen3_omni_sft32_tp2_pp2_ep8_sp_ab_flex_off_full.log`
- `flex_deepep`:
  - `/nfs/ofs-llab-hdd/users/liuwei/omni/qwen3_omni_train/logs/qwen3_omni_sft32_tp2_pp2_ep8_sp_ab_flex_deepep_full.log`
- `flex_hybridep`:
  - `/nfs/ofs-llab-hdd/users/liuwei/omni/qwen3_omni_train/logs/qwen3_omni_sft32_tp2_pp2_ep8_sp_ab_flex_hybridep_full.log`

Stable-step readout after warmup:

- `flex_off`:
  - `14.30s` to `14.64s`
  - `22.0` to `22.5 TFLOP/s/GPU`
- `flex_deepep`:
  - `14.23s` to `14.59s`
  - `22.1` to `22.7 TFLOP/s/GPU`
- `flex_hybridep`:
  - `14.55s` to `14.84s`
  - `21.7` to `22.2 TFLOP/s/GPU`

Conclusion:

- `deepep` is the only variant with a visible positive signal
- `deepep` is only a small gain over `flex_off`, not a step-function improvement
- `hybridep` does not improve the current recipe and is slightly worse in the stable window
- current preference for follow-up communication work:
  1. `flex_deepep`
  2. `flex_off`
  3. `flex_hybridep`

Best observed point:

- `flex_deepep`: `14.23s`, `22.7 TFLOP/s/GPU`

Interpretation:

- changing dispatcher backend alone is not enough to explain or remove the large communication share
- profiler-driven root cause work is still required

## 2026-04-17: deeper communication follow-up to answer

The profiler already shows that communication is expensive, but the root cause is still not pinned down.

What is known:

- direct NCCL kernels are a major visible cost
- MoE dispatcher reorder work is the largest non-NCCL hotspot
- `deepep` helps slightly, which suggests dispatcher/runtime choice matters
- the gain is too small to claim that dispatcher backend was the dominant bottleneck

What still needs to be answered:

1. is the communication time mostly real collective time?
2. or are ranks arriving at collective boundaries at different times and then waiting?
3. is backward disproportionately heavier than forward because of MoE dispatch/combine and communication?
4. are pipeline stages or experts imbalanced enough to create communication-side idle gaps?

Required next profiling pass:

- compare one full stable step across representative ranks
- measure `forward` vs `backward` wall time inside the same step
- inspect entry-time skew for:
  - `all_to_all`
  - `all_gather`
  - `reduce_scatter`
  - `SendRecv`
- compare pipeline stages to see whether one stage consistently reaches collectives late
- inspect whether MoE reorder kernels cluster before late collectives on only a subset of ranks

Decision rule for next optimization round:

- if ranks enter collectives at almost the same time:
  - treat communication fabric / collective volume as the main target
- if ranks enter collectives at clearly different times:
  - treat upstream compute imbalance as the main target
  - focus on MoE token skew, pipeline skew, and multimodal variance before trying more communication knobs

## 2026-04-17: deeper rank-0 trace readout from `ProfilerStep#10/#11`

Trace used:

- `/nfs/ofs-llab-hdd/users/liuwei/omni/qwen3_omni_train/results/qwen3_omni_sft32_tp2_pp2_ep8_sp_profile/torch_profile/rank-0.json.gz`

Important limitation:

- only `rank-0` trace is available
- this is enough to confirm what is heavy on rank 0
- this is not enough to prove whether collective slowness is caused by cross-rank skew

### What is now confirmed on rank 0

Within the two stable profiled windows:

- `ProfilerStep#10` wall time is about `15.41s`
- `ProfilerStep#11` wall time is about `14.76s`

Selected kernel totals inside those windows:

- `ProfilerStep#10`
  - NCCL kernels: about `11.87s`
  - Flash kernels: about `1.22s`
  - MoE reorder kernels: about `0.56s`
- `ProfilerStep#11`
  - NCCL kernels: about `11.29s`
  - Flash kernels: about `1.21s`
  - MoE reorder kernels: about `0.56s`

Interpretation:

- communication is not only “waiting overhead”
- NCCL kernels themselves already dominate the stable step on rank 0
- MoE reorder kernels are still material, but they are much smaller than direct NCCL kernel time

Largest kernels inside stable steps:

- `ncclDevKernel_SendRecv`: about `4.24s` to `4.51s`
- `ncclDevKernel_AllGather_RING_LL`: about `3.32s` to `3.36s`
- `ncclDevKernel_ReduceScatter_Sum_f32_RING_LL`: about `3.11s` to `3.39s`
- `_sort_chunks_by_idxs_kernel`: about `0.35s`
- flash attention backward main kernel: about `0.67s`
- flash attention forward main kernel: about `0.51s`

### Backward-side evidence

Backward-related CPU hotspots in both stable steps are led by:

- `CheckpointFunctionBackward`: about `3.30s` to `3.36s`
- `IndexPutBackward0`: about `0.62s`
- `_GroupedLinearBackward`: about `0.16s` to `0.18s`
- `_AllToAllBackward`: about `0.04s` to `0.05s`
- `FlashAttnFuncBackward`: about `0.02s`
- `_moe_chunk_sortBackward`: about `0.02s`
- `_moe_permute_mask_mapBackward`: about `0.01s`
- `_moe_unpermute_mask_mapBackward`: about `0.01s`

Interpretation:

- backward is clearly non-trivial and recompute-heavy
- backward-side MoE and communication work is visible
- but from rank-0 alone, this still does not establish an exact `forward:backward` wall-time ratio

### Synchronization evidence

Within the stable step windows, cumulative runtime sync API time is still very large:

- `ProfilerStep#10`: about `12.75s`
- `ProfilerStep#11`: about `12.16s`

Interpretation:

- rank 0 spends substantial time blocked on GPU/communication progress
- however, these runtime sync totals are cumulative and nested, so they should not be added directly to kernel time
- the safe conclusion is:
  - rank 0 is both launching heavy NCCL kernels and spending large amounts of host/runtime time waiting on them

### Updated conclusion

What is now stronger than before:

- direct NCCL kernel cost is definitely a primary bottleneck on rank 0
- this is not just an illusion caused by CPU-side waiting counters
- `SendRecv`, `AllGather`, and `ReduceScatter` are the dominant communication kernels

What is still unresolved:

- whether other ranks arrive late to those same collectives
- whether pipeline-stage imbalance or MoE token skew is amplifying the observed communication cost

Next required profiling improvement:

- capture the same step on multiple representative ranks, not just rank 0
- priority ranks:
  - at least one rank per pipeline stage
  - one or two ranks that carry different expert / device positions

## 2026-04-17: multi-rank profiling result (`ranks=[0,1,16,17]`)

Trace directory:

- `/nfs/ofs-llab-hdd/users/liuwei/omni/qwen3_omni_train/results/qwen3_omni_sft32_tp2_pp2_ep8_sp_profile_multirank/torch_profile`

Profiled ranks:

- `0`, `1`, `16`, `17`

Reason for choosing them:

- cover both `PP` stages
- cover both `TP` lanes inside those stages
- keep profiler overhead moderate while still exposing cross-stage skew

### Step-window consistency

Stable profiled windows are very close across ranks:

- `ProfilerStep#10`: about `15.117s` to `15.119s`
- `ProfilerStep#11`: about `15.142s` to `15.143s`

Interpretation:

- all selected ranks are measuring the same logical step
- the skew findings below are not caused by comparing different steps

### Per-rank kernel mix

Across all four ranks, the stable-step totals are broadly similar:

- NCCL kernels: about `11.66s` to `11.95s`
- Flash kernels: about `1.19s` to `1.22s`
- MoE reorder kernels: about `0.47s` to `0.58s`

Interpretation:

- the heavy communication signature is not rank-0-only
- direct NCCL kernel cost is consistently dominant on every sampled rank

### Collective entry skew

Even though per-rank step time is aligned, the first entry time into major collectives is not.

#### `ProfilerStep#10`

- first `nccl:all_to_all` entry skew:
  - about `6093 ms`
- first `nccl:coalesced` entry skew:
  - about `5683 ms`
- first `nccl:reduce_scatter_tensor_coalesced` entry skew:
  - about `1736 ms`
- first `nccl:all_gather_into_tensor_coalesced` entry skew:
  - about `1773 ms`

#### `ProfilerStep#11`

- first `nccl:all_to_all` entry skew:
  - about `1924 ms`
- first `nccl:coalesced` entry skew:
  - about `973 ms`
- first `nccl:reduce_scatter_tensor_coalesced` entry skew:
  - about `1752 ms`
- first `nccl:all_gather_into_tensor_coalesced` entry skew:
  - about `189 ms`

Interpretation:

- communication is expensive for two reasons at once:
  1. NCCL kernels themselves are heavy
  2. ranks do not arrive at important collective boundaries at the same time
- this means the communication bottleneck is not “pure fabric cost only”
- there is real upstream skew before at least:
  - `all_to_all`
  - `coalesced`
  - `reduce_scatter`

### Stronger root-cause statement

Current best explanation:

- the training step is paying a double penalty
  - high direct collective cost
  - plus cross-rank arrival skew before those collectives

What this rules out:

- the bottleneck is not just a misleading synchronization artifact
- the bottleneck is not just one bad rank with an otherwise healthy collective pattern

What this now points toward:

- upstream imbalance is likely contributing materially
- most likely sources:
  - pipeline-stage skew
  - MoE token / expert skew
  - multimodal variance before decoder-side collectives

Updated optimization implication:

- continuing to tune communication backend alone may yield only incremental gains
- the next meaningful round should inspect where the late-arriving ranks spend time before:
  - `all_to_all`
  - `coalesced`
  - `reduce_scatter`

## 2026-04-17: what the late rank is doing before `all_to_all`

Method:

- for each stable profiled step, identify:
  - earliest rank entering the first `nccl:all_to_all`
  - latest rank entering the first `nccl:all_to_all`
- then inspect the late rank only over the interval:
  - `[ earliest_all_to_all_entry , late_all_to_all_entry ]`

Result:

- in both `ProfilerStep#10` and `ProfilerStep#11`, the latest sampled rank is `rank-1`
- the earliest sampled rank is:
  - `rank-16` for `ProfilerStep#10`
  - `rank-17` for `ProfilerStep#11`

### `ProfilerStep#10`

- `all_to_all` entry gap:
  - about `6093 ms`
- during that late-rank gap on `rank-1`, the dominant work is still backward / recompute-heavy work

Top signals on the late rank:

- `CheckpointFunctionBackward`: about `2985 ms`
- `IndexPutBackward0` and related index-put / copy-slices work: about `380 ms`
- `_GroupedLinearBackward`: about `149 ms`
- flash backward main kernel: about `562 ms`
- flash forward kernel: about `285 ms`
- `_sort_chunks_by_idxs_kernel`: about `174 ms`
- `_permute_kernel` + `_unpermute_kernel`: about `122 ms`
- `ncclDevKernel_SendRecv`: about `4041 ms`

Interpretation:

- the late rank is not simply idle before `all_to_all`
- it is still actively executing backward / recompute work, plus MoE reorder work, plus smaller NCCL work
- this strongly suggests the late arrival is caused by upstream compute progress, not just scheduler noise

### `ProfilerStep#11`

- `all_to_all` entry gap:
  - about `1924 ms`
- the same pattern remains, but the skew is smaller

Top signals on the late rank:

- `CheckpointFunctionBackward`: about `1706 ms`
- index-put / copy-slices family: about `331 ms`
- `_GroupedLinearBackward`: about `76 ms`
- flash backward main kernel: about `323 ms`
- flash forward kernel: about `127 ms`
- `_sort_chunks_by_idxs_kernel`: about `70 ms`
- `_permute_kernel` + `_unpermute_kernel`: about `61 ms`
- `ncclDevKernel_SendRecv`: about `879 ms`

Interpretation:

- even in the smaller-skew step, the late rank is still busy in backward/recompute-heavy work before joining `all_to_all`
- the skew is therefore repeatable, not a one-off anomaly

### Stronger practical conclusion

Current best explanation is now:

- communication is genuinely expensive
- and some ranks, especially the late side represented by `rank-1`, are still inside backward / recompute-heavy work when earlier ranks have already reached `all_to_all`

What this suggests operationally:

- the next bottleneck investigation should focus less on “which NCCL backend is best”
- and more on why this side of the model reaches MoE / communication boundaries later

Most likely candidates to inspect next:

- pipeline-stage imbalance
- backward/recompute imbalance across stages
- MoE token/expert skew feeding into the late stage

## 2026-04-17: early-rank vs late-rank comparison before first `all_to_all`

Method:

- for each stable step:
  - pick the earliest sampled rank entering the first `all_to_all`
  - pick the latest sampled rank entering the first `all_to_all`
- compare their own pre-`all_to_all` windows instead of only looking at the late-rank gap

### `ProfilerStep#10`

- earliest sampled rank: `rank-16`
  - first `all_to_all` at about `2022 ms`
- latest sampled rank: `rank-1`
  - first `all_to_all` at about `8115 ms`

Pre-`all_to_all` window summary:

- `rank-16`
  - checkpoint-heavy CPU work exists, but is much smaller
  - `CheckpointFunction`: about `478 ms`
  - `_AllToAll`: about `13 ms`
  - `_moe_chunk_sort` / `_moe_permute_mask_map`: single-digit milliseconds
- `rank-1`
  - clearly heavier backward/recompute path before first `all_to_all`
  - `CheckpointFunctionBackward`: about `2092 ms`
  - `CheckpointFunction`: about `1714 ms`
  - index-put / copy-slices family: about `312 ms`
  - `_GroupedLinearBackward`: about `149 ms`
  - `_AllToAll`: about `107 ms`

Interpretation:

- the late side is not just delayed by communication
- it does substantially more backward / recompute-heavy CPU-side work before the first `all_to_all`

### `ProfilerStep#11`

- earliest sampled rank: `rank-17`
  - first `all_to_all` at about `6070 ms`
- latest sampled rank: `rank-1`
  - first `all_to_all` at about `7994 ms`

Pre-`all_to_all` window summary:

- `rank-17`
  - `CheckpointFunctionBackward`: about `2748 ms`
  - `CheckpointFunction`: about `1489 ms`
  - `_GroupedLinearBackward`: about `146 ms`
  - `_AllToAll`: about `128 ms`
  - very little index-put family work
- `rank-1`
  - `CheckpointFunctionBackward`: about `3341 ms`
  - `CheckpointFunction`: about `1622 ms`
  - index-put / copy-slices family: about `640 ms`
  - `_GroupedLinearBackward`: about `150 ms`
  - `_AllToAll`: about `113 ms`

Interpretation:

- both sides do heavy checkpoint/backward work
- but the late side (`rank-1`) still carries noticeably more checkpoint-backward time
- the most obvious extra burden on the late side is the index-put / copy-slices family

### Stronger current hypothesis

This looks increasingly like a stage-side imbalance rather than a pure communication-backend issue.

What stands out:

- `rank-1` repeatedly arrives late
- the late arrival correlates with:
  - more `CheckpointFunctionBackward`
  - more index-put / copy-slices work
  - similar or slightly heavier grouped-linear / MoE-related backward work

Practical reading:

- the sampled late side appears to be spending more time in backward/recompute-heavy logic before it can join the first major `all_to_all`
- this is consistent with a pipeline-side imbalance or a stage-specific computation hotspot
- communication tuning alone is unlikely to remove this asymmetry

Recommended next investigation:

- identify which exact module path is responsible for the extra:
  - `CheckpointFunctionBackward`
  - `IndexPutBackward0`
  - `torch::autograd::CopySlices`
- then decide whether the next optimization target is:
  - reducing recompute on the late side
  - reducing stage-specific tensor/index operations
  - or rebalancing stage work

## 2026-04-17: likely code-path mapping for the late-rank index/copy hotspot

Most plausible source in current Qwen3-Omni thinker code:

- `src/megatron/bridge/models/qwen_omni/modeling_qwen3_omni/thinker_model.py`

Candidate hotspots:

- image feature insertion:
  - `inputs_embeds_bsh.masked_scatter_(...)`
  - around lines `432`
- video feature insertion:
  - `inputs_embeds_bsh.masked_scatter_(...)`
  - around lines `437`
- deepstack joint visual embedding construction:
  - `embed_joint[image_mask_joint] = image_embed`
  - `embed_joint[video_mask_joint] = video_embed`
  - around lines `450-452`
- audio feature insertion:
  - `combined_embeddings_bsh.masked_scatter_(...)`
  - around lines `477-480`

Why these are strong candidates:

- the late rank shows repeated `IndexPutBackward0`, `torch::autograd::CopySlices`, and related index-put family work
- these operations are consistent with in-place masked writes and indexed tensor assignment
- the sampled early rank does not show the same level of index/copy cost before first `all_to_all`
- these multimodal embedding insertion paths only exist on the `pre_process` side of the thinker forward

Important contrast:

- `rope.py` also contains indexed assignment:
  - `position_ids[..., i, attention_mask[i]] = llm_positions`
  - around line `243`
- however, this path writes into position ids and is much less likely to explain the observed backward-side `IndexPutBackward0` hotspot
- current evidence points much more strongly to multimodal embedding insertion / merge logic in `thinker_model.py`

Working interpretation:

- the sampled late side likely corresponds to the pipeline side that owns multimodal pre-processing
- that side pays extra tensor-write and merge overhead before entering major collectives
- this likely contributes to the stage-side imbalance seen in the multi-rank trace

## 2026-04-17: `masked_scatter_` itself is not the main backward hotspot

Additional trace evidence across sampled ranks:

- `rank-0` and `rank-1` both show:
  - `torch::autograd::CopySlices`: about `1290 ms`
  - `IndexPutBackward0`: about `1285 ms`
  - `aten::masked_scatter_`: about `1 ms`
  - `MaskedScatterBackward0`: below `1 ms`
- `rank-16` and `rank-17` do not show the same `CopySlices` / `IndexPutBackward0` pattern

Interpretation:

- `masked_scatter_` is present, but its own backward cost is tiny
- the real backward hotspot is the index-write / copy-slices family
- this makes the direct indexed assignments in the multimodal merge path the stronger suspect than `masked_scatter_` itself

Most likely culprit inside current code:

- `embed_joint[image_mask_joint] = image_embed`
- `embed_joint[video_mask_joint] = video_embed`

Why this matters:

- the multimodal insertion path likely has two distinct cost types:
  1. feature insertion via `masked_scatter_`
  2. deepstack/joint visual merge via explicit indexed writes
- current trace evidence says the second category is much more likely to explain the late-rank backward hotspot

Stronger stage-side interpretation:

- only the sampled ranks that appear to own `pre_process` show the `CopySlices` / `IndexPutBackward0` burden
- sampled ranks from the other side do not
- this is strong evidence that the first pipeline stage is carrying extra multimodal merge work before major collectives

## 2026-04-17: narrowed suspect after checking dataset modality mix

Dataset observation for the current local training set:

- every sampled training example is `image + audio`
- no sampled training example contains `video`

Implication:

- the image+video joint merge branch in `thinker_model.py`
  - `embed_joint[image_mask_joint] = image_embed`
  - `embed_joint[video_mask_joint] = video_embed`
  is not expected to execute for this current dataset

This rules out the earlier “mixed image+video branch” as the primary explanation for the current trace.

### More likely active culprit

The stronger remaining suspect is the deepstack visual path inherited from the Qwen3-VL text/decoder stack.

Relevant code:

- `src/megatron/bridge/models/qwen_vl/modelling_qwen3_vl/transformer_block.py`
  - `_deepstack_process(...)`
  - `hidden_states[visual_pos_masks, :].clone() + visual_embeds`
  - `hidden_states[visual_pos_masks, :] = local_this`
- call sites:
  - only when `self.pre_process and deepstack_visual_embeds is not None`
  - around lines `519-526`
  - around lines `735-745`

Why this is now the best match:

- current dataset definitely exercises image path
- image path in `thinker_model.py` populates `deepstack_visual_embeds`
- `deepstack_visual_embeds` is then consumed only on the first PP stage
- `_deepstack_process` performs explicit mask-based indexed read/write on `hidden_states`
- this operation shape is highly consistent with:
  - `torch::autograd::CopySlices`
  - `IndexPutBackward0`
  - `aten::index_put_*`

Updated working diagnosis:

- the dominant first-stage asymmetry is now most likely coming from image-side deepstack visual injection in the language transformer
- audio is present in the dataset, but the strongest current trace signature still looks like visual-mask-based indexed writeback

Most likely first optimization target, if we move from diagnosis to implementation:

- reduce or restructure `_deepstack_process` indexed writeback on the first PP stage

Planned experiment order:

1. current baseline recipe with overlap left off
2. baseline + `MOE_FLEX_DISPATCHER_BACKEND=deepep`
3. only if needed, test `MOE_FLEX_DISPATCHER_BACKEND=hybridep`

Success criteria:

- step time reduction without instability
- TFLOP/s/GPU increase
- MFU moves toward `8%+`

### MoE communication path

- investigate token dispatcher / combine path
- profile sensitivity to:
  - expert parallel degree
  - tensor parallel degree
  - token distribution skew
- goal:
  - reduce `all_to_all` and sort/unpermute pressure

### Parallel-shape exploration

- compare current `TP=2 / PP=2 / EP=8`
- possible later candidate:
  - `TP=4 / PP=2 / EP=4`
- only test after current low-risk runtime knobs plateau

## Notes

- Communication optimization notes belong in this file.
- General training bring-up, correctness, and recipe notes belong in:
  - `examples/models/vlm/qwen3_omni/local_training_notes.md`
  - `examples/models/vlm/qwen3_omni/local_optimization_plan.md`

## 2026-04-17: candidate optimization design after first-stage deepstack diagnosis

### What we now believe

- the strongest current imbalance is not the generic communication backend alone
- the most likely stage-local hotspot is first-PP-stage deepstack visual injection inside:
  - `src/megatron/bridge/models/qwen_vl/modelling_qwen3_vl/transformer_block.py`
  - `_deepstack_process(...)`
- the trace signature matches:
  - `torch::autograd::CopySlices`
  - `IndexPutBackward0`
- it does **not** match `masked_scatter_` as the dominant cost
- current local dataset is `image + audio` without `video`, so the image+video joint merge branch is not the main suspect for this run

### Scope of the hotspot

- `deepstack_visual_embeds` length is `3`
- language-side deepstack insertion is therefore only applied to the first `3` language layers on the first PP stage
- this is good news:
  - the hotspot is narrow
  - a targeted code-path optimization is more attractive than broad communication tuning

### Recommended option order

1. `P1`: rewrite `_deepstack_process` to avoid boolean advanced index read/write
2. `P2`: if needed, try PP rebalance after making uneven-layer recompute safe
3. `P3`: only if needed, reduce replay cost of deepstack insertion under recompute

### `P1` Recommended: rewrite `_deepstack_process`

Current pattern:

- `hidden_states[visual_pos_masks, :].clone() + visual_embeds`
- `hidden_states[visual_pos_masks, :] = local_this`

Why this is the best first target:

- this pattern matches the observed `CopySlices + IndexPutBackward0` hotspot
- it is isolated to the first PP stage
- it only runs for the first `3` language layers
- it does not require changing distributed topology

Design direction:

- precompute flat visual token indices once from `visual_pos_masks`
- replace boolean advanced indexing writeback with additive scatter on flattened hidden states
- the most promising formulation is:
  - flatten `[b, s, h] -> [b*s, h]`
  - use stable token indices from the mask
  - apply `index_add_` / equivalent additive scatter with `visual_embeds`

Why this is attractive:

- we already saw `masked_scatter_` itself is cheap
- the expensive part is the read-modify-write boolean indexing family
- additive scatter should reduce:
  - `CopySlices`
  - `IndexPutBackward0`
  - repeated boolean indexing overhead

Main risks:

- must verify no duplicate indices
- must confirm numerical equivalence to the current “add onto selected positions” semantics
- may still introduce a different backward kernel profile, so it needs measurement

Expected validation path:

1. short correctness/smoke run
2. 1-step or 2-step profiler rerun on representative ranks
3. 30-step A/B against current best baseline

### `P2` Secondary: PP rebalance, but only after making uneven split safe

Why this is still interesting:

- first PP stage is clearly heavier than the later sampled stage
- moving one or more decoder layers off stage 0 is a plausible mitigation

Likely mechanism:

- use explicit uneven PP split via:
  - `decoder_first_pipeline_num_layers`
  - `decoder_last_pipeline_num_layers`

Important constraint:

- current model has `48` decoder layers with `PP=2`
- `account_for_embedding_in_pipeline_split=True` alone is not a clean fit here
- explicit first/last stage layer counts are the cleaner direction

Important hidden risk discovered during diagnosis:

- core Megatron `transformer_block.py` clamps the end of uniform recompute chunks
- the custom Qwen-VL `transformer_block.py` uniform recompute path currently does **not**
- this means uneven PP split is **not** a pure knob-only experiment today when using:
  - `recompute_granularity=full`
  - `recompute_method=uniform`

Implication:

- before trying uneven PP split, we likely need a small safety patch so the custom uniform-recompute loop clamps chunk end like core Megatron

Candidate PP experiments after that patch:

- conservative:
  - first stage `23`, last stage `25`
- more aggressive:
  - first stage `22`, last stage `26`

This option is promising, but it is **not** the fastest first move because it couples:

- parallel layout changes
- recompute loop safety
- performance effects from both stage rebalance and checkpoint chunking

### `P3` Tertiary: reduce replay cost under recompute

Why this is a later option:

- the first hotspot path sits inside the checkpointed forward region
- backward replay therefore re-executes the deepstack insertion path

Possible direction:

- move deepstack insertion out of the checkpointed region where feasible
- or reduce per-replay overhead by caching / reusing precomputed visual indices

Why it is lower priority than `P1`:

- it is more invasive
- it interacts with activation memory behavior
- we can likely learn a lot first by only rewriting the indexed writeback path

### Current recommendation

- start with `P1`
- do **not** keep exploring communication backend knobs first
- keep `P2` as the next structural option if `P1` only gives a small gain
