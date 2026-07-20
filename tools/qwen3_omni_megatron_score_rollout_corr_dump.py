#!/usr/bin/env python3
"""Score verl rollout-corr samples with the Megatron Qwen3-Omni bridge."""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import socket
from pathlib import Path
from typing import Iterable

import torch
import torch.distributed as dist
from megatron.core import parallel_state
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnBackend

from megatron.bridge.models.conversion.auto_bridge import AutoBridge


def _emit(record: dict, output_file: str | None) -> None:
    line = json.dumps(record, ensure_ascii=False, sort_keys=True)
    print(line, flush=True)
    if output_file:
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def _load_records(path: str, limit: int) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("event") and record.get("event") != "rollout_corr_sample":
                continue
            if "input_ids" not in record or "responses" not in record:
                continue
            records.append(record)
            if limit > 0 and len(records) >= limit:
                break
    if not records:
        raise ValueError(f"No rollout_corr_sample records found in {path}")
    return records


def _finite_pairs(left: list[float], right: list[float], mask: list[int]) -> list[tuple[float, float]]:
    pairs = []
    for a, b, keep in zip(left, right, mask):
        if not keep:
            continue
        a = float(a)
        b = float(b)
        if math.isfinite(a) and math.isfinite(b):
            pairs.append((a, b))
    return pairs


def _corr(pairs: list[tuple[float, float]]) -> float | None:
    if len(pairs) < 2:
        return None
    left = torch.tensor([x for x, _ in pairs], dtype=torch.float32)
    right = torch.tensor([y for _, y in pairs], dtype=torch.float32)
    if left.std() == 0 or right.std() == 0:
        return None
    return float(torch.corrcoef(torch.stack([left, right]))[0, 1].item())


def _stats(values: list[float], mask: list[int]) -> dict:
    kept = [float(value) for value, keep in zip(values, mask) if keep and math.isfinite(float(value))]
    if not kept:
        return {"count": 0, "mean": None, "min": None, "max": None}
    return {"count": len(kept), "mean": sum(kept) / len(kept), "min": min(kept), "max": max(kept)}


def _compare(left_name: str, right_name: str, left: list[float], right: list[float], mask: list[int]) -> dict:
    pairs = _finite_pairs(left, right, mask)
    prefix = f"{left_name}_vs_{right_name}"
    if not pairs:
        return {
            f"{prefix}/paired_count": 0,
            f"{prefix}/abs_diff_mean": None,
            f"{prefix}/abs_diff_max": None,
            f"{prefix}/signed_diff_mean": None,
            f"{prefix}/corr": None,
        }
    diffs = [a - b for a, b in pairs]
    abs_diffs = [abs(x) for x in diffs]
    return {
        f"{prefix}/paired_count": len(pairs),
        f"{prefix}/abs_diff_mean": sum(abs_diffs) / len(abs_diffs),
        f"{prefix}/abs_diff_max": max(abs_diffs),
        f"{prefix}/signed_diff_mean": sum(diffs) / len(diffs),
        f"{prefix}/corr": _corr(pairs),
    }


def _find_free_port() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return str(sock.getsockname()[1])


def _init_dist() -> None:
    if dist.is_initialized():
        return
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", _find_free_port())
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    dist.init_process_group(
        backend="nccl" if torch.cuda.is_available() else "gloo",
        world_size=1,
        rank=0,
        timeout=datetime.timedelta(minutes=30),
    )


def _init_model_parallel(seed: int) -> None:
    if parallel_state.model_parallel_is_initialized():
        parallel_state.destroy_model_parallel()
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        virtual_pipeline_model_parallel_size=None,
        context_parallel_size=1,
    )
    if torch.cuda.is_available():
        torch.cuda.set_device(int(os.getenv("LOCAL_RANK", "0")))
        model_parallel_cuda_manual_seed(seed)
    else:
        torch.manual_seed(seed)


def _attention_backend(value: str):
    value = value.strip().lower()
    if value in {"", "keep"}:
        return None
    return getattr(AttnBackend, value)


def _build_model(args: argparse.Namespace):
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    bridge = AutoBridge.from_hf_pretrained(args.model_path, dtype=dtype)
    provider = bridge.to_megatron_provider(load_weights=True)
    provider.tensor_model_parallel_size = 1
    provider.pipeline_model_parallel_size = 1
    provider.pipeline_dtype = dtype
    provider.params_dtype = dtype
    provider.sequence_parallel = False
    if args.moe_grouped_gemm != "keep":
        provider.moe_grouped_gemm = args.moe_grouped_gemm == "true"
    if args.rotary_interleaved != "keep":
        provider.rotary_interleaved = args.rotary_interleaved == "true"
    backend = _attention_backend(args.attention_backend)
    if backend is not None:
        provider.attention_backend = backend
    provider.finalize()
    model = provider.provide_distributed_model(wrap_with_ddp=False)
    if isinstance(model, list):
        model = model[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    return model, device


def _record_tensors(record: dict, mode: str, device: torch.device) -> tuple[torch.Tensor, torch.Tensor | None, int]:
    input_ids = torch.tensor(record["input_ids"], dtype=torch.long)
    attention_mask = torch.tensor(record["attention_mask"], dtype=torch.long)
    response_len = len(record["responses"])
    position_ids = None

    if mode == "unpadded_none":
        keep = attention_mask.bool()
        input_ids = input_ids[keep].unsqueeze(0)
        response_start = input_ids.size(1) - response_len
    elif mode == "padded_none":
        input_ids = input_ids.unsqueeze(0)
        response_start = input_ids.size(1) - response_len
    elif mode == "padded_dump":
        input_ids = input_ids.unsqueeze(0)
        raw_position_ids = torch.tensor(record["position_ids"], dtype=torch.long)
        if raw_position_ids.ndim == 1:
            position_ids = raw_position_ids.unsqueeze(0)
        elif raw_position_ids.ndim == 2:
            position_ids = raw_position_ids
        else:
            raise ValueError(f"Unsupported position_ids ndim={raw_position_ids.ndim}")
        response_start = input_ids.size(1) - response_len
    elif mode == "unpadded_dump":
        keep = attention_mask.bool()
        input_ids = input_ids[keep].unsqueeze(0)
        raw_position_ids = torch.tensor(record["position_ids"], dtype=torch.long)
        if raw_position_ids.ndim == 1:
            position_ids = raw_position_ids[keep].unsqueeze(0)
        elif raw_position_ids.ndim == 2:
            position_ids = raw_position_ids[:, keep]
        else:
            raise ValueError(f"Unsupported position_ids ndim={raw_position_ids.ndim}")
        response_start = input_ids.size(1) - response_len
    else:
        raise ValueError(f"Unsupported input mode: {mode}")

    if response_start <= 0:
        raise ValueError(f"Invalid response_start={response_start} for row={record.get('row')}")
    input_ids = input_ids.to(device)
    if position_ids is not None:
        position_ids = position_ids.to(device)
    return input_ids, position_ids, response_start


def _score_record(
    model,
    device: torch.device,
    record: dict,
    mode: str,
    logit_offset: int,
) -> tuple[list[float], int]:
    input_ids, position_ids, response_start = _record_tensors(record, mode, device)
    response_len = len(record["responses"])
    with torch.inference_mode():
        logits = model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=None,
            labels=None,
            runtime_gather_output=True,
        ).float()
        target_ids = input_ids[0, response_start : response_start + response_len]
        logit_start = response_start + logit_offset
        logit_end = logit_start + response_len
        if logit_start < 0 or logit_end > logits.size(1):
            raise ValueError(
                f"Invalid logit slice [{logit_start}:{logit_end}] for logits seq_len={logits.size(1)}"
            )
        pred_logits = logits[0, logit_start:logit_end, :]
        log_probs = pred_logits.log_softmax(dim=-1).gather(1, target_ids.unsqueeze(1)).squeeze(1)
    return [float(x) for x in log_probs.detach().cpu().tolist()], response_start


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump-jsonl", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--record-limit", type=int, default=4)
    parser.add_argument("--sample-limit", type=int, default=16)
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--attention-backend", choices=["keep", "auto", "flash", "fused", "unfused", "local"], default="unfused")
    parser.add_argument("--moe-grouped-gemm", choices=["keep", "true", "false"], default="keep")
    parser.add_argument("--rotary-interleaved", choices=["keep", "true", "false"], default="keep")
    parser.add_argument(
        "--input-mode",
        action="append",
        choices=["unpadded_none", "unpadded_dump", "padded_none", "padded_dump"],
        default=None,
        help="Can be repeated. Defaults to unpadded_none.",
    )
    parser.add_argument(
        "--logit-offset",
        action="append",
        type=int,
        default=None,
        help="Logit start relative to response_start. Defaults to -1, the standard next-token shift.",
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output-file")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    input_modes = args.input_mode or ["unpadded_none"]
    logit_offsets = args.logit_offset or [-1]
    if args.output_file:
        Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_file).write_text("", encoding="utf-8")

    _init_dist()
    _init_model_parallel(args.seed)
    model, device = _build_model(args)
    records = _load_records(args.dump_jsonl, args.record_limit)
    _emit(
        {
            "event": "megatron_rollout_corr_score_start",
            "dump_jsonl": args.dump_jsonl,
            "model_path": args.model_path,
            "record_count": len(records),
            "input_modes": input_modes,
            "logit_offsets": logit_offsets,
            "attention_backend": args.attention_backend,
            "moe_grouped_gemm": args.moe_grouped_gemm,
            "rotary_interleaved": args.rotary_interleaved,
            "dtype": args.dtype,
        },
        args.output_file,
    )

    try:
        for record in records:
            response_mask = [int(x) for x in record["response_mask"]]
            rollout = [float(x) for x in record["rollout_log_probs"]]
            actor_old = [float(x) for x in record["actor_old_log_probs"]]
            ref = [float(x) for x in record.get("ref_log_probs", [])]
            for mode in input_modes:
                for logit_offset in logit_offsets:
                    megatron_logprobs, response_start = _score_record(model, device, record, mode, logit_offset)
                    result = {
                        "event": "megatron_rollout_corr_score_row",
                        "row": record.get("row"),
                        "input_mode": mode,
                        "logit_offset": logit_offset,
                        "response_start": response_start,
                        "response_len": len(record["responses"]),
                        "valid_tokens": sum(response_mask),
                        "megatron_stats": _stats(megatron_logprobs, response_mask),
                        "rollout_stats": _stats(rollout, response_mask),
                        "actor_old_stats": _stats(actor_old, response_mask),
                        "megatron_vs_rollout": _compare("megatron", "rollout", megatron_logprobs, rollout, response_mask),
                        "megatron_vs_actor_old": _compare(
                            "megatron", "actor_old", megatron_logprobs, actor_old, response_mask
                        ),
                        "sample": [
                            {
                                "i": i,
                                "token_id": record["responses"][i],
                                "mask": response_mask[i],
                                "megatron": round(megatron_logprobs[i], 6),
                                "rollout": round(rollout[i], 6),
                                "actor_old": round(actor_old[i], 6),
                                **({"ref": round(ref[i], 6)} if ref else {}),
                            }
                            for i in range(min(args.sample_limit, len(record["responses"])))
                        ],
                    }
                    if ref:
                        result["ref_stats"] = _stats(ref, response_mask)
                        result["megatron_vs_ref"] = _compare("megatron", "ref", megatron_logprobs, ref, response_mask)
                    _emit(result, args.output_file)
        _emit({"event": "megatron_rollout_corr_score_done"}, args.output_file)
    finally:
        if parallel_state.model_parallel_is_initialized():
            parallel_state.destroy_model_parallel()
        if dist.is_initialized():
            dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
