"""Opt-in, response-token attention-stage audit for Megatron/HF parity runs."""

import os
from typing import Any

import torch
from torch import Tensor
from megatron.core import parallel_state as mpu


_STATE: dict[str, Any] | None = None


def _enabled_layers() -> set[int]:
    raw = os.getenv("VERL_OMNI_MEGATRON_ATTENTION_AUDIT_LAYERS", "1")
    return {int(value) for value in raw.split(",") if value.strip().isdigit()}


def configure_attention_audit(rows: list[dict], input_ids: torch.Tensor, padded_length: int) -> None:
    """Enable a single forward's attention-stage capture from the caller."""
    global _STATE
    _STATE = {
        "rows": rows,
        "input_ids": input_ids,
        "padded_length": padded_length,
        "layers": _enabled_layers(),
        "records": [],
        "execution": [],
    }


def clear_attention_audit() -> list[dict]:
    """Return and clear the active capture, including on forward failures."""
    global _STATE
    records = [] if _STATE is None else _STATE["records"]
    _STATE = None
    return records


def attention_execution_audit() -> list[dict]:
    """Return attention-path metadata for the active forward without clearing it."""
    return [] if _STATE is None else _STATE["execution"]


def is_all_valid_2d_mask(mask: torch.Tensor | None) -> bool:
    """Whether a Qwen BSHD valid-token mask carries no padding information."""
    return bool(
        isinstance(mask, torch.Tensor)
        and mask.ndim == 2
        and mask.dtype == torch.bool
        and mask.numel()
        and mask.detach().all().item()
    )


def qwen_valid_mask_to_te_mask(mask: torch.Tensor | None) -> torch.Tensor | None:
    """Convert Qwen's [B, S] True=valid convention to TE's True=masked convention."""
    if not isinstance(mask, torch.Tensor) or mask.ndim != 2 or mask.dtype != torch.bool:
        return mask
    if is_all_valid_2d_mask(mask):
        return None
    return (~mask).unsqueeze(1).unsqueeze(1)


def _sequence_axis(tensor: torch.Tensor, batch_size: int, padded_length: int) -> tuple[int, int] | None:
    if tensor.ndim < 3:
        return None
    if tensor.shape[0] == batch_size:
        axis, local_length = 1, int(tensor.shape[1])
    elif tensor.shape[1] == batch_size:
        axis, local_length = 0, int(tensor.shape[0])
    else:
        return None
    tp_world = mpu.get_tensor_model_parallel_world_size()
    if local_length == padded_length:
        return axis, 0
    if local_length * tp_world == padded_length:
        return axis, mpu.get_tensor_model_parallel_rank() * local_length
    return None


def _stats(vector: torch.Tensor) -> dict:
    vector = vector.detach().float().reshape(-1)
    width = min(8, vector.numel())
    return {
        "sum": float(vector.sum().cpu().item()),
        "square_sum": float((vector * vector).sum().cpu().item()),
        "head": [float(value) for value in vector[:width].cpu().tolist()],
    }


def capture_attention_stage(layer: int, stage: str, tensor: torch.Tensor | None) -> None:
    """Capture response positions for a tensor with BSH... or SBH... layout."""
    state = _STATE
    if state is None or layer not in state["layers"] or not isinstance(tensor, torch.Tensor):
        return
    layout = _sequence_axis(tensor, int(state["input_ids"].shape[0]), int(state["padded_length"]))
    if layout is None:
        return
    sequence_axis, offset = layout
    local_length = int(tensor.shape[sequence_axis])
    for audit_row in state["rows"]:
        row = audit_row["row"]
        for response_index, model_position in enumerate(audit_row["positions"]):
            local_position = model_position - offset
            if not 0 <= local_position < local_length:
                continue
            vector = tensor[row, local_position] if sequence_axis == 1 else tensor[local_position, row]
            state["records"].append(
                {
                    "layer": layer,
                    "stage": stage,
                    "tp_rank": mpu.get_tensor_model_parallel_rank(),
                    "response_index": response_index,
                    "model_position": model_position,
                    "input_ids_sha256": audit_row["input_ids_sha256"],
                    "input_token_id": int(state["input_ids"][row, model_position].item()),
                    "shape": list(tensor.shape),
                    "stats": _stats(vector),
                }
            )


def _mask_summary(mask: torch.Tensor | None, row: int, position: int) -> dict[str, Any]:
    if not isinstance(mask, torch.Tensor):
        return {"present": False}
    result: dict[str, Any] = {"present": True, "shape": list(mask.shape), "dtype": str(mask.dtype)}
    if mask.ndim == 2 and row < mask.shape[0]:
        valid = mask[row].detach().bool().reshape(-1)
        result.update(
            {
                "true_count": int(valid.sum().item()),
                "false_count": int((~valid).sum().item()),
                "all_true": bool(valid.all().item()),
            }
        )
        return result
    if mask.ndim < 3 or row >= mask.shape[0] or position >= mask.shape[-2]:
        return result
    query_mask = mask[row, 0, position].detach().bool().reshape(-1)
    result.update(
        {
            "query_masked_count": int(query_mask.sum().item()),
            "query_unmasked_count": int((~query_mask).sum().item()),
            "self_masked": bool(query_mask[position].item()) if position < query_mask.numel() else None,
        }
    )
    return result


def _tensor_layout(tensor: torch.Tensor | None) -> dict[str, Any]:
    if not isinstance(tensor, torch.Tensor):
        return {"present": False}
    storage = tensor.untyped_storage()
    return {
        "present": True,
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "dtype": str(tensor.dtype),
        "is_contiguous": tensor.is_contiguous(),
        "storage_offset": tensor.storage_offset(),
        "storage_data_ptr": storage.data_ptr(),
        "storage_nbytes": storage.nbytes(),
        "requires_grad": tensor.requires_grad,
        "is_leaf": tensor.is_leaf,
    }


def _enum_or_value(value: Any) -> Any:
    return getattr(value, "name", value)


def _core_attention_runtime(core_attention: Any) -> dict[str, Any]:
    config = getattr(core_attention, "config", None)
    config_fields = (
        "tensor_model_parallel_size",
        "sequence_parallel",
        "context_parallel_size",
        "attention_backend",
        "attention_softmax_in_fp32",
        "apply_query_key_layer_scaling",
        "softmax_scale",
        "num_attention_heads",
        "num_query_groups",
        "kv_channels",
        "attention_dropout",
        "deterministic_mode",
        "fp8",
    )
    details = {
        "module_training": bool(getattr(core_attention, "training", False)),
        "qkv_format": getattr(core_attention, "qkv_format", None),
        "te_forward_mask_type": getattr(core_attention, "te_forward_mask_type", None),
        "num_splits": getattr(core_attention, "num_splits", None),
        "softmax_scale": getattr(core_attention, "softmax_scale", None),
        "config": {
            field: _enum_or_value(getattr(config, field, None)) for field in config_fields
        },
        "grad_enabled": torch.is_grad_enabled(),
        "cuda_autocast_enabled": torch.is_autocast_enabled("cuda"),
        "nvte_env": {key: value for key, value in sorted(os.environ.items()) if key.startswith("NVTE_")},
    }
    group = getattr(core_attention, "_tp_group", None)
    if group is not None and torch.distributed.is_initialized():
        details["tp_group"] = {
            "backend": str(torch.distributed.get_backend(group)),
            "ranks": torch.distributed.get_process_group_ranks(group),
        }
    return details


def capture_attention_execution(
    layer: int,
    *,
    core_attention: Any,
    path: str,
    packed_seq_params: Any,
    attention_bias: Tensor | None,
    inference_context: Any,
    attention_mask: Tensor | None,
    query: Tensor | None = None,
    key: Tensor | None = None,
    value: Tensor | None = None,
) -> None:
    """Capture path selection and mask state for the response rows in one forward."""
    state = _STATE
    if state is None or layer not in state["layers"]:
        return
    common = {
        "layer": layer,
        "core_attention_type": type(core_attention).__name__,
        "path": path,
        "has_packed_seq_params": packed_seq_params is not None,
        "has_attention_bias": attention_bias is not None,
        "has_inference_context": inference_context is not None,
        "query_layout": _tensor_layout(query),
        "key_layout": _tensor_layout(key),
        "value_layout": _tensor_layout(value),
        "runtime": _core_attention_runtime(core_attention),
    }
    if all(isinstance(tensor, torch.Tensor) for tensor in (query, key, value)):
        common["qkv_aliases"] = {
            "query_key": query.untyped_storage().data_ptr() == key.untyped_storage().data_ptr(),
            "query_value": query.untyped_storage().data_ptr() == value.untyped_storage().data_ptr(),
            "key_value": key.untyped_storage().data_ptr() == value.untyped_storage().data_ptr(),
        }
    for audit_row in state["rows"]:
        result = dict(common)
        result["input_ids_sha256"] = audit_row["input_ids_sha256"]
        result["response_mask"] = [
            _mask_summary(attention_mask, audit_row["row"], position) for position in audit_row["positions"]
        ]
        state["execution"].append(result)
