from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from typing import Any


def _read_text(path: str) -> str | None:
    try:
        return Path(path).read_text(encoding="utf-8").strip()
    except OSError:
        return None


def _run_read_only(command: list[str], timeout: float = 3.0) -> dict[str, Any]:
    if shutil.which(command[0]) is None:
        return {"command": command, "status": "unavailable", "output": ""}
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return {
            "command": command,
            "status": "error",
            "output": str(error),
        }
    output = "\n".join(
        part.strip() for part in (completed.stdout, completed.stderr) if part.strip()
    )
    return {
        "command": command,
        "status": "ok" if completed.returncode == 0 else "error",
        "return_code": completed.returncode,
        "output": output,
    }


def _capture_tegrastats() -> dict[str, Any]:
    command = ["tegrastats", "--interval", "100"]
    if shutil.which(command[0]) is None:
        return {"command": command, "status": "unavailable", "output": ""}
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=0.6,
        )
        output = "\n".join(
            part.strip()
            for part in (completed.stdout, completed.stderr)
            if part.strip()
        )
        return {
            "command": command,
            "status": "ok" if completed.returncode == 0 else "error",
            "return_code": completed.returncode,
            "output": output,
        }
    except subprocess.TimeoutExpired as error:
        values = []
        for value in (error.stdout, error.stderr):
            if isinstance(value, bytes):
                value = value.decode("utf-8", errors="replace")
            if value:
                values.append(value.strip())
        return {
            "command": command,
            "status": "ok" if values else "error",
            "output": "\n".join(values),
            "capture_method": "terminated_after_snapshot",
        }
    except OSError as error:
        return {"command": command, "status": "error", "output": str(error)}


def collect_jetson_runtime() -> dict[str, Any]:
    """전력 mode를 변경하지 않고 benchmark 재현에 필요한 상태만 수집한다."""

    nvpmodel = _run_read_only(["nvpmodel", "-q"])
    clocks = _run_read_only(["jetson_clocks", "--show"])
    tegrastats = _capture_tegrastats()
    combined = "\n".join(
        str(item.get("output", "")) for item in (nvpmodel, clocks, tegrastats)
    )
    clock_pairs = [
        (int(maximum), int(current))
        for maximum, current in re.findall(
            r"MaxFreq=([0-9]+).*?CurrentFreq=([0-9]+)",
            str(clocks.get("output", "")),
            re.I,
        )
    ]
    mode_match = re.search(r"(?:NV Power Mode|POWER_MODEL)\s*:?\s*([^\n]+)", combined, re.I)
    gpu_matches = re.findall(r"GR3D_FREQ\s+([^\s]+(?:\s+@[0-9]+)?)", combined)
    emc_matches = re.findall(r"EMC_FREQ\s+([^\s]+(?:\s+@[0-9]+)?)", combined)
    temperatures = {
        name.lower(): float(value)
        for name, value in re.findall(r"([A-Za-z0-9_]+)@(-?[0-9.]+)C", combined)
    }
    throttle_status = _read_text(
        "/sys/devices/platform/13e10000.host1x/15340000.pva0/power/control"
    )
    return {
        "nvpmodel": nvpmodel,
        "jetson_clocks": clocks,
        "tegrastats": tegrastats,
        "parsed": {
            "power_mode": mode_match.group(1).strip() if mode_match else "unknown",
            "maxn_super_detected": "MAXN_SUPER" in combined.upper(),
            "jetson_clocks_applied": (
                all(current >= maximum for maximum, current in clock_pairs)
                if clock_pairs else "unknown"
            ),
            "gpu_frequency": gpu_matches[-1] if gpu_matches else "unknown",
            "emc_frequency": emc_matches[-1] if emc_matches else "unknown",
            "temperatures_c": temperatures,
            "thermal_throttling": "unknown",
            "platform_power_control": throttle_status or "unknown",
        },
        "notes": [
            "No power or clock setting was changed.",
            "If nvpmodel details require privilege, run `sudo nvpmodel -q --verbose` manually.",
            "thermal_throttling remains unknown unless the captured platform output states it explicitly.",
        ],
    }


def inspect_model_state(model: Any, torch: Any) -> dict[str, Any]:
    dtype_counts: dict[str, int] = {}
    vision_dtypes: set[str] = set()
    language_dtypes: set[str] = set()
    parameter_bytes = 0
    for name, parameter in model.named_parameters():
        dtype = str(parameter.dtype).replace("torch.", "")
        count = int(parameter.numel())
        dtype_counts[dtype] = dtype_counts.get(dtype, 0) + count
        parameter_bytes += count * int(parameter.element_size())
        lowered = name.lower()
        if "vision" in lowered:
            vision_dtypes.add(dtype)
        if any(marker in lowered for marker in ("text_model", "language_model", "lm_head")):
            language_dtypes.add(dtype)

    module_names = [type(module).__name__ for module in model.modules()]
    linear4 = sum(name == "Linear4bit" for name in module_names)
    linear8 = sum(name == "Linear8bitLt" for name in module_names)
    quantization_config = getattr(getattr(model, "config", None), "quantization_config", None)
    has_bnb = linear4 > 0 or linear8 > 0 or quantization_config is not None
    dtype_set = set(dtype_counts)
    if linear4:
        classification = "4-bit"
    elif linear8:
        classification = "INT8"
    elif dtype_set == {"float32"}:
        classification = "FP32"
    elif dtype_set == {"float16"}:
        classification = "FP16"
    elif dtype_set == {"bfloat16"}:
        classification = "BF16"
    elif dtype_set:
        classification = "mixed"
    else:
        classification = "unknown"

    config = getattr(model, "config", None)
    generation_config = getattr(model, "generation_config", None)
    memory_after_load = {"allocated_mb": 0.0, "reserved_mb": 0.0}
    if torch.cuda.is_available():
        memory_after_load = {
            "allocated_mb": round(torch.cuda.memory_allocated() / 1024**2, 2),
            "reserved_mb": round(torch.cuda.memory_reserved() / 1024**2, 2),
        }
    return {
        "classification": classification,
        "parameter_dtype_counts": dtype_counts,
        "parameter_memory_mb": round(parameter_bytes / 1024**2, 2),
        "vision_model_dtypes": sorted(vision_dtypes) or ["unknown"],
        "language_model_dtypes": sorted(language_dtypes) or ["unknown"],
        "bitsandbytes_modules_present": has_bnb,
        "linear4bit_count": linear4,
        "linear8bitlt_count": linear8,
        "bitsandbytes_config_present": quantization_config is not None,
        "quantization_config": str(quantization_config) if quantization_config is not None else None,
        "attention_implementation": getattr(config, "_attn_implementation", "unknown"),
        "model_config_use_cache": getattr(config, "use_cache", None),
        "generation_config_use_cache": getattr(generation_config, "use_cache", None),
        "generation_cache_implementation": getattr(
            generation_config, "cache_implementation", None
        ),
        "gpu_memory_after_load": memory_after_load,
    }
