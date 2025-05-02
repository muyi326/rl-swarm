from collections import OrderedDict
from enum import Enum
import re

import psutil
import torch
from trl import GRPOConfig

DEFAULT_MEMORY_FRACTION = 0.95


def get_cuda_free_memory(device, memory_fraction=DEFAULT_MEMORY_FRACTION):
    total_memory = torch.cuda.get_device_properties(device).total_memory
    return max(
        0, int(total_memory * memory_fraction - torch.cuda.memory_reserved(device))
    )


def get_xpu_free_memory(device, memory_fraction=DEFAULT_MEMORY_FRACTION):
    total_memory = torch.xpu.get_device_properties(device).total_memory
    return max(
        0,
        int(total_memory * memory_fraction - torch.xpu.memory_reserved(device)),
    )


def get_mps_free_memory(memory_fraction=DEFAULT_MEMORY_FRACTION):
    total_memory = torch.mps.recommended_max_memory()
    return max(
        0, int(total_memory * memory_fraction - torch.mps.driver_allocated_memory())
    )


def get_cpu_free_memory(memory_fraction=DEFAULT_MEMORY_FRACTION):
    mem = psutil.virtual_memory()
    return int(mem.available * memory_fraction)


class Quantization(Enum):
    NONE = 1
    _4BIT = 2
    _16BIT = 3


# From https://docs.unsloth.ai/get-started/beginner-start-here/unsloth-requirements#fine-tuning-vram-requirements

# K = # of parameters (in B)
# V = min VRAM usage (in GB)

vram_lookup = {
    Quantization.NONE: OrderedDict(
        [
            (0.5, 4),
            (3, 16),
            (7, 38),
            (8, 44),
            (9, 48),
            (11, 58),
            (14, 66),
            (27, 128),
            (32, 152),
            (40, 192),
        ]
    ),
    Quantization._16BIT: OrderedDict(
        [
            (0.5, 2),
            (3, 8),
            (7, 19),
            (8, 22),
            (9, 24),
            (11, 29),
            (14, 33),
            (27, 64),
            (32, 76),
            (40, 96),
            (70, 164),
            (81, 192),
        ]
    ),
    Quantization._4BIT: OrderedDict(
        [
            (0.5, 1),
            (3, 3.5),
            (7, 5),
            (8, 6),
            (9, 6.5),
            (11, 7.5),
            (14, 8.5),
            (27, 22),
            (32, 26),
            (40, 30),
            (70, 41),
            (81, 48),
            (90, 53),
        ]
    ),
}

DEC_PATTERN = r"\d+\.?\d*"
B_PATTERN = re.compile(f"({DEC_PATTERN})[Bb]")
M_PATTERN = re.compile(f"({DEC_PATTERN})[Mm]")


def parse_param_count(model_name: str) -> float:
    if b := B_PATTERN.search(model_name):
        return float(b.group(1)) * 1e9
    if m := M_PATTERN.search(model_name):
        return float(m.group(1)) * 1e6
    return 0.0


Q_PATTERN = re.compile(r"(\d+)bit")


def parse_quantization(model_name: str) -> Quantization:
    if m := Q_PATTERN.search(model_name):
        q = m.group(1)
        match q:
            case "4":
                return Quantization._4BIT
            case "16":
                return Quantization._16BIT

    return Quantization.NONE


def estimate_peak_mem_percentage(
    model_name,
    grpo_config: GRPOConfig,
    quantization=Quantization.NONE,
) -> float:
    lookup = vram_lookup[quantization]

    # Look up estimated VRAM usage given parameter count.
    estimate = 0.0
    model_param_b = parse_param_count(model_name) / 1e9
    for param_b, value in lookup.items():
        if param_b >= model_param_b:
            estimate = value * 1e9
            break

    # Buffer for peak usage.
    if model_param_b > 32:
        estimate *= 1.25
    else:
        estimate *= 1.5

    # Find percentage of available memory.
    vllm_device = grpo_config.vllm_device
    if vllm_device == "auto":
        vllm_device = ""

    if torch.cuda.is_available():
        device = torch.device(vllm_device if vllm_device else "cuda:0")
        free = get_cuda_free_memory(device)
    elif torch.backends.mps.is_available():
        free = get_mps_free_memory()
    else:
        try:
            if torch.xpu.is_available():  # type: ignore
                device = torch.device(vllm_device if vllm_device else "xpu:0")
                free = get_xpu_free_memory(device)
            else:
                free = get_cpu_free_memory()
        except AttributeError:
            pass

    percentage = estimate / free
    return min(max(0.05, percentage), 0.95)
