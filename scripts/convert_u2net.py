#!/usr/bin/env python3
"""Convert a PyTorch U^2-Net checkpoint to safetensors with BN fused into Conv.

Source weights:
  - Full U^2-Net (176 MB): https://github.com/xuebinqin/U-2-Net (u2net.pth)
  - Small U^2-NetP (4.7 MB): u2netp.pth from the same repo

Each REBNCONV block in U^2-Net is `Conv2d + BatchNorm2d + ReLU`. This script
folds the BatchNorm affine into the preceding Conv2d so the C++ inference path
only needs Conv2d + ReLU, no BN block.

Usage:
  pip install torch safetensors
  python convert_u2net.py u2net.pth u2net.safetensors
"""

import argparse
import sys
from pathlib import Path

try:
    import torch
    from safetensors.torch import save_file
except ImportError as e:
    sys.exit(f"missing dependency: {e}. install with: pip install torch safetensors")


def fuse_bn_into_conv(state: dict) -> dict:
    """Fold BatchNorm parameters into the preceding Conv2d weight + bias.

    U^2-Net's REBNCONV is `conv_s1 -> bn_s1 -> relu_s1`. After fusion the conv
    keeps weight + bias and the bn keys are dropped.
    """
    out = {}
    bn_keys_to_drop: set[str] = set()

    # Group state dict by REBNCONV prefix so we can find each conv/bn pair.
    prefixes: set[str] = set()
    for key in state:
        if key.endswith(".conv_s1.weight"):
            prefixes.add(key[:-len(".conv_s1.weight")])

    for prefix in prefixes:
        conv_w_key = f"{prefix}.conv_s1.weight"
        conv_b_key = f"{prefix}.conv_s1.bias"  # rarely present, sometimes None
        bn_w_key   = f"{prefix}.bn_s1.weight"
        bn_b_key   = f"{prefix}.bn_s1.bias"
        bn_m_key   = f"{prefix}.bn_s1.running_mean"
        bn_v_key   = f"{prefix}.bn_s1.running_var"

        if not all(k in state for k in (conv_w_key, bn_w_key, bn_b_key, bn_m_key, bn_v_key)):
            continue  # not a fusable REBNCONV pattern

        w  = state[conv_w_key].float()
        b  = state[conv_b_key].float() if conv_b_key in state else torch.zeros(w.shape[0])
        gamma = state[bn_w_key].float()
        beta  = state[bn_b_key].float()
        mean  = state[bn_m_key].float()
        var   = state[bn_v_key].float()
        eps   = 1e-5

        scale = gamma / torch.sqrt(var + eps)
        new_w = w * scale.view(-1, 1, 1, 1)
        new_b = (b - mean) * scale + beta

        out[conv_w_key] = new_w.contiguous()
        out[conv_b_key] = new_b.contiguous()

        bn_keys_to_drop.update([bn_w_key, bn_b_key, bn_m_key, bn_v_key,
                                f"{prefix}.bn_s1.num_batches_tracked"])

    # Copy through any tensor we didn't already write and that isn't a dropped BN.
    for key, tensor in state.items():
        if key in out or key in bn_keys_to_drop:
            continue
        out[key] = tensor.contiguous().float()

    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path, help="u2net.pth or u2netp.pth")
    ap.add_argument("output", type=Path, help="destination .safetensors")
    args = ap.parse_args()

    state = torch.load(args.input, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    # Strip "module." prefix (DataParallel checkpoints).
    state = {(k[len("module."):] if k.startswith("module.") else k): v for k, v in state.items()}

    before = len(state)
    state = fuse_bn_into_conv(state)
    after = len(state)
    print(f"fused {before - after} bn tensors into preceding conv")

    print(f"saving {len(state)} tensors -> {args.output}")
    save_file(state, str(args.output), metadata={"format": "u2net-bn-fused"})
    return 0


if __name__ == "__main__":
    sys.exit(main())
