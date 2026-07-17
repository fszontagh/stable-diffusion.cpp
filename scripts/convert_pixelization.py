#!/usr/bin/env python3
"""Convert arenasys/pixelization_inference checkpoints (3x .pth) into one gguf."""
import argparse
import contextlib
import os
import sys

import gguf
import numpy as np
import torch

DEAD = [f"mod_conv_{i}" for i in range(3, 9)]  # untrained noise, unreachable; see spec 2.3


@contextlib.contextmanager
def chdir(path):
    prev = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def build_style_code(px_dir, g_state):
    sys.path.insert(0, px_dir)
    from PIL import Image
    from pixelization import greyscale, process

    with chdir(px_dir):
        # C2PGen's PixelBlockEncoder does torch.load('./pixelart_vgg19.pth')
        # at construction time; must run with cwd == px_dir (symlinked there).
        from models.c2pGen import C2PGen

        G = C2PGen(3, 3, 64, 2, 4, 256, 256, activ="relu", pad_type="reflect")
        G.load_state_dict(g_state)
        G.eval()

        ref = process(greyscale(Image.open("reference.png")))
    with torch.no_grad():
        code = G.MLP(G.PBEnc(ref)).flatten()
    code = code / code.abs().max()  # exact: demodulation is scale-invariant; see spec 2.2
    return code.numpy().astype(np.float32)


def convert(args):
    g = torch.load(f"{args.models}/160_net_G_A.pth", map_location="cpu", weights_only=True)
    alias = torch.load(f"{args.models}/alias_net.pth", map_location="cpu", weights_only=True)
    vgg = torch.load(f"{args.models}/pixelart_vgg19.pth", map_location="cpu", weights_only=True)

    w = gguf.GGUFWriter(args.out, "pixelization")
    w.add_uint32("pixelization.arch_version", 1)

    def emit(prefix, state, skip_dead=False, only=None):
        for k, v in state.items():
            if skip_dead and any(d in k for d in DEAD):
                continue
            if only and not k.startswith(only):
                continue
            name = prefix + k
            name = name.replace("RGBEnc.", "rgb_enc.").replace("RGBDec.", "rgb_dec.")
            name = name.replace("PBEnc.", "pb_enc.").replace("MLP.", "mlp.")
            w.add_tensor(name, v.numpy().astype(np.float32))

    emit("c2p.", g, skip_dead=True)
    emit("alias.", alias)
    emit("vgg.", vgg, only="features")  # classifier unused; see spec 3.2
    w.add_tensor("default_style_code", build_style_code(args.px_dir, g))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {args.out}")


def self_test(gguf_path):
    """Assert the conversion contract that the C++ side depends on."""
    from gguf import GGUFReader

    r = GGUFReader(gguf_path)
    names = {t.name for t in r.tensors}

    dead = {n for n in names if any(f"mod_conv_{i}" in n for i in range(3, 9))}
    assert not dead, f"dead mod_conv_3..8 must be dropped, found: {dead}"
    assert not any("classifier" in n for n in names), "vgg classifier must be dropped"

    assert "default_style_code" in names, "default_style_code missing"
    code = next(t for t in r.tensors if t.name == "default_style_code")
    assert code.data.shape == (2048,), f"expected [2048], got {code.data.shape}"
    absmax = abs(code.data).max()
    assert absmax <= 1.0 + 1e-6, f"code must be normalized, absmax={absmax} (F16 overflows at 65504)"
    assert absmax > 0.99, f"code should be normalized to absmax==1, got {absmax}"

    for req in [
        "c2p.rgb_enc.",
        "c2p.rgb_dec.mod_conv_1.",
        "c2p.rgb_dec.mod_conv_2.",
        "c2p.pb_enc.",
        "c2p.mlp.",
        "alias.rgb_enc.",
        "alias.rgb_dec.",
        "vgg.features.",
    ]:
        assert any(n.startswith(req) for n in names), f"missing tensors under {req}"

    print(f"self-test PASS: {len(names)} tensors, code absmax={absmax:.6f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--models", default="/data/SD_MODELS/pixelization")
    p.add_argument(
        "--px-dir",
        required=False,
        help=(
            "Path to a checkout of arenasys/pixelization_inference "
            "(not part of sd.cpp; no canonical location). Required unless "
            "--self-test is used."
        ),
    )
    p.add_argument("--out", default="pixelization.gguf")
    p.add_argument("--self-test", metavar="GGUF", default=None)
    a = p.parse_args()
    if a.self_test:
        self_test(a.self_test)
    else:
        if not a.px_dir:
            p.error("the following arguments are required: --px-dir")
        convert(a)
