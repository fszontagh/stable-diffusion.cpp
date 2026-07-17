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
        # The file is a convert-time-only dependency: load_state_dict below
        # overwrites every tensor it provides, so none of it reaches the gguf.
        # The style encoder reads c2p.pb_enc.vgg.* instead.
        from models.c2pGen import C2PGen

        G = C2PGen(3, 3, 64, 2, 4, 256, 256, activ="relu", pad_type="reflect")
        G.load_state_dict(g_state)
        G.eval()

        ref = process(greyscale(Image.open("reference.png")))
    with torch.no_grad():
        code = G.MLP(G.PBEnc(ref)).flatten()
    # Must stay RAW: normalizing shrinks demodulation's sum(w*code)^2, making
    # the +1e-8 eps non-negligible and breaking scale invariance; see spec 2.2.
    return code.numpy().astype(np.float32)


def convert(args):
    g = torch.load(f"{args.models}/160_net_G_A.pth", map_location="cpu", weights_only=True)
    alias = torch.load(f"{args.models}/alias_net.pth", map_location="cpu", weights_only=True)

    w = gguf.GGUFWriter(args.out, "pixelization")
    w.add_uint32("pixelization.arch_version", 1)

    def emit(prefix, state, skip_dead=False):
        for k, v in state.items():
            if skip_dead and any(d in k for d in DEAD):
                continue
            name = prefix + k
            name = name.replace("RGBEnc.", "rgb_enc.").replace("RGBDec.", "rgb_dec.")
            name = name.replace("PBEnc.", "pb_enc.").replace("MLP.", "mlp.")
            w.add_tensor(name, v.numpy().astype(np.float32))

    emit("c2p.", g, skip_dead=True)
    emit("alias.", alias)
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
    # The standalone pixelart_vgg19 weights are dead: load_state_dict overwrites them at
    # convert time, and the style encoder reads c2p.pb_enc.vgg.* instead; see spec 3.2.
    standalone_vgg = {n for n in names if n.startswith("vgg.")}
    assert not standalone_vgg, f"standalone vgg.* must be dropped, found: {standalone_vgg}"

    assert "default_style_code" in names, "default_style_code missing"
    code = next(t for t in r.tensors if t.name == "default_style_code")
    assert code.data.shape == (2048,), f"expected [2048], got {code.data.shape}"
    assert code.data.dtype == np.float32, f"code must be F32, got {code.data.dtype}"
    absmax = abs(code.data).max()
    # Raw, NOT normalized: normalizing breaks demodulation's eps assumption; see spec 2.2
    assert absmax > 1e6, f"code must be RAW (expect ~8.4e8), got absmax={absmax} -- did someone normalize it?"

    for req in [
        "c2p.rgb_enc.",
        "c2p.rgb_dec.mod_conv_1.",
        "c2p.rgb_dec.mod_conv_2.",
        "c2p.pb_enc.",
        "c2p.pb_enc.vgg.",
        "c2p.mlp.",
        "alias.rgb_enc.",
        "alias.rgb_dec.",
    ]:
        assert any(n.startswith(req) for n in names), f"missing tensors under {req}"

    print(f"self-test PASS: {len(names)} tensors, code absmax={absmax:.4e}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--models",
        required=False,
        help=(
            "Directory holding 160_net_G_A.pth, alias_net.pth, and "
            "pixelart_vgg19.pth (downloadable from the Google Drive links in "
            "the upstream arenasys/pixelization_inference README). Required "
            "unless --self-test is used."
        ),
    )
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
        missing = [name for name, val in (("--px-dir", a.px_dir), ("--models", a.models)) if not val]
        if missing:
            p.error(f"the following arguments are required: {', '.join(missing)}")
        convert(a)
