## Pixelization: turning images into pixel art

[Pixelization](https://github.com/WuZongWei6/Pixelization) is not a diffusion
model. It is a two-stage GAN pixel-art filter: a "cell-to-pixel" generator
(C2PGen) that maps an input photo/illustration to a pixel-art-styled image
conditioned on a learned style code, followed by an "AliasNet" stage that
removes aliasing artifacts from the generator's output. The final step
downscales the AliasNet output 4x with nearest-neighbor sampling and then
upscales it back 4x, also with nearest-neighbor sampling. The net effect is
that the output image is presented at the same resolution as the input, but
its true detail is that of a 1/4-resolution pixel-art image, with each 4x4
block filled with a single flat color.

Pixelization runs as its own filter stage, independent of the diffusion
pipeline. It can post-process an image you already generated with sd.cpp, or
any external image, through `--mode pixelize`.

### 1. Get the upstream checkpoints

sd.cpp does not ship the Pixelization weights. Convert them from the
checkpoints published by the original repo. This port follows
[arenasys/pixelization_inference](https://github.com/arenasys/pixelization_inference),
a stripped-down inference-only fork of the original training code; its
`README.md` links to three Google Drive files:

- [`pixelart_vgg19.pth`](https://drive.google.com/file/d/1VRYKQOsNlE1w1LXje3yTRU5THN2MGdMM/view?usp=sharing)
- [`alias_net.pth`](https://drive.google.com/file/d/17f2rKnZOpnO9ATwRXgqLz5u5AZsyDvq_/view?usp=sharing)
- [`160_net_G_A.pth`](https://drive.google.com/file/d/1i_8xL3stbLWNF4kdQJ50ZhnRFhSDh3Az/view?usp=sharing)

Download all three into one directory (referred to below as
`<path-to-pth-files>`).

You also need a checkout of `arenasys/pixelization_inference` itself (referred
to below as `<path-to-pixelization_inference>`), because the conversion
script imports the original `C2PGen` module to compute the default style
code, and that module loads `pixelart_vgg19.pth` by a relative path at
construction time. Clone the repo and place (or symlink) `pixelart_vgg19.pth`
inside it, matching the layout the upstream README itself describes:

```bash
git clone https://github.com/arenasys/pixelization_inference <path-to-pixelization_inference>
ln -s <path-to-pth-files>/pixelart_vgg19.pth <path-to-pixelization_inference>/pixelart_vgg19.pth
```

### 2. Convert to gguf

```bash
python scripts/convert_pixelization.py \
    --models <path-to-pth-files> \
    --px-dir <path-to-pixelization_inference> \
    --out pixelization.gguf
```

This reads all three `.pth` files, drops the untrained/unreachable
`mod_conv_3`..`mod_conv_8` weights and the whole standalone VGG19 checkpoint,
and bakes in a `default_style_code` tensor computed from the upstream
`reference.png` image so that the converted file works standalone without a
separate style reference. The resulting gguf is roughly 170 MB (all tensors
are stored as F32; nothing is quantized).

`pixelart_vgg19.pth` is still required at convert time because `C2PGen`'s
constructor loads it by relative path, but none of it reaches the gguf: the
generator's `load_state_dict` overwrites every one of those tensors, and the
style encoder reads the resulting `c2p.pb_enc.vgg.*` weights instead.

Validate a converted file with:

```bash
python scripts/convert_pixelization.py --self-test pixelization.gguf
```

### 3. Run pixelization

```bash
sd-cli --mode pixelize \
    --pixelization-model pixelization.gguf \
    --init-img input.png \
    -o output.png
```

`--mode pixelize` skips the diffusion pipeline entirely; it only needs
`--init-img` for the source image. Input images whose width or height is not
a multiple of 4 are center-cropped to the nearest multiple of 4 (rounded
half-to-even, matching the upstream Python), reproducing the black-padding
upstream gets from `PIL.Image.crop` when the rounded size grows past the
source edge.

### 4. Custom style reference

By default the converted gguf uses the style baked in from upstream's
`reference.png`, so no reference image is needed and the VGG19-based style
encoder branch is never loaded. To use a different pixel-art style, pass a
reference image explicitly:

```bash
sd-cli --mode pixelize \
    --pixelization-model pixelization.gguf \
    --pixelization-ref my_style.png \
    --init-img input.png \
    -o output.png
```

When `--pixelization-ref` is set, the reference image is run through the
style encoder (C2PGen's `PixelBlockEncoder` + MLP) to derive a fresh style
code, which is then used instead of the baked-in default.

### Tiling

`--pixelization-tile-size N` (default `0`) processes the image in `N`x`N`
tiles instead of all at once, to bound peak memory on very large images.

## Limitations

Setting `--pixelization-tile-size` to a positive value produces output that
does not match the reference implementation. The GAN stages use instance
normalization, whose statistics are computed by reducing over the entire
spatial extent of the feature map. When the image is tiled, each tile only
sees its own extent, so the normalization statistics differ from the
untiled/reference computation. This is a correctness difference in the
generated pixel art, not a cosmetic seam between tiles; the whole-image
default (`--pixelization-tile-size 0`) is required to match upstream output.

## Credits

- [WuZongWei6/Pixelization](https://github.com/WuZongWei6/Pixelization) -
  the original C2PGen/AliasNet model and training code.
- [arenasys/pixelization_inference](https://github.com/arenasys/pixelization_inference) -
  the stripped-down inference-only fork this port follows.
