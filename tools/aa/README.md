# AnimateAnyone reference tooling

Scripts for pulling the pretrained PyTorch weights and dumping golden-fixture
`.npy`/`.json` files from the Moore-AnimateAnyone reference implementation.
These fixtures are the ground truth the sd.cpp C++ AnimateAnyone port is
checked against.

Weights, venv, and fixtures all live OUTSIDE this repo (they are large
binary artifacts) - only this directory (scripts + README) is committed.

- Weights -> `$SDCPP_AA_WEIGHTS` (default `/data/sdcpp-pixel-refs/weights`)
- Python venv -> `/data/sdcpp-pixel-refs/venv`
- Fixtures -> `$SDCPP_AA_FIXTURES` (default `/data/sdcpp-pixel-refs/fixtures`)
- Reference PyTorch source -> `/data/sdcpp-pixel-refs/moore-animate-anyone`
  (`dump_fixtures.py` imports directly from its `src/` package - clone it
  there first, e.g. `git clone https://github.com/MooreThreads/Moore-AnimateAnyone /data/sdcpp-pixel-refs/moore-animate-anyone`)

## 1. Download weights

```
tools/aa/download_weights.sh
```

Pulls (see `docs/superpowers/notes/animate-anyone-pytorch-map.md` sections
5, 7, 8 for why each file is needed):

- `patrolli/AnimateAnyone`: `denoising_unet.pth`, `reference_unet.pth`,
  `pose_guider.pth`, `motion_module.pth`
- `stabilityai/sd-vae-ft-mse`: `config.json` + `diffusion_pytorch_model.safetensors`
- `lambdalabs/sd-image-variations-diffusers`: `image_encoder/{config.json,pytorch_model.bin}`
- SD1.5 base unet (`stable-diffusion-v1-5/stable-diffusion-v1-5` on the Hub,
  the current redirect target of `runwayml/stable-diffusion-v1-5`):
  `unet/config.json` **and** `unet/diffusion_pytorch_model.bin`.

  Deviation from a naive "config only" reading of the brief: both
  `UNet2DConditionModel.from_pretrained(...)` (ReferenceNet) and
  `UNet3DConditionModel.from_pretrained_2d(...)` (denoising net) build the
  model from this path AND load a base weights file from it before the
  AnimateAnyone `.pth` state dicts are loaded on top (see
  `moore-animate-anyone/src/models/unet_3d.py:641-660`, which raises
  `FileNotFoundError` if the weights file is absent). The base SD1.5 values
  are fully overwritten by the AnimateAnyone checkpoints immediately after,
  so only the shapes/existence of this file matter - but it must be present.

Writes `$SDCPP_AA_WEIGHTS/MANIFEST.txt` with `path  size_bytes  sha256` for
every downloaded file. Re-running the script skips files that already exist.

## 2. Python environment

```
python3.11 -m venv /data/sdcpp-pixel-refs/venv
/data/sdcpp-pixel-refs/venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
/data/sdcpp-pixel-refs/venv/bin/pip install diffusers==0.24.0 "transformers==4.30.2" "huggingface_hub==0.20.3" einops omegaconf safetensors numpy pillow
```

Notes:
- Use **python3.11** (not 3.12/3.14) - diffusers 0.24.0 / transformers
  4.30.2 predate newer CPython support and this combination is verified to
  import cleanly on 3.11.
- `diffusers==0.24.2` (as a literal string) does not exist on PyPI; the
  reference repo's own `requirements.txt` pins `diffusers==0.24.0`, which is
  what we install.
- `huggingface_hub` must be pinned to `<0.21` (we use `0.20.3`) - diffusers
  0.24.0 imports `cached_download` from `huggingface_hub`, which newer
  `huggingface_hub` releases removed. `pip` will warn that this conflicts
  with `accelerate`'s stated minimum; that's fine, we don't use
  `accelerate`'s `device_map` features here.
- CPU-only torch is intentional (`--index-url .../whl/cpu`) - all fixture
  dumps are fp32 CPU tensors; no GPU is required or used.

Verify:

```
/data/sdcpp-pixel-refs/venv/bin/python -c "import torch, diffusers; print(torch.__version__, diffusers.__version__)"
```

## 3. Dump fixtures

```
/data/sdcpp-pixel-refs/venv/bin/python tools/aa/dump_fixtures.py
```

Env overrides:
- `SDCPP_AA_WEIGHTS` - weights dir (default `/data/sdcpp-pixel-refs/weights`)
- `SDCPP_AA_FIXTURES` - output dir (default `/data/sdcpp-pixel-refs/fixtures`)
- `SDCPP_AA_REF` - path to the moore-animate-anyone checkout (default
  `/data/sdcpp-pixel-refs/moore-animate-anyone`)
- `SDCPP_AA_SPRITE_REF` - path to the sprite-sheet-diffusion checkout, used
  only to look for sample character/pose images (default
  `/data/sdcpp-pixel-refs/sprite-sheet-diffusion`)

Runs entirely on CPU. The full F=8 512x512 denoising UNet forward
(`unet_step_f8`) is the slowest step and can take several minutes on CPU -
this is expected and acceptable.

All dumps use a fixed seed (`torch.manual_seed(12580)`) and fp32 CPU
tensors. Re-run the script any time the reference source or weights change
to regenerate fixtures; it overwrites files in place.

Fixtures produced (see `manifest.json` for the authoritative shape/dtype
table once generated):

- `pose.png`, `ref.png`, `ref_pose.png` - 512x512 sample inputs. The
  sprite-sheet-diffusion sample checkout at the time this was run carried no
  `ground_truth`/`poses` character data, so these are synthesized: `ref.png`
  is a flat-color placeholder "character", and `pose.png`/`ref_pose.png` are
  COCO-18 skeletons rendered with the exact keypoint/limb BGR color table
  from `Dataprocessing/handlabel.py` onto a black background.
- `pose_guider_a_out.npy` - PoseGuider((16,32,96,256)) output on `pose.png`
- `clip_embeds.npy` - CFG-paired CLIP image embeddings for `ref.png`
- `ref_latents.npy` - VAE-encoded `ref.png` latent mean * 0.18215
- `ref_bank_00.npy` .. `ref_bank_15.npy` - ReferenceNet write-mode bank,
  harvested in descending `norm1` width order
- `unet_step_f1_in.npy` / `unet_step_f1.npy` - denoising UNet input/output,
  F=1 (motion module path skipped)
- `unet_step_f8_in.npy` / `unet_step_f8.npy` - same, F=8 (motion module
  active)
- `sched_v2.json` - DDIM v-pred/zero-SNR/trailing scheduler timesteps,
  post-rescale alphas_cumprod, and one worked `step()` example
- `manifest.json` - shape/dtype for every `.npy`, plus seed and package
  versions
