# AnimateAnyone

Pose-driven character image and animation generation
([Moore-AnimateAnyone](https://github.com/MooreThreads/Moore-AnimateAnyone)
architecture, SD1.5 family). Given one reference image of a character and one
or more pose skeleton images, it generates the character re-posed (img_gen) or
animated (vid_gen, one output frame per pose image).

There is no text prompt: conditioning is a CLIP-vision embedding of the
reference image (the sd-image-variations encoder), a ReferenceNet pass whose
self-attention states are injected into the denoising UNet, and a pose guider
that adds pose features after conv_in. Sampling is a dedicated 25-step
trailing DDIM v-prediction zero-SNR loop (the model's inference_v2 scheduler);
the generic samplers/schedulers flags are ignored for this family.

## Weight set

Download from HF [patrolli/AnimateAnyone](https://huggingface.co/patrolli/AnimateAnyone),
[stabilityai/sd-vae-ft-mse](https://huggingface.co/stabilityai/sd-vae-ft-mse) and
[lambdalabs/sd-image-variations-diffusers](https://huggingface.co/lambdalabs/sd-image-variations-diffusers)
(only its `image_encoder/`), e.g. via `tools/aa/download_weights.sh`.

| flag | file | role |
|---|---|---|
| `--diffusion-model` | `AnimateAnyone/denoising_unet.pth` | denoising UNet (spatial weights) |
| `--motion-module` | `AnimateAnyone/motion_module.pth` | temporal/motion modules (vid_gen) |
| `--reference-net` | `AnimateAnyone/reference_unet.pth` | ReferenceNet (headless SD1.5 UNet) |
| `--pose-guider` | `AnimateAnyone/pose_guider.pth` | pose guider |
| `--clip_vision` | `image_encoder/pytorch_model.bin` | sd-image-variations CLIP vision encoder |
| `--vae` | `sd-vae-ft-mse/diffusion_pytorch_model.safetensors` | VAE |

The family is detected when an SD1-signature diffusion model is loaded
together with `--reference-net`.

## Inputs

- `-r/--ref-image`: the character reference image (any resolution; it is
  resized internally for the VAE and CLIP paths).
- `--pose-dir`: a directory of pose skeleton images (OpenPose-style RGB
  skeleton on black, rendered at the target resolution), consumed in
  lexicographic filename order.
  - img_gen: the first pose image is used (one output image).
  - vid_gen: one output frame per pose image, currently capped at 24 frames
    (the reference context window; longer clips need the sliding-window path).

Width/height must be multiples of 8. Recommended defaults: 512x512 (or the
pose image's own aspect, e.g. 512x640), `--steps 25`, `--cfg-scale 3.5`,
`--seed 42`.

## Single frame (img_gen)

```sh
./build/bin/sd-cli \
  --diffusion-model ${SDCPP_AA_WEIGHTS}/AnimateAnyone/denoising_unet.pth \
  --motion-module   ${SDCPP_AA_WEIGHTS}/AnimateAnyone/motion_module.pth \
  --reference-net   ${SDCPP_AA_WEIGHTS}/AnimateAnyone/reference_unet.pth \
  --pose-guider     ${SDCPP_AA_WEIGHTS}/AnimateAnyone/pose_guider.pth \
  --clip_vision     ${SDCPP_AA_WEIGHTS}/image_encoder/pytorch_model.bin \
  --vae             ${SDCPP_AA_WEIGHTS}/sd-vae-ft-mse/diffusion_pytorch_model.safetensors \
  --type f16 --offload-to-cpu \
  -r ref.png --pose-dir poses/ \
  -W 512 -H 640 --steps 25 --cfg-scale 3.5 -s 42 \
  -o output.png
```

Single-frame generation runs the denoising UNet WITHOUT the motion modules.
This is deliberate: the motion modules were trained on 24-frame windows and
degenerate at a single frame. Verified against the PyTorch reference: the
pose2img flow with motion modules active at F=1 fails to denoise (the latent
std grows monotonically and the output is noise) in the reference and in this
port identically, while the spatial-only forward denoises normally.

## Animation clip (vid_gen)

```sh
./build/bin/sd-cli -M vid_gen \
  --diffusion-model ${SDCPP_AA_WEIGHTS}/AnimateAnyone/denoising_unet.pth \
  --motion-module   ${SDCPP_AA_WEIGHTS}/AnimateAnyone/motion_module.pth \
  --reference-net   ${SDCPP_AA_WEIGHTS}/AnimateAnyone/reference_unet.pth \
  --pose-guider     ${SDCPP_AA_WEIGHTS}/AnimateAnyone/pose_guider.pth \
  --clip_vision     ${SDCPP_AA_WEIGHTS}/image_encoder/pytorch_model.bin \
  --vae             ${SDCPP_AA_WEIGHTS}/sd-vae-ft-mse/diffusion_pytorch_model.safetensors \
  --type f16 --offload-to-cpu --diffusion-fa \
  -r ref.png --pose-dir pose_frames/ \
  -W 512 -H 640 --steps 25 --cfg-scale 3.5 -s 42 --fps 8 \
  -o clip.png
```

The frame count follows the number of files in `--pose-dir` (`--video-frames`
is ignored); the result is saved as a video next to `-o`'s path.

## The 12 GB GPU recipe

All four model files are fp32 on disk (about 8.5 GB of parameters as-is,
which does not fit a 12 GB card together with compute buffers). The measured
working recipe on an RTX 3060 12 GB:

- `--type f16`: halves the parameter memory (about 4.2 GB total).
- `--offload-to-cpu`: parameters live in RAM and are uploaded per runner
  while it computes. This matters because the ReferenceNet runs exactly once
  (its banks are cached for the whole denoise loop) but would otherwise stay
  VRAM-resident next to the denoising UNet.
- `--diffusion-fa` for vid_gen (F > 1): without flash attention the
  reference-bank-doubled spatial attention at 64x64 materializes a single
  ~8.6 GB score matrix at F=8 which no offload/streaming setting can split.
  F=1 does not need it (and skipping it there avoids flash attention's
  precision cost).

Measured on the RTX 3060 (512x640, 25 steps, models on a slow disk; first run
includes reading the torch zip checkpoints):

| run | wall time | peak VRAM |
|---|---|---|
| img_gen 1 frame | ~94 s | ~5.2 GB |
| vid_gen 8 frames | ~251 s | ~7.4 GB |

## Known limitations

- 24-frame cap: the reference `context_frames=24` window; longer pose
  sequences currently truncate (sliding-window generation is planned).
- Flash attention (required for F > 1 on 12 GB cards) costs measurable
  precision in the bank-doubled attention (task 8 measured ~1.6e-2 rel L2 on
  the step fixture vs ~3e-4 on CPU). End-to-end quality is still good; exact
  parity is arbitrated at the fixture level on CPU.
- `--type f16` quantizes weights that were fixture-verified at f32; visually
  fine, but CPU/f32 remains the numerical reference.
- The generic `--sampling-method`/`--scheduler` flags are ignored: the family
  always uses its DDIM v-prediction zero-SNR trailing scheduler (the zero-SNR
  terminal alpha cannot be represented in the sigma-space samplers).
- Back-view outputs: like the reference implementation, the model sometimes
  renders the character facing away when the pose skeleton is ambiguous.
