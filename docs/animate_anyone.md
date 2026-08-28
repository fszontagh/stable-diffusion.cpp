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

## Pose guider variant B (Sprite-Sheet-Diffusion)

Sprite-Sheet-Diffusion forks Moore-AnimateAnyone but
replaces the pose guider with a different, sprite-finetuned module
(`ModelTraining/models/pose_guider.py`): an 8-layer Conv+BatchNorm+ReLU stem
into a zero-initialized 1x1 projection and a learnable scalar `scale`
(init 2), followed by four downsampling towers each paired with a
self-attention `Transformer2DModel` block ("cross_attn1..4" in the
checkpoint - see the caveat below). It returns 5 pyramid features
(320/320/640/1280/1280 channels at 1/8, 1/16, 1/32, 1/64, 1/64 resolution)
instead of variant A's single 320-channel 1/8-resolution feature, and is
injected ControlNet-style: `fea[0]` after `conv_in` (shared with variant A),
`fea[1..4]` added after down_blocks 0-3 respectively.

**Variant selection is automatic**, probed from the loaded `--pose-guider`
checkpoint's own tensor names: `conv_layers.0.weight` present selects variant
B, `blocks.0.weight` selects variant A (Moore). Variant B additionally
requires `--ref-pose <path>` (a single reference pose image, matching
the upstream `pose_guider(pose, ref_pose)` API); generation fails loudly if
it's missing.

**Pose normalization differs by variant**: variant A images stay in `[0,1]`
(`do_normalize=False` in Moore's `cond_image_processor`); variant B rescales
both the per-frame pose images AND `--ref-pose` to `[-1,1]`
(`do_normalize=True` in the S pipelines - getting this wrong silently
degrades output). This port applies the correct normalization automatically
based on the probed variant; no extra flag is needed.

**Wiring caveat (transcribed exactly, not "fixed"):** reading
`ModelTraining/models/attention.py`'s `BasicTransformerBlock.__init__`, the
local `Transformer2DModel` used by `cross_attn1..4` is instantiated with
`cross_attention_dim=None` and `double_self_attention=False`, so that block
never allocates `attn2`/`norm2` at all (`self.attn2 = None`), and its
`forward()` only runs cross-attention `if self.attn2 is not None`. That means
despite `pose_guider.py`'s `forward(x, ref_x)` computing a whole `ref_x`
branch through the same conv towers and calling
`cross_attn{k}(x, ref_x)`, the `encoder_hidden_states=ref_x` argument is
**never consumed** - `cross_attn{k}` is pure self-attention on `x`. The
reference-pose branch is dead code in the upstream implementation (almost
certainly an authoring bug). This port transcribes the wiring exactly per
the task brief ("the .py file is the source of truth"): `PoseGuiderB`
accepts and validates a `ref_pose` tensor (matching the upstream signature
and the `--ref-pose` CLI requirement) but does not run it through any
weights, since doing so is provably inert to the returned features. See
`src/model/adapter/pose_guider_ssd.hpp` for the exact evidence trail.

Weights: no stable HTTP mirror is known; the only source found is a Google
Drive share (`tools/aa/download_weights.sh` attempts it with `gdown`, best
effort). As of this port those individual file downloads were **blocked**
(quota/permission errors), and the shared folder itself was confirmed to
contain only `denoising_unet-30000.pth`/`reference_unet-30000.pth`/
`motion_module.pth` - no `pose_guider.pth`. `tools/aa/dump_fixtures.py
--variant b` falls back to a fixed-seed **random-init** `PoseGuiderB`
checkpoint in that case (still a valid numerical fixture for the module
port - it exercises the exact same wiring/shapes, just not trained weights).
If you obtain a real checkpoint, identify it by tensor inspection (a
`conv_layers.0.weight` key of shape `(3,3,3,3)`), not by filename, and place
it at `--pose-guider <path>`.

Test: `sd-aa-test pose-guider-b` compares all 5 pyramid features against
`tools/aa/dump_fixtures.py --variant b`'s `pose_guider_b_out_{0..4}.npy` at
relative L2 <= 2e-3 per feature (variant A's `pose-guider` mode uses 1e-3;
this one is loosened per Task 5's atol+rtol ruling, with a numeric
justification: measured error grows near-linearly with the number of
`cross_attn{k}` `Transformer2DModel` blocks applied so far - 0, 3.1e-4,
7.5e-4, 9.7e-4, 1.15e-3 for fea[0..4] - tracing to the shared
`FeedForward`/`GEGLU` block's `ggml_gelu()` tanh approximation, a
long-standing ggml behavior that diffusers' own exact erf-based GEGLU
doesn't use; PoseGuiderB simply chains more of these blocks (4) than
anything else in the port checks in one fixture. See the comment at the
tolerance constant in `examples/aa_test/main.cpp` for the full derivation).

## Inputs

- `-r/--ref-image`: the character reference image (any resolution; it is
  resized internally for the VAE and CLIP paths).
- `--pose-dir`: a directory of pose skeleton images (OpenPose-style RGB
  skeleton on black, rendered at the target resolution), consumed in
  lexicographic filename order.
  - img_gen: the first pose image is used (one output image).
  - vid_gen: one output frame per pose image. Clips longer than 24 frames run
    through the reference's sliding-window scheme (moore-animate-anyone's
    `context.py` `uniform()`, ported faithfully: 24-frame windows, stride 1,
    4-frame overlap, closed_loop wraparound) - the per-window UNet outputs are
    scatter-accumulated and averaged before one scheduler step per timestep,
    exactly as `pipeline_pose2vid_long.py` does. Total frame count is capped
    at 128 (a sanity ceiling, not a model limit).

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

## Extracting pose frames from the Moore demo assets

Moore-AnimateAnyone ships demo ref images and pose *videos* (not frame
directories) under `configs/inference/{ref_images/,pose_videos/}`. `--pose-dir`
wants a directory of individual frame images, so extract with ffmpeg first:

```sh
mkdir -p poses/
ffmpeg -y -i moore-animate-anyone/configs/inference/pose_videos/anyone-video-2_kps.mp4 \
  -vf "select='lt(n\,8)'" -vsync 0 -frame_pts 0 poses/pose_%02d.png
```

This grabs the first 8 frames (in decode order; `select='lt(n,N)'` for the
first N). Rename/renumber to `00.png`, `01.png`, ... if you want a clean
`--pose-dir` (the loader only requires lexicographic order, not zero-padded
`NN.png` specifically - `pose_00.png ... pose_07.png` sorts the same way).
The demo pose videos are 512x768; resize frames to the target `-W`/`-H` ahead
of time (e.g. with Pillow) if you want bit-identical inputs across repeated
runs or across the C++/PyTorch comparison harness below - both `--pose-dir`
and the reference pipeline resize internally, but through different resample
implementations, and pre-resizing removes that as a variable.

Pair a ref image with its matching pose video using
`configs/prompts/test_cases.py` in the Moore repo (it lists validated
ref-image/pose-video pairs, e.g. `anyone-2.png` with
`anyone-video-2_kps.mp4`).

## End-to-end parity vs the PyTorch reference

`tools/aa/e2e_compare.py` drives both sides from the same pose frames, ref
image, seed, and steps, and scores per-frame SSIM:

```sh
/data/sdcpp-pixel-refs/venv/bin/python tools/aa/e2e_compare.py \
  --frames 6 --steps 10 --seed 42 --cfg 3.5 --width 512 --height 512 \
  --dump-steps
```

It (1) runs `Pose2VideoPipeline` (v2 config, motion modules on) on the
PyTorch side and captures the exact initial latent `prepare_latents()` drew
(post `torch.randn`, post `scheduler.init_noise_sigma` scale) to a `.npy`;
(2) invokes `sd-cli -M vid_gen` with `SDCPP_AA_DEBUG_INIT_LATENT` pointed at
that `.npy` so both sides denoise from the identical initial noise instead of
relying on two unrelated RNGs to agree; (3) scores per-frame SSIM
(`skimage.metrics.structural_similarity`) between the two frame sets.
`--dump-steps` additionally saves the per-step CFG'd v-pred model output and
post-DDIM-step latents on both sides (`SDCPP_AA_DEBUG_STEP_LATENTS` on the
C++ side) and reports their per-step rel L2 divergence - the tool used below
to tell a systematic bug apart from ordinary sampling-trajectory drift.

**Why not the full 8-frame/25-step clip on CPU.** The reference venv is
CPU-only (no CUDA torch installed) and PyTorch's 3D UNet is slow there: a
1-step timing probe at F=6 measured ~180s/step (worse at F=8), which
extrapolates to 60-75+ minutes for a 25-step, 8-frame run - past a reasonable
budget for one comparison pass. Per the task plan's fallback, the parity clip
was reduced to **6 frames, 10 steps** (same seed/cfg/resolution on both
sides); the PyTorch run took ~29 min on CPU, the C++ run ~2 min on GPU
(recipe below).

**Result** (measured 2026-08-28, RTX 3060 12 GB, CPU-only PyTorch venv):

| setting | value |
|---|---|
| frames | 6 |
| steps | 10 |
| cfg | 3.5 |
| size | 512x512 |
| seed | 42 |
| ref image / pose video | `anyone-2.png` / `anyone-video-2_kps.mp4` (Moore demo assets) |

| frame | SSIM |
|---|---|
| 0 | 0.9762 |
| 1 | 0.9754 |
| 2 | 0.9805 |
| 3 | 0.9796 |
| 4 | 0.9787 |
| 5 | 0.9792 |
| **mean** | **0.9783** |

**PASS (bar >= 0.85).** The first measurement came in at mean SSIM 0.7271 -
below the bar. Per the task plan ("if below: investigate per-module first,
then orchestration deltas - do not tune the threshold"), that number was
investigated rather than accepted or the bar adjusted, and the investigation
found and fixed a real bug; the number above is the post-fix, re-measured
result. The trail:

1. All `aa_test` fixture modes (`version`, `pose-guider`, `clip-embeds`,
   `scheduler`, `context-schedule`, `ref-bank`, `unet-step-f1`,
   `unet-step-f8`) passed against the rebuilt binary, at their existing
   tolerances (rel L2 <= 1e-3 per module/step) - the bug was not in any
   fixture-covered path.
2. The Task 10 sliding-window scheduler produces the *identical* single
   window `[0,1,2,3,4,5]` as the reference's `context.py uniform()` for
   `num_frames=6 <= context_frames=24` (both take the same early-return
   branch) - window scheduling was not a factor at this frame count.
3. **Precision bisection**: re-ran the C++ side forced to `--backend cpu
   --type f32` (no `--diffusion-fa`, no f16). Mean SSIM: 0.7233 -
   statistically the same as the GPU f16 + flash-attention run (0.7271),
   ruling out flash-attention precision and f16 quantization as the cause.
4. **Per-step divergence curve** (`--dump-steps`): rel L2 between the two
   sides' CFG'd v-pred model output at **step 0** - before any DDIM
   accumulation, with the injected identical initial latent - was **15%**,
   dozens of times larger than any fixture's module-level tolerance. Per the
   controller's decision rule, a large divergence already at step 0 means
   the bug is in per-step *input assembly*, not sampling-trajectory
   accumulation - chaotic amplification cannot explain a divergence that is
   already huge before any accumulation has happened.
5. **Bisecting the inputs**: with the identical injected latent, banks,
   pose feature, and clip embeds are the only remaining step-0 inputs.
   Dumped each independently (pose-guider output, VAE reference latent) from
   both sides on the *real* e2e images (not the aa_test fixtures) and
   compared directly:
   - pose-guider output (6 real, distinct pose frames - never exercised by
     the aa_test fixture, which broadcasts one pose image to every frame):
     rel L2 1.4e-3/frame, uniform across all 6 frames - in line with the
     fixture's own tolerance. Not the bug.
   - VAE reference latent: rel L2 **19%** (11-30% per channel) - the smoking
     gun.
6. **Root cause**: `sd_image_to_tensor()`'s resize (used for the VAE
   reference-latent path, and for pose-frame loading) defaults to
   `InterpolateMode::Nearest` (literal nearest-neighbor), while Moore's
   `VaeImageProcessor` (both the ref-image and pose-frame preprocessors) use
   `resample="lanczos"`. This was invisible to every existing fixture
   because the fixtures' `ref.png`/`pose.png` are already exactly
   512x512 - `sd::ops::interpolate()` short-circuits to a verbatim copy when
   input and output shapes already match (`if (input.shape() ==
   output_shape) return input;`), so the resize path was never actually
   exercised by any prior test. This task's real reference image
   (`anyone-2.png`, 576x768) is the first input in this port's history to
   force a genuine resize on this path.
7. **Fix**: `sd_image_to_tensor()` gained optional `resize_mode` /
   `resize_antialias` parameters (default unchanged - `Nearest`, no
   antialias - so every other caller across the codebase is untouched); the
   two AnimateAnyone call sites (`aa_load_image_tensor` for pose frames,
   and the ref-image VAE resize in `generate_animate_anyone()`) now pass
   `Lanczos` + antialias, matching Moore's resampler family. `src/core/util.h`,
   `src/core/util.cpp`, `src/stable-diffusion.cpp`.
8. **Effect of the fix**: VAE ref-latent rel L2 dropped 19% -> 3.7%; step-0
   v-pred rel L2 dropped 15% -> 1.4%; mean SSIM went 0.7271 -> **0.9783**.

**Residual divergence, post-fix**: the per-step rel L2 curve is now smooth
and monotonically growing with no jumps - v-pred: 1.4e-2, 3.8e-2, 5.6e-2,
6.4e-2, 8.8e-2, 9.5e-2, 10.6e-2, 10.6e-2, 10.4e-2, 11.9e-2 (steps 0-9);
latents: 1.3e-3, 3.3e-3, 7.2e-3, 12.6e-3, 22.3e-3, 33.0e-3, 46.3e-3, 59.4e-3,
70.7e-3, 84.2e-3. This is the textbook signature of ordinary DDIM
sampling-trajectory sensitivity (small per-module deltas - the port's own
~3.7% residual resize-algorithm mismatch, `InterpolateMode::Lanczos` is not
literally Pillow-bit-exact the way the dedicated CLIP-preprocessing path is
- compounding smoothly across a CFG'd loop), not a further systematic bug,
and is consistent with the task brief's own framing that bit-parity is not
the bar.

Reproduce: `tools/aa/e2e_compare.py --skip-python --dump-steps` re-scores
existing `python_frames/`/`python_steps/` against a freshly re-run C++ side;
drop `--skip-python` to redo the full comparison from scratch. Frames, the
injected initial latent, per-step dumps, and the JSON/markdown report are
saved under `/data/sdcpp-pixel-refs/outputs/task11/`.

## Known limitations

- End-to-end parity: mean SSIM against the PyTorch reference on a reduced
  6-frame/10-step clip is 0.978 (see above), passing the 0.85 bar. A
  step-by-step divergence bisection along the way found and fixed a real
  bug - `sd_image_to_tensor()`'s image resize defaulted to nearest-neighbor
  where the reference uses Lanczos, invisible to every fixture because the
  fixture images are already exactly target-sized (a no-op resize) - now
  fixed for the two AnimateAnyone resize call sites (pose frames, VAE
  reference latent). The small residual divergence left after the fix shows
  a smooth, jump-free per-step growth curve consistent with ordinary DDIM
  sampling-trajectory sensitivity, not a further orchestration bug.
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
- Pose guider variant B (Sprite-Sheet-Diffusion): the module port is verified
  against a fixture (`sd-aa-test pose-guider-b`), but no released checkpoint
  was obtainable (Google Drive quota/permission errors on every known
  source - see the variant B section above), so it is fixture-verified only
  with a random-init checkpoint. An end-to-end sprite-generation smoke test
  and SSIM comparison against the PyTorch reference were not run for this
  reason.
