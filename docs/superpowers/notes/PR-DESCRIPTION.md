# PR draft: AnimateAnyone model family (feat/sprite-sheet-diffusion)

## What this branch adds
- New VERSION_ANIMATE_ANYONE model family (SD1.5 UNet): ReferenceNet with
  16-bank post-norm1 hidden-state capture and concat-KV attn1 injection
  (cond half only), Moore pose guider (variant A) and Sprite-Sheet-Diffusion
  multi-scale pose guider (variant B, auto-probed from checkpoint tensor
  names), CLIP-vision-only conditioning (sd-image-variations, quick_gelu,
  PIL-exact bicubic preprocessing), AnimateDiff-v2 temporal reuse
  (norm eps 1e-6, AA-only mid-block ordering), a dedicated zero-SNR v-pred
  trailing DDIM stepper in alphas space, and a faithful context.py
  uniform() sliding-window long-video sampler (24/1/4 closed loop, up to
  128 frames).
- CLI: --reference-net, --pose-guider, --ref-pose, --pose-dir. Detection:
  SD1 signature with new positive corroboration for standalone diffusers
  UNets (cross-attn dim 768 probe) plus promotion when --reference-net is
  set.
- Verification harness sd-aa-test (9 modes), fixture dumper, weight
  downloader, and e2e comparison tooling under tools/aa/;
  docs/animate_anyone.md with a measured RTX 3060 12 GB recipe.
- Shared-code improvements along the way (all default-inert): version
  detection cached across tensor-name conversion, opt-in F32 conv kernels,
  optional Lanczos/antialias resize in sd_image_to_tensor.

## Verification story
- Per-module golden fixtures (fp32, fixed seed 12580) dumped from the
  PyTorch reference; all pass on CPU at rel L2 <= 1e-3 (2e-3 where ruled:
  CLIP preprocessing chain, pose-guider-b GELU-approx compounding).
- End-to-end parity: SSIM 0.9783 vs the PyTorch pose2vid pipeline
  (6 frames / 10 steps, identical injected initial latent) against a 0.85
  bar. The initial 0.727 failure was root-caused via a per-step divergence
  curve to a real nearest-vs-lanczos ref-image resize bug and fixed - not
  threshold-tuned.
- GPU runs are smoke-level (recipe, timings, VRAM); numerical parity is
  arbitrated on CPU.

## Open items (known, not defects)
- SSD released pose_guider weights externally blocked (gdown quota; the
  shared Drive folder lacks the file) - variant B is fixture-verified with
  a documented random-init checkpoint; e2e sprite smoke deferred until
  weights are obtained.
- Flash attention at F>1 costs ~1.63e-2 rel L2 on the step fixture;
  e2e-arbitrated acceptable (SSIM 0.978); required for attention memory on
  12 GB cards.
- Pre-existing, untouched, flagged for follow-up: PhotoMaker CLIP-vision
  gelu/quick_gelu heuristic suspicion; nearest-neighbor resize in other
  sd_image_to_tensor callers.
- cfg 3.5 family default is docs-only (no unset-marker exists for
  txt_cfg); docs recommend --cfg-scale 3.5.
