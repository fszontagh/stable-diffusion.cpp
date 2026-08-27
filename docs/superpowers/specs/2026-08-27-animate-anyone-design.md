# AnimateAnyone / Sprite-Sheet-Diffusion port for stable-diffusion.cpp

Date: 2026-08-27
Status: design approved in chat; this file is the binding spec
Repo: /data/sdcpp-pixel, branch feat/sprite-sheet-diffusion
Reference code: /data/sdcpp-pixel-refs/moore-animate-anyone (M),
/data/sdcpp-pixel-refs/sprite-sheet-diffusion (S). Paper: arXiv:2412.03685.
Detailed maps: docs/superpowers/notes/animate-anyone-pytorch-map.md and
docs/superpowers/notes/sdcpp-reuse-map.md (normative companions to this spec).

## 1. Goal

Add an AnimateAnyone model family to this stable-diffusion.cpp fork:
reference-image-driven, pose-skeleton-controlled character animation
(SD1.5 UNet + ReferenceNet + pose guider + AnimateDiff v2 temporal motion
module + CLIP-vision conditioning). Full port in one project (temporal
included). Two pose-guider variants, Moore's first:

- Milestone A (baseline): Moore AnimateAnyone - simple 8-conv pose guider,
  single injection at conv_in, weights from HF patrolli/AnimateAnyone.
- Milestone B (sprite finetune): Sprite-Sheet-Diffusion's multi-scale pose
  guider (pyramid injection after conv_in and each down block, internal
  cross-attention against a reference pose) with their Google Drive
  weights (folder 1VxbOv5PE441NsNStQlmqbIw0iyY9Mn9L; file ids in the
  notes map).

Driving use case: the pixelanim.cpp sprite pipeline (identity-locked
per-frame animation with preset pose-skeleton libraries), which consumes
this through the normal sd.cpp API.

## 2. Architecture summary (decisions)

- New SDVersion VERSION_ANIMATE_ANYONE, SD1.5 UNet family. Registration
  follows the checklist in the sdcpp-reuse map section 5 (model.h,
  model_version_to_str, get_sd_version detector on a signature tensor,
  construction branch, capability predicates incl.
  sd_version_supports_animatediff and supports_video_generation).
- ReferenceNet: a second UNetModelRunner instance loaded from
  --reference-net under prefix model.reference_net. (clone of the
  high_noise_diffusion_model plumbing). It is a stock SD1.5 UNet with NO
  conv_norm_out/conv_out (checkpoint omits them; do not allocate).
- Reference pass: runs ONCE per generation, timestep 0, batch = CFG pair
  (uncond ref latents use zeroed CLIP context). Each of the 16
  BasicTransformerBlocks banks its post-norm1 hidden states
  (NOT projected K/V). Banked tensors cross to the denoising runner as
  host sd::Tensor<float> (ControlNet-residual pattern) and enter the
  denoising graph as optional inputs.
- Injection in the denoising UNet's attn1: for the cond half of the CFG
  batch, context = concat([x, bank[block]] , sequence dim) so K/V see
  doubled sequence; for the uncond half, plain self-attention (the
  reference is NOT applied to uncond). Block pairing = descending norm1
  channel width with stable DFS order (16 blocks, fusion "full").
  Implementation: per-block nullable ref tensor map threaded through
  GGMLRunnerContext (generalization of the ip_context mechanism).
- Pose guider A (Moore): conv_in 3->16, blocks 16-16-32-32-96-96-256
  (strides 1,2,1,2,1,2), zero-init conv_out ->320, SiLU after every conv
  except conv_out; input = RGB skeleton image at full resolution, values
  [0,1] (do_normalize=false); output added to conv_in output of the
  denoising UNet. Cached across steps (guided-hint pattern). Loaded from
  --pose-guider into a sub-prefix of the main model map.
- Pose guider B (SSD): conv pyramid per the pytorch map (8 conv+BN+ReLU
  stages to 128, zero-init 1x1 final_proj to 320, learnable scale init 2,
  three downscale towers 320/640/1280/1280 and four Transformer2D
  cross-attn blocks attending pose features against reference-pose
  features); returns 5 pyramid features injected after conv_in and after
  each down block. Pose images normalized to [-1,1] (do_normalize=true -
  differs from Moore; keep per-variant). Requires a second input image
  (reference pose). Variant selected by probing checkpoint tensor names
  (conv_layers.* vs blocks.*).
- Temporal: reuse the existing AnimateDiff v2 machinery unchanged
  (mm layout matches; mid-block on; max 32 frames; frames on ne[3]).
  Motion module loads via existing --motion-module.
- Conditioning: CLIP-vision only (no text). Image encoder =
  sd-image-variations flavor: ViT hidden 1024, 24 layers, quick_gelu,
  projection_dim 768; single projected token (b,1,768) fed as c_crossattn;
  uncond = zeros. Reuse FrozenCLIPVisionEmbedder infrastructure; verify
  quick_gelu and visual_projection are honored; ref image resized to
  224x224 with CLIP preprocessing.
- Scheduler: DDIM, betas scaled-linear 0.00085..0.012 (existing),
  v-prediction (existing CompVisVDenoiser), trailing spacing
  (ddim_trailing + SIMPLE scheduler), plus zero-SNR beta rescale which
  must be verified and added if missing - this is the only scheduler
  work item. v1-config fallback (epsilon, no zero-SNR) must also work.
- VAE: stock AutoEncoderKL (sd-vae-ft-mse weights recommended), scale
  0.18215; reference latents use the distribution MEAN (not a sample).
- Video path: existing generate_animatediff_video flow; long-video
  sliding-window context (uniform schedule, frames 24, overlap 4,
  closed loop) is IN scope for parity with the reference pipeline, as a
  loop over windowed UNet computes with accumulate/average.
- CLI surface: -M vid_gen (and img_gen for single frame) with
  --diffusion-model (denoising_unet.pth), --reference-net, --pose-guider,
  --motion-module, --clip_vision (image encoder), --vae, -r (reference
  character image), --pose-dir <dir of skeleton frame PNGs> (new flag;
  single-frame mode accepts one pose image path), plus for variant B a
  --ref-pose <path>. cfg default 3.5, steps 25.

## 3. Weights

Milestone A set (download via HF, torch-pickle loads natively):
patrolli/AnimateAnyone: denoising_unet.pth, reference_unet.pth,
pose_guider.pth, motion_module.pth; base SD1.5 not needed at runtime
(denoising_unet.pth is complete); VAE sd-vae-ft-mse; image_encoder from
lambdalabs/sd-image-variations-diffusers. Name conversion: diffusers
UNet names (down_blocks.*) via the existing
convert_diffusers_unet_to_original_sd1, with model.reference_net. added
to the diffusion prefix vector; motion module keys already match.
Milestone B set: Google Drive per notes map (gdown), same base files.

## 4. Verification

- Fixture dumper (Python, in /data/sdcpp-pixel-refs, venv with torch +
  diffusers pinned per the repos' requirements): dumps, at fixed seed,
  (a) pose guider A and B outputs for a stored skeleton image,
  (b) the 16 ReferenceNet bank tensors for a stored ref image,
  (c) CLIP image embeds, (d) one denoising UNet forward (with banks,
  pose feature, motion modules, F=8) input/output pair, (e) DDIM v-pred
  zero-SNR trailing sigma/timestep tables and one scheduler step.
- C++ tests compare against fixtures at documented float tolerances
  (accumulation-order differences expected; target rel err <= 1e-3
  per tensor, exact shapes).
- Gated end-to-end: generate a short pose-driven clip from the S repo's
  sample character data (data/custom/characters layout) and compare
  visually + SSIM against the PyTorch pipeline's output for the same
  seed on the same machine.
- All work on branch feat/sprite-sheet-diffusion; fork conventions per
  AGENTS.md/CONTRIBUTING.md apply; PR review is done by a separate
  session per the user's workflow.

## 5. Out of scope

- Training; face-reenact (lmks2vid) paths; task_type != action machinery;
  the S repo's IP-Adapter ablation baseline; SDXL variants; audio.
- pixelanim.cpp integration (separate follow-on project in that repo).

## 6. Open items

- Zero-SNR rescale presence in the fork's scheduler stack (verify early;
  small addition if absent).
- SSD checkpoint availability via gdown (fallback: ask the user to
  download the Drive folder manually).
- VRAM plan for 12 GB: ReferenceNet runs once (can offload after);
  denoising UNet + motion modules at F<=24, 512x512 - validate with the
  existing offload/streaming options; document a working recipe.
