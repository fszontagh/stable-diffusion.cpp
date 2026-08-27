# AnimateAnyone Port Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the AnimateAnyone model family (ReferenceNet + pose guider + AnimateDiff temporal + CLIP-vision conditioning) to this stable-diffusion.cpp fork, Moore baseline first, Sprite-Sheet-Diffusion pose-guider variant second.

**Architecture:** Reuse the existing SD1.5 UNetModelRunner for both UNets (second instance via the high-noise-model plumbing pattern); bank ReferenceNet post-norm1 hidden states once per generation and inject them as concat-context into the denoising UNet's attn1 via a per-block map on GGMLRunnerContext; pose guider merges into the main model tensor map like the motion module; existing AnimateDiff v2 machinery provides temporal.

**Tech Stack:** C++17, ggml, existing fork infrastructure; Python venv with torch+diffusers only for fixture generation from the PyTorch reference.

**Spec:** docs/superpowers/specs/2026-08-27-animate-anyone-design.md
**Normative companions:** docs/superpowers/notes/animate-anyone-pytorch-map.md (called P-map below), docs/superpowers/notes/sdcpp-reuse-map.md (called R-map below). Section references like "P-map section 2" are binding requirements, not background reading.

## Global Constraints

- Branch: feat/sprite-sheet-diffusion in /data/sdcpp-pixel. Conventional commits. Fork conventions per AGENTS.md/CONTRIBUTING.md.
- Reference code (read-only): /data/sdcpp-pixel-refs/moore-animate-anyone, /data/sdcpp-pixel-refs/sprite-sheet-diffusion.
- Weights and fixtures live OUTSIDE the repo under /data/sdcpp-pixel-refs/ (weights/ and fixtures/); only the dumper script and download script are committed. Tests read fixtures via env SDCPP_AA_FIXTURES (default /data/sdcpp-pixel-refs/fixtures) and SKIP with a clear message when absent.
- Exact model constants come from P-map section 8 (UNet dims, CLIP encoder config, scheduler betas 0.00085..0.012, vae scale 0.18215, pose guider channels (16,32,96,256), motion module v2 config) - copy values from there verbatim; never invent.
- Reference pass: once per generation, timestep 0, CFG-doubled batch, banks = post-norm1 hidden states of the 16 BasicTransformerBlocks, pairing by descending norm1 width with stable DFS order (P-map section 2).
- Injection semantics: cond half attn1 context = concat([x, bank], seq dim); uncond half = plain self-attention (P-map section 2, "CFG handling").
- Pose image normalization: variant A input in [0,1]; variant B input in [-1,1] (P-map porting note 5).
- Tolerances for fixture comparisons: shapes exact; per-tensor relative L2 error <= 1e-3 (fp32 compute) unless a task states otherwise.
- ASCII hyphens only in authored text. No machine-specific absolute paths in committed files except the documented /data/sdcpp-pixel-refs default (this machine's convention, overridable by env).
- Verification harness: examples/aa_test/ (built as sd-aa-test) accumulates one subcommand per component; every task's GREEN step runs it.

---

### Task 1: Weights, Python env, fixture dumper

**Files:**
- Create: `tools/aa/download_weights.sh` (HF downloads via curl: patrolli/AnimateAnyone {denoising_unet,reference_unet,pose_guider,motion_module}.pth; stabilityai/sd-vae-ft-mse diffusion_pytorch_model.safetensors + config; lambdalabs/sd-image-variations-diffusers image_encoder/{config.json,pytorch_model.bin}; runwayml SD1.5 unet config.json only - into /data/sdcpp-pixel-refs/weights/)
- Create: `tools/aa/dump_fixtures.py` (runs inside the venv; imports from /data/sdcpp-pixel-refs/moore-animate-anyone/src)
- Create: `tools/aa/README.md` (venv setup: python3 -m venv /data/sdcpp-pixel-refs/venv; pip install torch torchvision diffusers==0.24.2 transformers einops omegaconf safetensors numpy pillow - pin diffusers to the version the reference targets; document fixture regeneration)

**Interfaces:**
- Produces fixture files (all fp32 .npy plus a manifest.json with shapes/metadata) under $SDCPP_AA_FIXTURES:
  - `pose.png` (a 512x512 COCO-18 skeleton render taken from /data/sdcpp-pixel-refs/sprite-sheet-diffusion sample data; also `ref.png` reference character and `ref_pose.png`)
  - `pose_guider_a_out.npy` [1,320,1,64,64] from PoseGuider((16,32,96,256)) with pose.png in [0,1]
  - `clip_embeds.npy` [2,1,768] (row 0 uncond zeros, row 1 cond) for ref.png
  - `ref_bank_{00..15}.npy` post-norm1 hidden states, CFG-doubled batch, from the write-mode hooks at t=0 (P-map section 2), plus `ref_latents.npy` (vae mean * 0.18215)
  - `unet_step_f1.npy` + `unet_step_f1_in.npy`: denoising UNet forward input/output, F=1, t=999, with banks + pose feature, motion modules PRESENT in weights but F=1 (skipped path)
  - `unet_step_f8.npy` + `unet_step_f8_in.npy`: same at F=8 (motion active), 512x512
  - `sched_v2.json`: timesteps + alphas_cumprod (post zero-SNR rescale) + one DDIM v-pred step (x_t, v, x_prev) for steps=25 trailing
  - `pose_guider_b_out_{0..4}.npy` (added in Task 12; dumper gains a --variant b flag then)
- Fixed seeds: torch.manual_seed(12580) everywhere; all dumps fp32 CPU.

- [ ] **Step 1: Write download_weights.sh, run it, verify file sizes/hashes recorded into weights/MANIFEST.txt.**
- [ ] **Step 2: Create venv per README; verify `python -c "import torch, diffusers"`.**
- [ ] **Step 3: Write dump_fixtures.py**: load models exactly as /data/sdcpp-pixel-refs/moore-animate-anyone/scripts/pose2vid.py does (P-map section 5 order); register write-mode hooks via ReferenceAttentionControl to harvest banks; run each dump; write manifest.json.
- [ ] **Step 4: Run the dumper; verify every fixture file exists with the shapes above (print table).**
- [ ] **Step 5: Commit** `feat: aa weight download + fixture dumper tooling` (scripts + README only).

### Task 2: Family registration skeleton + CLI flags

**Files:**
- Modify: `src/model.h` (VERSION_ANIMATE_ANYONE before VERSION_ESRGAN; predicate `sd_version_is_animate_anyone`; include in `sd_version_is_unet`/`sd_version_is_sd1` family logic per R-map section 5 item 1)
- Modify: `src/stable-diffusion.cpp` (model_version_to_str entry at matching index; capability predicates: supports_animatediff, supports_video_generation, supports_image_generation)
- Modify: `src/model_loader.cpp` get_sd_version(): detect via presence of BOTH `model.reference_net.` tensors in the merged map OR - since detection runs on the diffusion file alone - detect the denoising_unet.pth by SD1 signature plus `--model-args animate_anyone=true`; DECISION: detection = SD1 signature + reference_net_path non-empty at context level (add an explicit override in new_sd_ctx when reference_net_path is set, after get_sd_version returns VERSION_SD1; implement as a post-detection promotion in stable-diffusion.cpp init)
- Modify: `include/stable-diffusion.h` (+`const char* reference_net_path; const char* pose_guider_path; const char* ref_pose_image_path;` in sd_ctx_params_t; `const char* pose_images_dir;` + `sd_image_t pose_image;` additions in sd_img_gen_params_t/sd_vid_gen_params_t per R-map section 6 precedent)
- Modify: `examples/common/common.h` + `examples/common/common.cpp` (flags `--reference-net`, `--pose-guider`, `--pose-dir`, `--ref-pose`; forwarded into params)
- Modify: `src/name_conversion.cpp` (add `model.reference_net.` to the diffusion prefix vector so diffusers-name conversion applies under it, R-map section 6 "Wrinkle")
- Create: `examples/aa_test/main.cpp` + `examples/aa_test/CMakeLists.txt` (+ hook in `examples/CMakeLists.txt`): subcommand skeleton `sd-aa-test <mode> [args]`, modes added per task; mode `version` prints detected version for a file set and exits 0/1
- Test: `sd-aa-test version` run manually

- [ ] **Step 1: Add enum/string/predicates; build.**
- [ ] **Step 2: Add CLI flags + struct fields + forwarding; build; `sd-cli --help` shows the four flags.**
- [ ] **Step 3: aa_test skeleton builds; `sd-aa-test version --diffusion-model <denoising_unet.pth> --reference-net <reference_unet.pth>` prints `Animate Anyone`.**
- [ ] **Step 4: Commit** `feat: register AnimateAnyone model family and CLI surface`.

### Task 3: Pose guider A (Moore)

**Files:**
- Create: `src/model/adapter/pose_guider.hpp`: 
```cpp
namespace AnimateAnyone {
struct PoseGuiderA : public GGMLBlock {
  // conv_in 3->16 k3s1p1; blocks: 16->16 s1, 16->32 s2, 32->32 s1,
  // 32->96 s2, 96->96 s1, 96->256 s2; conv_out 256->320 k3s1p1 (zero-init
  // in checkpoint); SiLU after conv_in and every block, none after conv_out.
  // Checkpoint keys: conv_in.{weight,bias}, blocks.{0..5}.{weight,bias},
  // conv_out.{weight,bias} (P-map section 3 table).
  ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* pose_rgb); // [W,H,3,N] -> [W/8,H/8,320,N]
};
}
```
- Modify: `src/stable-diffusion.cpp`: load `--pose-guider` with prefix `model.pose_guider.` (motion-module pattern, R-map section 2 "Weight loading"); ownership inside the AnimateAnyone denoising wrapper (Task 7 consumes)
- Modify: `examples/aa_test/main.cpp`: mode `pose-guider`: load pose_guider.pth standalone (its own GGMLRunner harness inside aa_test), read `$SDCPP_AA_FIXTURES/pose.png`, scale [0,1], forward, compare to `pose_guider_a_out.npy` (write a tiny npy reader in `examples/aa_test/npy.hpp` - shape header parse + fp32 data)
- Test: `sd-aa-test pose-guider`

- [ ] **Step 1: Write npy.hpp + the pose-guider mode calling a not-yet-existing PoseGuiderA; build fails (RED).**
- [ ] **Step 2: Implement PoseGuiderA; build; run mode; iterate to rel-L2 <= 1e-3 (GREEN; print the error).**
- [ ] **Step 3: Commit** `feat: AnimateAnyone pose guider (Moore variant)`.

### Task 4: CLIP-vision conditioning path

**Files:**
- Modify: `src/conditioning/conditioner.hpp`: new `AnimateAnyoneConditioner` (or a branch in FrozenCLIPVisionEmbedder usage): CLIP-preprocess ref image at 224, ViT forward with quick_gelu, visual_projection to 768, output token shape [768,1,N]; uncond = zeros; expose as the family's cond_stage_model (selected in the Task 7 construction branch)
- Verify: quick_gelu + projection handling in `src/model/te/clip.hpp` for the sd-image-variations encoder config (hidden 1024, 24 layers, proj 768 - P-map section 8); fix activation selection if the fork hardcodes gelu
- Modify: `examples/aa_test/main.cpp`: mode `clip-embeds`: load image encoder (`--clip_vision` path), embed `$SDCPP_AA_FIXTURES/ref.png`, compare row 1 of `clip_embeds.npy`; assert row 0 equals zeros by construction
- Test: `sd-aa-test clip-embeds`

- [ ] **Step 1: Mode calling the embedder (RED if changes needed).**
- [ ] **Step 2: Implement/adjust; GREEN at rel-L2 <= 1e-3.**
- [ ] **Step 3: Commit** `feat: CLIP-vision-only conditioning for AnimateAnyone`.

### Task 5: Scheduler - zero-SNR v-pred trailing DDIM

**Files:**
- Modify: `src/runtime/denoiser.hpp` (+`stable-diffusion.cpp` alphas plumbing): add zero-SNR beta rescale (Lin et al: rescale alphas_cumprod sqrt so last step SNR=0) applied when the family requests it; confirm ddim_trailing + SIMPLE scheduler reproduces diffusers trailing spacing with steps_offset 1 for 25 steps; v-pred via existing CompVisVDenoiser
- Modify: `examples/aa_test/main.cpp`: mode `scheduler`: compute timestep grid + rescaled alphas_cumprod + one DDIM v-pred step from `sched_v2.json` inputs; compare all three (grid exact integers; alphas rel <= 1e-6; step output rel <= 1e-5)
- Test: `sd-aa-test scheduler`

- [ ] **Step 1: Mode + comparisons (RED).**
- [ ] **Step 2: Implement rescale + grid fixes minimally, without changing behavior for other families (guard by a flag on the denoiser setup); GREEN.**
- [ ] **Step 3: Commit** `feat: zero-SNR trailing DDIM support for AnimateAnyone`.

### Task 6: ReferenceNet runner + bank capture

**Files:**
- Modify: `src/model/diffusion/unet.hpp`: (a) UNetConfig flag `reference_headless` - skip out-norm/conv (weights absent; guard init and forward tail); (b) bank capture: give SpatialTransformer/BasicTransformerBlock its block index in DFS order at construction (16 spatial transformer blocks for SD1.5); when `ctx->aa_bank_capture != nullptr`, after norm1 write `ggml_cont` of the normed hidden states to `aa_bank_capture->tensors[idx]` via the cache mechanism (`persist_cache_tensor("aa.bank.<idx>", t)`, R-map section 1a)
- Modify: `src/core/ggml_extend.hpp`: GGMLRunnerContext members `struct AABank { std::vector<ggml_tensor*> tensors; }* aa_bank_capture = nullptr;` and `const std::vector<sd::Tensor<float>>* aa_bank_read = nullptr;` (Task 7 consumes read side)
- Modify: `src/stable-diffusion.cpp`: load `--reference-net` under `model.reference_net.` prefix, second UNetModelRunner with `reference_headless=true` (clone high_noise plumbing points listed in R-map section 6); implement `compute_reference_banks(ref_latents_cfg2, clip_embeds_cfg2) -> std::vector<sd::Tensor<float>> (16)` running one forward at t=0 and reading the 16 cached tensors back to host
- Modify: `examples/aa_test/main.cpp`: mode `ref-bank`: encode `$SDCPP_AA_FIXTURES/ref.png` with VAE (mean path * 0.18215, compare `ref_latents.npy` first at rel <= 1e-3), run compute_reference_banks with `clip_embeds.npy`, compare all 16 banks vs `ref_bank_XX.npy`; ordering must match the descending-width stable-DFS pairing (P-map section 2) - encode the mapping as a static table in the mode and assert widths {320x1:?..} match fixture shapes
- Test: `sd-aa-test ref-bank`

- [ ] **Step 1: Mode (RED).**
- [ ] **Step 2: headless flag + capture + host readback; GREEN per-bank rel <= 1e-3, print worst.**
- [ ] **Step 3: Commit** `feat: ReferenceNet forward with hidden-state bank capture`.

### Task 7: Injection + denoising UNet single-frame step

**Files:**
- Modify: `src/model/common/block.hpp` CrossAttention/BasicTransformerBlock: attn1 path - when `ctx->aa_bank_read` set and this block has an assigned bank index: cond rows use context = concat(x, bank) on the sequence dim; uncond rows plain. Implementation approach: the runner splits the CFG batch into two forwards (uncond batch and cond batch) so the graph stays static per forward - mirror how the sampler already distinguishes cond/uncond computes; the bank tensor enters as an optional graph input per block (`make_optional_input`)
- Modify: `src/model/diffusion/unet.hpp`: UNetDiffusionExtra gains `const std::vector<sd::Tensor<float>>* aa_banks; bool aa_is_uncond; ggml_tensor pose feature input` (pose feature = pose_guider output added after input_blocks_0_0, R-map section 3 step 4 precedent); construction branch in stable-diffusion.cpp for VERSION_ANIMATE_ANYONE wires conditioner (Task 4), pose guider (Task 3), banks (Task 6)
- Modify: `examples/aa_test/main.cpp`: mode `unet-step-f1`: load denoising_unet.pth (+pose guider, banks fixtures from .npy - NOT recomputed - isolate this task), t=999, input `unet_step_f1_in.npy`, compare `unet_step_f1.npy`
- Test: `sd-aa-test unet-step-f1`

- [ ] **Step 1: Mode (RED).**
- [ ] **Step 2: Implement injection + pose add; GREEN rel <= 1e-3.**
- [ ] **Step 3: Commit** `feat: reference-bank injection in denoising UNet (single frame)`.

### Task 8: Temporal step (F=8)

**Files:**
- Modify: only if needed - motion modules already hook via existing AnimateDiff paths (R-map section 2); ensure `model.diffusion_model.motion_module.` merge from `--motion-module` works with the family and banks repeat across frames on the sequence-concat read (bank is per-block [2,L,C] repeated to per-frame rows, P-map section 2 read-mode shapes)
- Modify: `examples/aa_test/main.cpp`: mode `unet-step-f8`: as Task 7 but F=8 latents `unet_step_f8_in.npy` vs `unet_step_f8.npy`
- Test: `sd-aa-test unet-step-f8`

- [ ] **Step 1: Mode (RED). Step 2: fix frame-repeat plumbing until GREEN rel <= 1e-3. Step 3: Commit** `feat: temporal denoising step with reference banks (F=8)`.

### Task 9: Full generation pipeline + CLI e2e (single frame and short clip)

**Files:**
- Modify: `src/stable-diffusion.cpp`: generation flow for the family: ref latents (VAE mean), CLIP embeds + zero uncond, pose images loaded from `--pose-dir` (sorted) or single pose file; pose guider forward once (cached); reference pass once; denoise 25 steps DDIM trailing v-pred zero-SNR cfg 3.5; decode per frame; wire into both img_gen (1 pose image) and vid_gen (`generate_animatediff_video` path with F = pose count <= 24 initially)
- Modify: `docs/animate_anyone.md` (usage doc with the working command lines) + README row
- Test: manual run against sample data from /data/sdcpp-pixel-refs/sprite-sheet-diffusion `data/custom/characters/...` (ground_truth first frame as -r, poses dir), 512x512, seed 42; output saved and eyeballed; no fixture compare (covered by Task 11)

- [ ] **Step 1: Implement wiring; build. Step 2: single-frame run produces a plausible character image following the pose (save artifacts under /data/sdcpp-pixel-refs/outputs/). Step 3: 8-frame clip runs on the 12 GB GPU (document offload flags used in docs/animate_anyone.md). Step 4: Commit** `feat: AnimateAnyone end-to-end generation pipeline`.

### Task 10: Long-video sliding window

**Files:**
- Modify: `src/stable-diffusion.cpp` (family sampling loop): uniform context scheduler (context_frames 24, stride 1, overlap 4, closed loop, index wrap - port `context.py:15` faithfully incl. ordered_halving jitter with step passed as 0), per-window UNet compute, accumulate noise_pred/counter, average, one scheduler step for all frames
- Modify: `examples/aa_test/main.cpp`: mode `context-schedule`: print the window list for F=64,step 0 and compare against a table dumped from the Python `uniform()` (add that dump to tools/aa/dump_fixtures.py as `context_windows.json`)
- Test: `sd-aa-test context-schedule` + a 32-frame generation run

- [ ] **Step 1: Fixture + mode (RED). Step 2: port scheduler, GREEN exact-match window list. Step 3: 32-frame clip generates. Step 4: Commit** `feat: sliding-window long-video sampling for AnimateAnyone`.

### Task 11: E2E parity vs PyTorch + VRAM recipe

**Files:**
- Create: `tools/aa/e2e_compare.py`: run the PyTorch pose2vid pipeline (v2 config) and the C++ CLI with identical inputs/seed/steps; compute per-frame SSIM between outputs; report
- Modify: docs/animate_anyone.md: verified 12 GB recipe (offload/streaming flags), known deltas
- Test: SSIM >= 0.85 mean on an 8-frame clip is the acceptance bar (sampling-noise and accumulation differences make bit-parity unrealistic; the fixtures already pin per-module correctness). If below: investigate per-module fixtures first, do not tune thresholds.

- [ ] **Step 1: Script + runs. Step 2: report committed into docs (numbers, not images). Step 3: Commit** `test: e2e parity harness and 12GB recipe`.

### Task 12: Milestone B - SSD pose guider + sprite weights

**Files:**
- Create: `src/model/adapter/pose_guider_ssd.hpp`: PoseGuiderB per P-map section 6 (conv_layers 8x Conv+BN+ReLU 3-3-16-16-32-32-64-64-128 with strides per table, zero-init final_proj 1x1 128->320, learnable scale (init 2), towers 320->320 s2, 320->640 s2, 640->1280 s2, 1280->1280 s1 each Conv+BN+ReLU pairs, cross_attn1..4 = Transformer2DModel heads 16 dim 88 inner 1408 conv proj k1, self/cross against ref-pose features); forward(pose, ref_pose) -> 5 features [1/8 320, 1/8 320 post-attn?, per P-map: five pyramid features at 1/8,1/16,1/32,1/64,1/64 with 320/320/640/1280/1280 channels]. IMPORTANT: transcribe the exact wiring from /data/sdcpp-pixel-refs/sprite-sheet-diffusion/ModelTraining/models/pose_guider.py - the P-map table is the checklist, the .py file is the source of truth.
- Modify: unet forward: when variant B active, add feature[0] after conv_in and feature[k] after down block k (P-map section 6 "Consumption")
- Modify: variant selection by checkpoint probe (`conv_layers.0.weight` present -> B; `blocks.0.weight` -> A); `--ref-pose` image required for B; pose normalization [-1,1] for B
- Modify: `tools/aa/download_weights.sh`: gdown the three file ids + folder listed in P-map section 6; `tools/aa/dump_fixtures.py --variant b` dumps `pose_guider_b_out_{0..4}.npy` using S's ModelTraining code
- Modify: `examples/aa_test/main.cpp`: mode `pose-guider-b` comparing all 5 features
- Test: `sd-aa-test pose-guider-b`, then an e2e sprite generation with the SSD weights on their sample character; SSIM comparison against the PyTorch S pipeline (pipeline_pose2vid_long_backup)

- [ ] **Step 1: Dumper extension + mode (RED). Step 2: implement module + injection + selection; GREEN. Step 3: e2e sprite clip generated + compared. Step 4: Commit** `feat: Sprite-Sheet-Diffusion multi-scale pose guider variant`.

---

## Self-review notes

- Spec coverage: spec section 2 decisions -> Tasks 2 (registration/CLI), 3+12 (guiders A/B), 4 (conditioning), 5 (scheduler incl. the open zero-SNR item), 6 (ReferenceNet), 7 (injection), 8 (temporal), 9 (pipeline+CLI+docs), 10 (long video), 11 (verification e2e + VRAM open item); spec section 3 weights -> Tasks 1 and 12; spec section 4 verification -> fixtures in Task 1, per-module modes in 3-8/10, e2e in 11.
- Gdrive availability risk (spec open item) is isolated to Task 12; Milestone A never blocks on it.
- Type consistency: aa_test modes and GGMLRunnerContext member names used consistently across Tasks 6-8; PoseGuiderA/B names consistent between Tasks 3/9/12.
- Deliberate scope cut: no bit-parity requirement at e2e (justified in Task 11); per-module fixtures carry the exactness burden.
