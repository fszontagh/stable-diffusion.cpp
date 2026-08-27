# Codebase Map: /data/sdcpp-pixel (== /data/sdcpp-fork, `src/` trees are byte-identical)

---

## 1. SD1.5 UNet — structure, attention, K/V caching feasibility

### Files
- `/data/sdcpp-pixel/src/model/diffusion/unet.hpp` (912 lines) — `UNetConfig`, `SpatialVideoTransformer` (SVD), `UnetModelBlock`, `UNetModelRunner`
- `/data/sdcpp-pixel/src/model/common/block.hpp` (663 lines) — `DownSampleBlock:8`, `UpSampleBlock:44`, `ResBlock:67`, `GEGLU:182`, `FeedForward:261`, `CrossAttention:307`, `BasicTransformerBlock:396`, `SpatialTransformer:469`, `AlphaBlender:580`, `VideoResBlock:615`
- `/data/sdcpp-pixel/src/model/diffusion/model.hpp` — `DiffusionModelRunner` base + `DiffusionParams`/`UNetDiffusionExtra`
- `/data/sdcpp-pixel/src/core/ggml_extend.hpp` — `GGMLBlock`, `GGMLRunner`, `GGMLRunnerContext:1724`, `ggml_ext_attention_ext:1349`

### Graph-construction pattern
Every module is a `GGMLBlock` with two phases:
1. **Constructor** populates `blocks["name"] = std::make_shared<...>` with a *dotted PyTorch-style key*. `init(params_ctx, tensor_storage_map, prefix)` walks the tree and allocates params named `prefix + key + ".weight"` — so **block key path == checkpoint tensor name**. See `UnetModelBlock` ctor `unet.hpp:304-492`.
2. **`forward(GGMLRunnerContext* ctx, ...)`** does `std::dynamic_pointer_cast<T>(blocks["name"])` then calls its `forward`, building ggml nodes (`unet.hpp:526-745`).

`init_params()` overrides (e.g. `SpatialTransformer::init_params` `block.hpp:478-492`, `CrossAttention::init_params` `block.hpp:317-327`) let a block *inspect the checkpoint* and swap sub-blocks / add optional sub-blocks (this is exactly how IP-Adapter's `to_k_ip`/`to_v_ip` get conditionally created).

Runner wrapper: `UNetModelRunner` `unet.hpp:748-910` — `build_graph()` calls `make_input()`/`make_optional_input()` for each host `sd::Tensor<float>`, builds the block forward, `ggml_build_forward_expand`. `compute()` at `unet.hpp:818` and the polymorphic `compute(n_threads, DiffusionParams&)` at `unet.hpp:841`.

`UNET_GRAPH_SIZE 102400` at `unet.hpp:14`.

### Where self-attention K/V lives
`BasicTransformerBlock::forward` `block.hpp:427-466`:
```
x = attn1->forward(ctx, x, x);        // block.hpp:454  self-attention (context == x)
x = attn2->forward(ctx, x, context);  // block.hpp:458  cross-attention
```
`CrossAttention::forward` `block.hpp:354-393` is the single place K/V are produced:
```
auto q = to_q->forward(ctx, x);        // block.hpp:370
auto k = to_k->forward(ctx, context);  // block.hpp:375
auto v = to_v->forward(ctx, context);  // block.hpp:376
x = ggml_ext_attention_ext(..., q, k, v, n_head, nullptr, false, ctx->flash_attn_enabled); // block.hpp:380
```
Note `attn1` is constructed with `context_dim == dim` (`block.hpp:414`), so `to_k`/`to_v` accept anything of width `dim` — **concatenating reference tokens along dim 1 (`n_context`) requires no weight changes at all.**

### (a) Caching K/V from a ReferenceNet pass — **already supported infrastructure**
`GGMLRunnerContext` carries cache hooks: `ggml_extend.hpp:1745-1749` (`get_cache_tensor`, `cache_tensor`) plus helpers `load_cache_tensor:1751` / `persist_cache_tensor:1758`. Backed by `GGMLRunner::cache()` `ggml_extend.hpp:3137` (writes into `cache_tensor_map`, `ggml_cont`s views) and `get_cache_tensor_by_name()` `ggml_extend.hpp:3144` (reads from a persistent `cache_ctx`/cache buffer that survives across `compute()` calls; freed by `free_cache_ctx_and_buffer()` `ggml_extend.hpp:3074`).

Two working precedents:
- **ControlNet guided-hint**: cached once, reused every step — `control.hpp:375` (read), `control.hpp:431` (flag), and the write at `control.hpp:~390` (`cache(guided_hint_cache_name(), guided_hint_output_ggml)`), with `ggml_set_output()` before `ggml_build_forward_expand`.
- **WAN/LTX VAE temporal feature cache**: `wan_vae.hpp:1355`, `ltx_vae.hpp:1341`.

**Difficulty: low-moderate.** In a `ReferenceNetRunner` you'd tap `CrossAttention::forward` in `attn1` to call `ctx->persist_cache_tensor("ref.<block_path>.k", k)` / `".v"`. The block already knows its own path only implicitly — you'd need to thread a name (simplest: give `CrossAttention` a `std::string ref_cache_key` member set during `init()`/construction, mirroring how `SpatialTransformer::init_params` receives `prefix`). Caveat: the cache lives on the *runner instance*, so cross-runner (ReferenceNet → main UNet) transfer needs either (i) reading back to host `sd::Tensor<float>` and re-uploading via `make_input`, or (ii) sharing a `cache_ctx`. Option (i) is the more conventional path in this codebase (that's how ControlNet residuals cross runners: `compute_sample_controls` `stable-diffusion.cpp:2484-2506` returns `std::vector<sd::Tensor<float>>`, then they're passed back in as `UNetDiffusionExtra::controls` `stable-diffusion.cpp:2729` and re-uploaded as graph inputs `unet.hpp:790-793`).

### (b) Concatenating extra K/V into the main UNet's attention — **easy**
Two idiomatic options, both already exemplified:
- **Concat-context (true AnimateAnyone semantics)**: `x = attn1->forward(ctx, x, ggml_concat(ctx->ggml_ctx, x, ref_tokens, 1))`. `ggml_concat` on dim 1 for K/V input is already done in `IPAdapter::Resampler::forward` `ip_adapter.hpp:91`, and on dim 2 for `c_concat` in `unet.hpp:558`.
- **Separate-attention-then-add (IP-Adapter style)**: `block.hpp:382-389` — compute a second `ggml_ext_attention_ext` against the injected K/V and add scaled. Requires plumbing one extra pointer through `GGMLRunnerContext` (`ip_context`/`ip_scale` at `ggml_extend.hpp:1731-1732`, set in `unet.hpp:800-801`).

The `ip_context`/`ip_scale` mechanism is the **exact template**: one nullable tensor + scale on the runner context, set in `build_graph`, consumed deep inside `CrossAttention`. For ReferenceNet you'd need a *per-block map* rather than a single tensor — add e.g. `std::map<std::string, std::pair<ggml_tensor*,ggml_tensor*>>* ref_kv` to `GGMLRunnerContext`.

---

## 2. AnimateDiff support

### Files
- `/data/sdcpp-pixel/src/model/diffusion/animatediff.hpp` (182 lines): `MotionModuleConfig:10`, `TemporalAttention:22`, `TemporalTransformerBlock:67`, `TemporalTransformer:99`, `MotionModule:134`, `AnimateDiffModel:146`
- `/data/sdcpp-pixel/docs/animatediff.md` — full user doc
- Hooks in `unet.hpp` and `stable-diffusion.cpp` (below)

### Weight loading
CLI `--motion-module <path>` → `SDContextParams::motion_module_path` (`examples/common/common.h:137`, option table `examples/common/common.cpp:435-439`) → `sd_ctx_params_t::motion_module_path` (`include/stable-diffusion.h:209`) → loaded at `stable-diffusion.cpp:823-830`:
```cpp
model_loader.init_from_file(sd_ctx_params->motion_module_path,
                            "model.diffusion_model.motion_module.");   // :825-826
animatediff_loaded = true;                                            // :829
```
So the motion module is merged into the *same* `tensor_storage_map` under a prefix, and becomes a **sub-block of the main UNet** — `blocks["motion_module"] = AnimateDiffModel(...)` at `unet.hpp:487-491`. No separate runner, no separate backend allocation.

**Auto-detection of topology**: `UNetConfig::detect_from_weights` `unet.hpp:90-95` probes `motion_module.down_blocks.0.motion_modules.0.temporal_transformer.proj_in.weight` → `enable_animatediff`; probes `mid_block.motion_modules.0...` → `animatediff_has_mid_block` (v2 has it, v3 doesn't).

### File format expected
Anything `ModelLoader::init_from_file` handles (`model_loader.cpp:231-258`): directory (diffusers), GGUF, `.json` safetensors index, safetensors, **torch zip (`.ckpt`/`.pth` pickle, `model_loader.cpp:244` → `torch_zip_io.cpp`)**, torch legacy (`model_loader.cpp:247` → `torch_legacy_io.cpp`).

Your `/data/SD_MODELS/animatediff/` has **both** supported forms:
- `mm_sd15_v3.safetensors` (836 MB) — v3, no mid-block
- `mm_sd_v15_v2.ckpt` (1.8 GB) — v2, torch-zip pickle, loads directly, **has** the mid-block motion module

`/data/SD_MODELS/animatediff_models/` are symlinks to the same two files. `animatediff_motion_lora/` is empty (motion LoRAs are not specifically supported; the domain-adapter LoRA goes through the normal `--lora-model-dir` path, `docs/animatediff.md:119-135`).

### Temporal hook points in the UNet
`unet.hpp:603-637` defines three lambdas, invoked in `forward`:
- `apply_motion_input(input_block_idx, h)` — called at `unet.hpp:652`, maps `di=(idx-1)/3`, `mj=(idx-1)%3` → `down_blocks.{di}.motion_modules.{mj}`
- `apply_motion_mid(h)` — `unet.hpp:680` (after middle block, before ControlNet residual add)
- `apply_motion_output(output_block_idx, h)` — `unet.hpp:717`, `ui=idx/3`, `mj=idx%3` → `up_blocks.{ui}.motion_modules.{mj}`

Gated by `config.enable_animatediff && num_video_frames > 1` (`unet.hpp:603`) — at F=1 the motion module is entirely skipped (documented at `docs/animatediff.md:162-165`).

### Frame batching — the `(b f)` convention
**There is no explicit `(b f)` collapse: b is implicitly 1 and frames occupy the batch dim `ne[3]`.**
- Latent shape is `[W, H, C, F]`; `num_video_frames` defaults to `x->ne[3]` (`unet.hpp:795-797`).
- Set in the sampler at `stable-diffusion.cpp:2726-2729`: `nvf = noised_input.shape()[3]` when `animatediff_loaded && shape()[3] > 1`; packed into `UNetDiffusionExtra{nvf, &controls, control_strength}`.
- `TemporalTransformer::forward` `animatediff.hpp:108-131` does the reshape: input `[W,H,C,F]` → `ggml_permute(h,2,3,0,1)` → `reshape_3d(C, F, W*H)` i.e. `(h w) f c`, runs temporal attention over `F`, permutes back. Asserts `x->ne[3] == num_frames` (`animatediff.hpp:117`).
- `TemporalAttention` `animatediff.hpp:43-64` adds a learned positional encoding `pos_encoder.pe` sized `[C, max_frames=32, 1]` (`animatediff.hpp:31`), sliced to F at `animatediff.hpp:53-55`.
- Spatial layers (`SpatialTransformer`) simply treat each frame as an independent batch element — no change needed.

Latent replication for img2video: `stable-diffusion.cpp:5034-5047`. Per-frame VAE decode: `stable-diffusion.cpp:5366-5383` (`sd::ops::slice(final_latents[i], 3, f, f+1)`).

### CLI usage & dispatch
`-M vid_gen --motion-module <path> --video-frames N --fps N` (`docs/animatediff.md:50-59`). Dispatch: `generate_video()` checks `animatediff_loaded && sd_version_supports_animatediff(version)` at `stable-diffusion.cpp:6866` and routes to `generate_animatediff_video()` `stable-diffusion.cpp:6806-6846`, which **repacks vid_gen params into `sd_img_gen_params_t` and calls `generate_image()`** with `animatediff_num_frames` set. Also `sd_ctx_supports_video_generation` at `stable-diffusion.cpp:3944-3946`.

### Constraints
- `sd_version_supports_animatediff` `stable-diffusion.cpp:75-77`: **only** `VERSION_SD1`, `VERSION_SD1_INPAINT`, `VERSION_SD1_PIX2PIX`. No SDXL.
- Frames capped at 32 (`stable-diffusion.cpp:6816-6819`), matching `MotionModuleConfig::max_frames = 32` (`animatediff.hpp:11`).
- Hard-coded channel topology `{320,640,1280,1280}` down / `{1280,1280,640,320}` up, 2 down-motion per block, 3 up-motion (`animatediff.hpp:14-17`) — i.e. it assumes the stock SD1.5 UNet shape.
- Fixed `transformer_blocks.0` (only depth-1 temporal transformers, `animatediff.hpp:104`).
- Output written as MJPEG AVI.

---

## 3. ControlNet — the per-block-residual pattern (reference for pose-guider)

### File
`/data/sdcpp-pixel/src/model/diffusion/control.hpp` (476 lines): `ControlNetBlock:16`, `ControlNet : GGMLRunner:311`.

### Conditioning image flow
1. Host: control image → `control_image_tensor` → `compute_sample_controls()` `stable-diffusion.cpp:2484-2506`, called per denoise step with `(noised_input, control_image, timesteps, cond.c_crossattn, cond.c_vector)`.
2. `ControlNet::build_graph` `control.hpp:353-402`: `hint` becomes a graph input only if the guided hint isn't cached (`control.hpp:373-382`).
3. **`input_hint_block`** — the "hint encoder", a plain 8-conv stack `Conv2d(3→16→16→32→32→96→96→256→320)` with 3 stride-2 downsamples, defined `control.hpp:93-109`, run by `input_hint_block_forward` `control.hpp:177-200`. Output is `[N, model_channels, h/8, w/8]`.
4. Added to `conv_in` output: `control.hpp:~262` — `h = input_blocks_0_0->forward(ctx, x); h = ggml_add(ctx->ggml_ctx, h, guided_hint);`
5. Residuals: a `zero_convs.{i}.0` (1×1 `Conv2d`) after every input block plus `middle_block_out.0` (`control.hpp:88-91, 130, 141, 163`), collected into `std::vector<ggml_tensor*> outs` and returned.
6. Read back to host as `std::vector<sd::Tensor<float>>` (`control.hpp:427-441`), passed to the UNet as `UNetDiffusionExtra::controls` (`stable-diffusion.cpp:2729`), re-uploaded at `unet.hpp:790-793`, applied at `unet.hpp:683-687` (middle) and `unet.hpp:696-700` (per skip connection), each scaled by `control_strength`.

### For a pose guider
Your pose guider is **strictly simpler**: it is exactly step 3+4 with no zero-convs and no per-block residuals. The `input_hint_block` stack in `control.hpp:93-109` + `input_hint_block_forward` `control.hpp:177-200` is a near-drop-in for AnimateAnyone's `PoseGuider` (which is `Conv(3→16)→…→zero_conv(→320)`). Add it as a sub-block of your UNet block (like `motion_module` at `unet.hpp:487-491`) and add its output to `h` right after `input_blocks_0_0->forward` at `unet.hpp:597`. The pose tensor enters as one more `make_optional_input` in `build_graph`.

Also note the **guided-hint caching precedent** (`control.hpp:317-319, 373-397, 431`) — for a static pose you compute it once and reuse across all steps.

---

## 4. IP-Adapter / PhotoMaker / PuLID — image-encoder → cross-attention injection

### `src/extensions/` structure
- `generation_extension.h` — `GenerationExtensionInitContext:18`, `GenerationExtensionConditionContext:30`, and the `GenerationExtension` vtable `:38-72`: `name()`, `is_enabled()`, `init()`, `get_param_tensors()`, `collect_loras()`, `add_ignore_tensors()`, `runner_done()`, `reset_runtime_condition()`, `prepare_condition()`, `before_condition()`, **`before_diffusion(DiffusionParams&, int step)` `:71`**. Factories declared `:74-75`.
- `photomaker_extension.cpp` — `PhotoMakerExtension:103`; modifies *conditioning* (`prepare_condition:189`), fuses ID embeds into text tokens.
- `pulid_extension.cpp` — `PuLIDExtension`; `before_diffusion` `:110-117` stuffs `id_embedding` into `FluxDiffusionExtra`. **This is the minimal template for injecting an extra tensor into the diffusion forward.**
- Registration: `stable-diffusion.cpp:1585-1613` (construct + `init`), `2123` (`collect_loras`), `2139-2161` (`reset`/`prepare`), `2788`/`2804` (`before_condition`/`before_diffusion` in the sample loop).
- CMake picks these up via glob `src/extensions/*.cpp` (`CMakeLists.txt:227`).

### CLIP-vision path (what AnimateAnyone needs)
IP-Adapter is **not** an extension — it's wired directly into `StableDiffusionGGML`:
- `--clip_vision <path>` → loaded with prefix `"clip_vision."` at `stable-diffusion.cpp:754-757`; `name_conversion.cpp:1473` remaps `clip_vision.` → `cond_stage_model.transformer.`.
- Runner: `FrozenCLIPVisionEmbedder` `conditioning/conditioner.hpp:561-612`, wrapping `CLIPVisionModelProjection` (`model/te/clip.hpp:426`), hard-coded `OPEN_CLIP_VIT_H_14`, with auto-detection of `self_attn.in_proj` fused-QKV layout (`conditioner.hpp:569-579`). Constructed at `stable-diffusion.cpp:1190` (SVD/WAN i2v) or `stable-diffusion.cpp:1376-1389` (auto-created when `--ip-adapter` given without an existing clip_vision).
- Feature extraction: `get_clip_vision_output()` `stable-diffusion.cpp:2164-2185` — `clip_preprocess(image, image_size, image_size)` then `clip_vision->compute(n_threads, pixel_values, return_pooled, clip_skip)`. `return_pooled=true` → `[projection_dim]`; `return_pooled=false, clip_skip=2` → `[hidden_size, 257]` penultimate patch tokens.
- Projection: `IPAdapter::IPAdapterRunner` `model/adapter/ip_adapter.hpp:116-205`; auto-selects `ImageProjModel` (classic, `:10-32`) vs `Resampler` (plus, `:34-114`) by presence of `ip_adapter.image_proj.latents` (`:128`). All dims inferred from tensor shapes (`:129-175`).
- Token computation: `compute_ip_adapter_tokens()` `stable-diffusion.cpp:2187-2214` — computes cond tokens plus **uncond tokens from a zeroed embedding** (`:2205-2207`).
- Injection: tokens → `UNetDiffusionExtra::ip_context`/`ip_scale` (`stable-diffusion.cpp:2730-2734`) → `UNetModelRunner::build_graph` `unet.hpp:788, 800-801` → `GGMLRunnerContext::ip_context` → consumed in `CrossAttention::forward` `block.hpp:382-389`, gated by `has_ip` which was set in `CrossAttention::init_params` `block.hpp:317-327` by probing for `to_k_ip.weight`. `attn2` (cross-attn only) is constructed with `enable_ip=true` (`block.hpp:415`).
- Name conversion for IP-Adapter checkpoints: `name_conversion.cpp:1307-1344` (`ip_adapter_index_map`, `convert_ip_adapter_name`), dispatched at `:1351-1353`.
- Docs: `docs/ip_adapter.md`. Your `/data/SD_MODELS/clip_vision/clip_vision_h.safetensors` (ViT-H, 1.26 GB) is exactly the file this path expects.

**For AnimateAnyone**, which replaces the text prompt with CLIP-image conditioning: the cleanest route is to reuse `FrozenCLIPVisionEmbedder` + a small `ImageProjModel`-like projector (`ip_adapter.hpp:10-32` is literally `Linear(clip_dim, 4*ctx_dim) + LayerNorm`, which is AnimateAnyone's image-proj) and feed the result as `DiffusionParams::context` (the `c_crossattn` slot) instead of `ip_context`.

---

## 5. Registering a new model family — every touchpoint (worked example: Mage-Flow, commit `8a51eb9`)

The Mage-Flow commit `8a51eb9 feat: add Mage-Flow support (#1808)` touched exactly:
```
README.md, docs/mage_flow.md, docs/edit.md, examples/CMakeLists.txt,
src/conditioning/conditioner.hpp, src/core/ggml_extend.hpp, src/model.h,
src/model/common/rope.hpp, src/model/diffusion/mage_flow.hpp,
src/model/diffusion/model.hpp, src/model/diffusion/qwen_image.hpp,
src/model/vae/mage_vae.hpp, src/model/vae/vae.hpp,
src/model_loader.cpp, src/name_conversion.cpp, src/stable-diffusion.cpp
```

### Checklist
1. **`src/model.h`** — add `VERSION_XXX` to the `SDVersion` enum **before `VERSION_ESRGAN`/`VERSION_COUNT`** (`model.h:13-62`; `VERSION_MAGE_FLOW` at `:59`, `VERSION_KREA2` at `:58`). Add a `sd_version_is_xxx()` predicate (`model.h:236-238`). Register in family predicates: `sd_version_is_dit` `:276-302`, or for you `sd_version_is_unet` `:85-92` / `sd_version_is_sd1` `:64-69`. VAE family: `sd_version_uses_wan_vae` `:254-259` etc.
2. **`src/stable-diffusion.cpp:79-126`** — add a display string to `model_version_to_str[]` **in the same index order** (Mage Flow at `:124`; there's no static_assert on this array, so ordering bugs are silent).
3. **`src/model_loader.cpp` `ModelLoader::get_sd_version()`** (`:448-…`) — add a tensor-name signature. Examples: `VERSION_KREA2` at `:478-481` (`model.diffusion_model.txtfusion.projector.weight`), `VERSION_MAGE_FLOW` at `:498-505` (nested: `transformer_blocks.0.img_mod.1.weight` **and** `img_in.weight` with `ne[0]==128`), `VERSION_ANIMA` at `:511`. Simple `return VERSION_X` on first match.
4. **`src/stable-diffusion.cpp` model construction** — add an `else if (sd_version_is_xxx(version))` branch in the big chain (`~:1000-1350`), assigning both `cond_stage_model` and `diffusion_model`. Mage-Flow: `:1233-1244`. Krea2: `:1073-1084`. **SD1/SD2/SDXL fall into the final `else` at `:3128`** (`FrozenCLIPEmbedderWithCustomWords` + `UNetModelRunner`, `stable-diffusion.cpp:1332-1341`).
5. **`src/model/diffusion/xxx.hpp`** — the model itself: a `GGMLBlock` config+block and a `XxxRunner : DiffusionModelRunner` implementing `get_desc()`, `get_param_tensors()`, `build_graph()`, `compute(n_threads, DiffusionParams&)`. Picked up automatically (header-only; no CMake edit needed — `CMakeLists.txt:216-240` globs `src/model/*/*.cpp` for translation units only).
6. **`src/name_conversion.cpp`** — if diffusers-format names differ, add a `convert_diffusers_dit_to_original_xxx()` (Krea2: `:797-831`) and dispatch it in `convert_diffusion_model_name()` (`:885-905`, Krea2 at `:900-901`). Add any prefix/vision-name special-casing (`:1490` for the Qwen3-VL vision tower shared by boogu/krea2/mage_flow/minimax).
7. **VAE selection** — `stable-diffusion.cpp` `create_vae()` lambda `:1437-1487` (Mage-Flow at `:1450-1454`); plus `sd_version_uses_*_vae()` in `model.h:240-263`; plus `src/model/vae/vae.hpp:77` for latent scaling/channel config; plus `src/name_conversion.cpp:1069` (`convert_first_stage_model_name`).
8. **Conditioner behavior** — `src/conditioning/conditioner.hpp` (Mage-Flow at `:1837`, `:2376-2451`) for prompt templates, vision-token formatting, min/max pixel budgets.
9. **Denoiser / prediction type** — `stable-diffusion.cpp:1840-1890` switch on `pred_type`; add a `XXX_FLOW_PRED` case + denoiser class in `src/runtime/denoiser.hpp` if needed.
10. **Ref-image preset** (edit models) — `src/model/diffusion/model.hpp:32-43` `REF_IMAGE_PRESETS` map + `get_default_ref_image_preset()` `stable-diffusion.cpp:3128-3151`.
11. **`DiffusionExtraParams`** — if you need per-step extra inputs, add a `XxxDiffusionExtra` struct and register it in the `std::variant` at `model/diffusion/model.hpp:124-134`; set it in the sample loop `stable-diffusion.cpp:2724-2745`.
12. **Capability predicates** — `sd_version_supports_video_generation` `stable-diffusion.cpp:3845-3847`, `sd_version_supports_image_generation` `:3849`, `sd_version_supports_ref_latent_img_cfg` `:158-167`, `sd_version_supports_animatediff` `:75-77`.
13. **CLI flags for auxiliary files** (see §6).
14. **Docs** — `docs/xxx.md` + a README row.

---

## 6. Weight conversion / loading & multi-file model sets

### Format handling
`ModelLoader::init_from_file(path, prefix)` `model_loader.cpp:231-258` dispatches on content:
| Check | Loader |
|---|---|
| `is_directory` | `init_from_diffusers_file` `:425-447` (expects `unet/`, `vae/`, `text_encoder/`, `text_encoder_2/` subdirs) |
| `is_gguf_file` | `init_from_gguf_file` `:287-311` |
| `.json` | `init_from_safetensors_index_file` |
| `is_safetensors_file` | `init_from_safetensors_file` `:315-…` |
| `is_torch_zip_file` | `init_from_torch_zip_file` `:398-422` → `src/model_io/torch_zip_io.cpp` |
| fallback | `init_from_torch_legacy_file` → `src/model_io/torch_legacy_io.cpp` |

**PyTorch pickle `.pth`/`.ckpt` loads directly — no offline conversion step.** Pickle parsing lives in `src/model_io/pickle_io.cpp`. `mm_sd_v15_v2.ckpt` in your model dir goes through the torch-zip path.

`init_from_file_and_convert_name(path, prefix)` `model_loader.cpp:277-283` = load + `convert_tensors_name()`.

### Name mapping
`convert_tensor_name(name, version)` `name_conversion.cpp:1346-1577` is the single entry point, called from `ModelLoader::convert_tensors_name()` `model_loader.cpp:260-272`. Pipeline:
1. LoRA/underscore normalization, `convert_sep_to_dot` `:1127` for UNet families.
2. `prefix_map` rewriting `:1457-1473` — `unet.` / `diffusion_model.` / `transformer.` → `model.diffusion_model.`; `vae.` → `first_stage_model.`; `te.`/`text_encoder.` → `cond_stage_model.transformer.`; `clip_vision.` → `cond_stage_model.transformer.`.
3. Diffusion-model sub-conversion `:1501-1522` → `convert_diffusion_model_name` `:885` → per-family converters: `convert_diffusers_unet_to_original_sd1` `:227-335` (the one relevant to you — maps `down_blocks.i.resnets.j.*` → `input_blocks.N.0.*`, `attentions` → `.1`, `mid_block` → `middle_block`, `up_blocks` → `output_blocks`, `conv_in` → `input_blocks.0.0`, `conv_out` → `out.2`), `..._sdxl` `:336`, `..._sd3` `:452`, `..._flux` `:558`, `..._krea2` `:797`.
4. cond_stage / first_stage / pmid / controlnet / ip_adapter blocks `:1524-1573`.

Note the ControlNet `.pth` special case at `:1559-1565` (strips `control_model.`).

**Wrinkle for AnimateAnyone**: the reference HF release ships **diffusers-format** UNets (`down_blocks.*`), and the ReferenceNet is a diffusers UNet too. `convert_diffusers_unet_to_original_sd1` handles that — but it's keyed off `starts_with(name, prefix)` where prefix ∈ `diffuison_model_prefix_vec`. A second UNet loaded under `model.reference_net.` would need that prefix added to the vector (see how `model.high_noise_diffusion_model.` is handled).

### Multi-file model sets — precedent for a second UNet
The `--high-noise-diffusion-model` flag (WAN 2.2 two-expert setup) is exactly the pattern:
- Public struct field: `include/stable-diffusion.h:200` `const char* high_noise_diffusion_model_path;`
- CLI: `examples/common/common.h:118` + option-table entry in `examples/common/common.cpp`, forwarded at `examples/common/common.cpp:~885`
- Load with a distinct prefix: `stable-diffusion.cpp:726-730` → `init_from_file(path, "model.high_noise_diffusion_model.")`
- Second runner member: `stable-diffusion.cpp:231` `std::shared_ptr<DiffusionModelRunner> high_noise_diffusion_model;`
- Constructed at `stable-diffusion.cpp:1177-1184` with prefix `"model.high_noise_diffusion_model"`
- Registered/configured: `:1365-1371` (`register_runner_params`, vram budget, stream layers), `:1639-1641` (flash attn), `:1976-1977` + `:2073-2074` (LoRA adapter)
- LoRA routing: `ModelManager::LoraSpec::is_high_noise` `model_manager.h:25`, used at `model_manager.cpp:573` and `stable-diffusion.cpp:1945-1952`
- Sampling: separate sample params/steps `stable-diffusion.cpp:4280-4341`

**A `--reference-net` flag would clone this line-for-line**, plus prefix registration in `name_conversion.cpp`'s `diffuison_model_prefix_vec`.

Also note `--motion-module` (`stable-diffusion.cpp:823-830`) as the *other* pattern: merge into the main model's tensor map under a sub-prefix and let the main block own it. For a pose guider (small, tightly coupled), that's the better fit; for a full ReferenceNet UNet, the `high_noise` pattern is better (separate runner, separately offloadable).

Extra-parameter escape hatch without new flags: `--model-args key=value,...` (`sd_ctx_params_t::model_args`, `examples/common/common.cpp:459-465`, parsed via `parse_key_value_args`).

---

## 7. Schedulers

`enum scheduler_t` `include/stable-diffusion.h:66-82`: `DISCRETE`, `KARRAS`, `EXPONENTIAL`, `AYS`, `GITS`, `SGM_UNIFORM`, `SIMPLE`, `SMOOTHSTEP`, `KL_OPTIMAL`, `LCM`, `BONG_TANGENT`, `LTX2`, `LOGIT_NORMAL`, `FLUX2`, `FLUX`, `BETA`. Implementations dispatched in `src/runtime/denoiser.hpp:1050-1120`.

Samplers (`sample_method_t`, `include/stable-diffusion.h:~40-60`, names at `stable-diffusion.cpp:128-149`) include **`DDIM_TRAILING_SAMPLE_METHOD`** (`stable-diffusion.h:49`). Implementation note at `denoiser.hpp:2843-2844`: *"DDIM is equivalent to Euler Ancestral with the Simple scheduler"* — i.e. `--sampling-method ddim_trailing` forces the `SIMPLE_SCHEDULER` timestep spacing. `eta` is plumbed through (`sd_sample_params.eta`).

**Beta schedule (matching diffusers `DDIMScheduler`)**: `calculate_alphas_cumprod()` `stable-diffusion.cpp:173-186` implements **scaled_linear** (`linear_start=0.00085`, `linear_end=0.0120`, 1000 timesteps) — identical to diffusers `beta_schedule="scaled_linear"`. Overridable: if the checkpoint contains an `alphas_cumprod` tensor it is used verbatim (`load_alphas_cumprod` `stable-diffusion.cpp:682-…`, applied in `refresh_compvis_denoiser_sigmas` `:665-680`). Sigmas are then `sqrt((1-ᾱ)/ᾱ)` (`:676`) — standard k-diffusion `CompVisDenoiser` (`denoiser.hpp:1126`), with `CompVisVDenoiser` `:1198` for v-pred.

There is **no exposed `beta_schedule` option** (linear vs scaled_linear vs squaredcos) — only scaled_linear or checkpoint-supplied. `docs/animatediff.md:56` recommends `--sampling-method euler --scheduler discrete` with `--cfg-scale 8.0` to reproduce reference AnimateDiff output. AnimateAnyone's reference pipeline uses `DDIMScheduler(beta_schedule="scaled_linear", steps_offset=1, clip_sample=False)` — the beta schedule matches out of the box; `steps_offset`/trailing spacing is what `ddim_trailing` + `SIMPLE_SCHEDULER` approximates.

---

## 8. VAE

- **SD1.5 VAE**: `AutoEncoderKL` in `/data/sdcpp-pixel/src/model/vae/auto_encoder_kl.hpp`, the default fall-through branch of `create_vae()` `stable-diffusion.cpp:1470-1486`. Prefix `first_stage_model`, 4-channel latents, ×8 downsample. Fully supported; nothing new needed for AnimateAnyone.
- **Selection logic**: `create_vae()` lambda `stable-diffusion.cpp:1437-1487`, chained on `sd_version_is_ltxav` → `minimax_h3` → `mage_flow` → `uses_hunyuan_video_vae` → `uses_wan_vae` → **else `AutoEncoderKL`**. The family→VAE predicates live in `model.h:240-263` (`sd_version_uses_flux_vae`, `..._flux2_vae`, `..._wan_vae`, `..._hunyuan_video_vae`); latent channel/scale config in `src/model/vae/vae.hpp:77`.
- Overrides: `--vae <path>` (external file, `examples/common/common.cpp:400-404`), `--vae-format {auto,flux,sd3,flux2,wan}` (`:405-409`, applied via `sd_vae_format_to_version` `stable-diffusion.cpp:1427-1435`), `--taesd`/`--tae` (`create_tae` `:1407-1425`), `FakeVAE` for pixel-space models (`:1489-1493`).
- Per-frame decode for video-batched latents already exists: `stable-diffusion.cpp:5366-5383`.
- Available weights on disk: `/data/SD_MODELS/vae`, `/data/SD_MODELS/taesd`, `/data/SD_MODELS/TAESD`.

---

## Summary: recommended integration shape for AnimateAnyone

| Component | Reuse | New work |
|---|---|---|
| SD1.5 UNet | `UNetModelRunner` as-is | pass ref-K/V map through `GGMLRunnerContext` |
| ReferenceNet | second `UNetModelRunner` instance, prefix `model.reference_net.` | `--reference-net` flag cloned from `high_noise_diffusion_model` (`stable-diffusion.cpp:231, 726-730, 1177-1184, 1365-1371`); a "capture mode" flag on the runner so `attn1` writes K/V instead of reading |
| Attention injection | `CrossAttention::forward` `block.hpp:354-393`; `ggml_concat(...,1)` on the attn1 context, or IP-Adapter's dual-attention-add `block.hpp:382-389` | per-block K/V lookup keyed by block path |
| Pose guider | `ControlNetBlock::input_hint_block` `control.hpp:93-109` + `input_hint_block_forward:177-200`; add after `unet.hpp:597`; cache like `control.hpp:373-397` | ~40 lines |
| Motion modules | `AnimateDiff::AnimateDiffModel` `animatediff.hpp:146` + hooks `unet.hpp:603-637, 652/680/717` — **already done** | AnimateAnyone ships its own `mm_sd_v15_v2`-compatible motion module; name prefix must be `model.diffusion_model.motion_module.` |
| CLIP image cond | `FrozenCLIPVisionEmbedder` `conditioner.hpp:561`, `get_clip_vision_output` `stable-diffusion.cpp:2164`, `IPAdapter::ImageProjModel` `ip_adapter.hpp:10` | feed result as `DiffusionParams::context` instead of text embeds; new `Conditioner` subclass or a `GenerationExtension` using `before_diffusion` (`pulid_extension.cpp:110`) |
| Frames | `animatediff_num_frames` machinery `stable-diffusion.cpp:243-244, 2726-2729, 5034-5047, 5366-5383, 6806-6846` | none |
| VAE / scheduler | `AutoEncoderKL`, `CompVisDenoiser` + scaled_linear betas, `ddim_trailing` | none |
| Registration | §5 checklist | `VERSION_ANIMATE_ANYONE` in `model.h`, string in `model_version_to_str`, detector in `model_loader.cpp:get_sd_version()`, branch in `stable-diffusion.cpp`, `sd_version_supports_video_generation` |
