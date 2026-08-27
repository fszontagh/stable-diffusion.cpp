# PyTorch → C++ Porting Spec: Moore-AnimateAnyone + Sprite-Sheet-Diffusion

Repos:
- `/data/sdcpp-pixel-refs/moore-animate-anyone` (primary; call it **M**)
- `/data/sdcpp-pixel-refs/sprite-sheet-diffusion` (wrapper; call it **S**)

---

## 1. ReferenceNet — `src/models/unet_2d_condition.py` + `unet_2d_blocks.py`

**It is a byte-for-byte copy of the diffusers 0.24-era SD1.5 `UNet2DConditionModel` with exactly one structural change: the output head is deleted.**

- `/data/sdcpp-pixel-refs/moore-animate-anyone/src/models/unet_2d_condition.py:645` — `self.conv_norm_out = None` (unconditionally overrides the GroupNorm created at :634)
- `:648-653` — `self.conv_out = nn.Conv2d(...)` is **commented out**. The module has no `conv_out` at all.
- `:1296-1299` — post-process block is commented out; forward returns the raw up-block output.
- `:1305-1308` — still returns `UNet2DConditionOutput(sample=sample)` where `sample` is the last up-block hidden state at 320ch full res. **This return value is discarded by all callers.**

Config: identical constructor signature to stock diffusers (`:161-220`), including `attention_type`, `addition_embed_type`, GLIGEN `position_net`, ControlNet residual args. None of it is used for SD1.5. Instantiated purely via `UNet2DConditionModel.from_pretrained(sd15_path, subfolder="unet")` — so it reads a stock SD1.5 `unet/config.json` (block_out_channels [320,640,1280,1280], cross_attention_dim 768, attention_head_dim 8, layers_per_block 2, in/out 4).

Forward signature: unchanged from diffusers (`:872-886`). **No `video_length`, no `pose_cond_fea`, no mode flag.** The reference features are not returned — they are captured as a side effect by monkey-patched attention blocks (§2).

`unet_2d_blocks.py` and `attention.py::BasicTransformerBlock` (`:12`, forward at `:178`) are verbatim diffusers copies. No hooks are baked in; the hooking is done at runtime.

**Practical consequence for the C++ port:** ReferenceNet = SD1.5 UNet, conv_in → down → mid → up, stopping before `conv_norm_out`/`conv_out`. You can drop `conv_norm_out.*` and `conv_out.*` weights entirely. The only thing you need out of it is the per-`BasicTransformerBlock` **post-`norm1` hidden state**, captured at each of the 16 transformer blocks.

---

## 2. Injection mechanism — `src/models/mutual_self_attention.py`

Class `ReferenceAttentionControl` (`:19`), adapted from MagicAnimate.

### What is hooked
`register_reference_hooks` (`:52`) does a `torch_dfs` over the UNet and collects every `BasicTransformerBlock` **or** `TemporalBasicTransformerBlock` (`:269-285`), then replaces `module.forward` with `hacked_basic_transformer_inner_forward` (`:290-299`). Only `attn1` (self-attention) behavior changes; `attn2` (cross-attn to CLIP) and `ff` are untouched.

Two scopes (`fusion_blocks`):
- `"midup"` → `mid_block` + `up_blocks` only
- `"full"` → whole UNet (down+mid+up). **All pipelines and both training stages use `"full"`** (`pipeline_pose2img.py:244,251`, `pipeline_pose2vid_long.py:~385,392`, `train_stage_1.py:317,324`).

Blocks are sorted by `-x.norm1.normalized_shape[0]` (`:286-288`) — descending channel width — so writer and reader lists pair up deterministically. Each gets `module.bank = []` and `module.attn_weight = i/len` (`:301-302`; `attn_weight` is never read).

### Write mode (ReferenceNet)
`:139-148`:
```python
self.bank.append(norm_hidden_states.clone())   # post-LayerNorm(norm1), PRE-attention
attn_output = self.attn1(norm_hidden_states, encoder_hidden_states=None, ...)
```
So the bank stores the **normalized hidden states**, not K/V projections. The ReferenceNet still runs its own normal self-attention. Shape written: `(b_ref, h*w, C)` where `b_ref = 2` under CFG (uncond+cond ref latents are both pushed through), `h*w` = resolution of that block (e.g. 64×64=4096 @320ch, …, 8×8=64 @1280ch for 512×512), C ∈ {320,640,1280}.

### Transfer
`update(writer)` (`:304-341`): re-collects `TemporalBasicTransformerBlock`s from the reader UNet and `BasicTransformerBlock`s from the writer UNet, sorts both by the same key, zips, and does `r.bank = [v.clone().to(dtype) for v in w.bank]`. Writer bank is **not** cleared (`:341` commented) — hence the `if i == 0` guard in the pipelines, otherwise the bank would grow every step.

### Read mode (denoising 3D UNet)
`:149-190`:
```python
bank_fea = [rearrange(d.unsqueeze(1).repeat(1, video_length, 1, 1), "b t l c -> (b t) l c") for d in self.bank]
modify_norm_hidden_states = torch.cat([norm_hidden_states] + bank_fea, dim=1)
hidden_states_uc = self.attn1(norm_hidden_states,
                              encoder_hidden_states=modify_norm_hidden_states) + hidden_states
```
So: **Q comes from the current (video) hidden states only; K and V come from `concat([self, reference], dim=seq)`.** This is spatial-attention concat, not a KV cache of projected tensors — the reference is re-projected through this block's own `to_k`/`to_v` each step. Sequence length doubles: query `L = h*w`, key/value `2L`.

Shapes at read: `norm_hidden_states` is `(b*f, L, C)`; each bank entry `(b_ref, L, C)` → repeated to `(b_ref*f, L, C)`; concat → `(b*f, 2L, C)`. `video_length` (f) comes down from `Transformer3DModel.forward` (`transformer_3d.py:119`).

### CFG handling in read mode
`:168-190`: if `do_classifier_free_guidance`, the **unconditional half is recomputed without reference** — `hidden_states_c[_uc_mask] = attn1(norm[_uc_mask], encoder_hidden_states=norm[_uc_mask]) + hidden_states[_uc_mask]`, i.e. plain self-attention. `_uc_mask` is `[True]*(N//2) + [False]*(N//2)` (`:171-179`), so index 0..N/2 = uncond. Note the constructor's default `uc_mask` (`:77-91`) hard-codes `*16` frames and is always overridden by the shape-mismatch branch in practice.

After attn1, read mode continues inline: cross-attn `attn2` with CLIP embeds (`:193-207`), FF (`:210`), then optional temporal attention with `(b f) d c -> (b d) f c` reshape (`:213-228`) — the latter is disabled (`unet_use_temporal_attention: false` in both inference configs).

### Timestep dependence
**None.** The bank is filled once at `i == 0` with `timestep = torch.zeros_like(t)` and reused unchanged for all denoising steps. `clear()` (`:343-365`) is called after the loop.

---

## 3. PoseGuider — `src/models/pose_guider.py` (Moore)

Full file is 57 lines. `PoseGuider(ModelMixin)` (`:12`):

| layer | op | in→out | k | s | p |
|---|---|---|---|---|---|
| `conv_in` | InflatedConv3d | 3 → boc[0] | 3 | 1 | 1 |
| `blocks.0` | InflatedConv3d | boc[0]→boc[0] | 3 | 1 | 1 |
| `blocks.1` | InflatedConv3d | boc[0]→boc[1] | 3 | **2** | 1 |
| `blocks.2` | InflatedConv3d | boc[1]→boc[1] | 3 | 1 | 1 |
| `blocks.3` | InflatedConv3d | boc[1]→boc[2] | 3 | **2** | 1 |
| `blocks.4` | InflatedConv3d | boc[2]→boc[2] | 3 | 1 | 1 |
| `blocks.5` | InflatedConv3d | boc[2]→boc[3] | 3 | **2** | 1 |
| `conv_out` | InflatedConv3d, **zero_module** | boc[3] → 320 | 3 | 1 | 1 |

- `block_out_channels` default `(16,32,64,128)` (`:17`) but **all real call sites use `(16, 32, 96, 256)`**: `scripts/pose2vid.py:70`, `app.py:66`, `train_stage_1.py:286`.
- `conditioning_embedding_channels = 320` (= UNet `block_out_channels[0]`).
- Activation: `F.silu` after `conv_in` and after **every** block, none after `conv_out` (`:47-57`).
- Zero-init: only `conv_out` via `zero_module` (`motion_module.py:15-19`).
- `InflatedConv3d` (`resnet.py:9-17`) = plain `nn.Conv2d` applied per-frame via `b c f h w -> (b f) c h w` and back. So weights are 2D conv weights.

**Input:** RGB pose skeleton image, 3 channels, **full image resolution** (same W×H as the latent's source image, e.g. 512×784), values in `[0,1]` — `cond_image_processor = VaeImageProcessor(..., do_normalize=False)` (`pipeline_pose2img.py:65-69`), so **no** [-1,1] rescale. Tensor `(b, 3, f, H, W)`. Total stride 8 → output `(b, 320, f, H/8, W/8)` = latent resolution.

**Where added:** `unet_3d.py:503-505`:
```python
sample = self.conv_in(sample)
if pose_cond_fea is not None:
    sample = sample + pose_cond_fea
```
i.e. added to the output of `conv_in`, before the first down block. Under CFG the pose feature is duplicated (`pipeline_pose2img.py:289-291`).

**Pretrain init** (`train_stage_1.py:288-295`): weights are seeded from ControlNet-openpose's `controlnet_cond_embedding.*` (excluding `conv_out`), key prefix stripped. That's why (16,32,96,256) is used — it matches `control_v11p_sd15_openpose`.

---

## 4. Motion module + 3D UNet

### `src/models/motion_module.py` (adapted from AnimateDiff)
- `get_motion_module` (`:34`) → `VanillaTemporalModule` (`:44`), only `"Vanilla"` supported.
- `VanillaTemporalModule` holds one `temporal_transformer` = `TemporalTransformer3DModel`; `attention_head_dim = in_channels // num_attention_heads // temporal_attention_dim_div` (`:62-64`); `proj_out` zero-initialized when `zero_initialize=True` (default) (`:72-75`).
- `TemporalTransformer3DModel` (`:94`): `GroupNorm(32, in_channels, eps=1e-6)` → `nn.Linear(in_channels, inner_dim)` `proj_in` → N × `TemporalTransformerBlock` → `nn.Linear(inner_dim, in_channels)` `proj_out`, plus residual (`:119-144`, `:172-182`). **Linear proj_in/proj_out, not conv.**
- Reshape (`:146-182`): asserts ndim==5, `b c f h w -> (b f) c h w`, GroupNorm, `permute(0,2,3,1).reshape(b, h*w, C)`, blocks, reshape back, `+ residual`, `(b f) c h w -> b c f h w`.
- `TemporalTransformerBlock` (`:185`): `attention_blocks` = one `VersatileAttention` per entry of `attention_block_types` (default two `Temporal_Self`), each with its own `nn.LayerNorm` in `norms`; then `ff` (`FeedForward`, geglu) + `ff_norm`. Residual around each (`:243-256`).
- `PositionalEncoding` (`:262-277`): standard sinusoidal, buffer `pe` shape `(1, max_len, d_model)`, added as `x + pe[:, :x.size(1)]`, applied **on the temporal axis**.
- `VersatileAttention(Attention)` (`:280`): `attention_mode` must be `"Temporal"`. Forward (`:351-388`): `d = hidden_states.shape[1]` (=h*w); `(b f) d c -> (b d) f c`; apply `pos_encoder`; attend over `f`; `(b d) f c -> (b f) d c`. `cross_attention_dim=None` for `Temporal_Self` → pure self-attention over frames.

### Placement in `unet_3d_blocks.py`
- `CrossAttnDownBlock3D` (`:313`): per layer `resnet → attn (Transformer3DModel) → motion_module` (`:477-486`), output appended to residuals after the MM.
- `DownBlock3D` (`:497`): `resnet → motion_module` (`:590-604`).
- `UNetMidBlock3DCrossAttn` (`:179`): `resnets[0]` then loop `attn → motion_module → resnet` (`:291-308`). Note MM sits **between** attn and resnet here, unlike the down blocks.
- `CrossAttnUpBlock3D` / `UpBlock3D`: same pattern as down (resnet → [attn] → motion_module).
- Enabled per-block by `use_motion_module and (res in motion_module_resolutions)` where `res = 2**i` down / `2**(3-i)` up (`unet_3d.py:130,156-158,201,244-245`); mid gated by `motion_module_mid_block` (`:184`).

### AnimateDiff compatibility
Yes — it matches **`mm_sd_v15_v2.ckpt`**. `configs/train/stage2.yaml:47` → `mm_path: './pretrained_weights/mm_sd_v15_v2.ckpt'`; README line 227 points at `guoyww/animatediff/mm_sd_v15_v2.ckpt`. `configs/inference/inference_v2.yaml` sets `motion_module_mid_block: true` and `temporal_position_encoding_max_len: 32` — the v2 layout (v1 config, `inference_v1.yaml`, uses `mid_block: false`, `max_len: 24`). Loading is a plain `state_dict.update(motion_state_dict)` then `load_state_dict(..., strict=False)` (`unet_3d.py:665-668`), so the checkpoint keys must already be in `down_blocks.X.motion_modules.Y.temporal_transformer.*` form.

### `unet_3d.py` reshaping
The 3D UNet is a **pseudo-3D inflation**: all convs are `InflatedConv3d` (Conv2d over `(b f)`), norms are `InflatedGroupNorm` when `use_inflated_groupnorm: true` (`resnet.py:20-28`), spatial attention is `Transformer3DModel` which does `b c f h w -> (b f) c h w` at entry (`transformer_3d.py:119-120`) and back at exit (`:196`), and only the motion modules mix across `f` (via `(b f) d c -> (b d) f c`). Cross-attn CLIP embeds are repeated `b n c -> (b f) n c` when batch mismatch (`transformer_3d.py:121-125`).

Head/tail: `conv_in` InflatedConv3d 4→320 (`unet_3d.py:229`); tail keeps `conv_norm_out` + `SiLU` + `conv_out` InflatedConv3d 320→4 (`:250-267`, applied `:591-593`) — unlike ReferenceNet.

Extra config keys vs. AnimateDiff: `mode` and `task_type="action"` (`unet_3d.py:83-84`) — when `task_type != "action"`, blocks get `name_index` strings and the `self_attention_additional_feats` dict path activates (used only by `lmks2vid`/face-reenact). For the pose2vid port, `task_type="action"` (default) → all names None → dead code.

---

## 5. Inference pipelines

### `src/pipelines/pipeline_pose2img.py` (single frame)
Components registered (`:52-59`): `vae`, `image_encoder`, `reference_unet`, `denoising_unet`, `pose_guider`, `scheduler`. `vae_scale_factor = 8`.

Order in `__call__` (`:192`):
1. `do_classifier_free_guidance = guidance_scale > 1.0` (`:216`); `batch_size = 1` (`:222`).
2. `scheduler.set_timesteps(num_inference_steps)`.
3. **CLIP image encoder** (`:225-237`): `CLIPImageProcessor()` default preprocessing on `ref_image.resize((224,224))` → `CLIPVisionModelWithProjection(...).image_embeds` → `(1, 768)` → `.unsqueeze(1)` → `(1, 1, 768)`. **Uncond = `torch.zeros_like(...)`** (a literal zero vector, not an empty-text embedding). CFG batch = `cat([uncond, cond])` → `(2, 1, 768)`. No text encoder, no tokenizer, no extra projection beyond CLIP's own `visual_projection`.
4. `ReferenceAttentionControl(reference_unet, mode="write", fusion_blocks="full")` and `(denoising_unet, mode="read", fusion_blocks="full")` (`:239-252`).
5. Latents `(1,4,H/8,W/8)` × `init_noise_sigma`, then `.unsqueeze(2)` → `(1,4,1,H/8,W/8)` (`:254-264`).
6. **Ref latents**: `VaeImageProcessor(do_convert_rgb=True)` (normalizes to [-1,1]) → `vae.encode(...).latent_dist.mean` (**mean, not sample**) `* 0.18215` (`:271-278`).
7. **Pose**: `cond_image_processor` (`do_normalize=False`, [0,1]) → `(1,3,1,H,W)` → `pose_guider` → `(1,320,1,H/8,W/8)` → duplicated ×2 for CFG (`:281-291`).
8. Denoise loop (`:296`):
   - **`if i == 0`**: `reference_unet(ref_latents.repeat(2,1,1,1), torch.zeros_like(t), encoder_hidden_states=image_prompt_embeds)` — **timestep = 0**, run once only (`:298-306`). Then `reference_control_reader.update(reference_control_writer)` (`:309`).
   - `latent_model_input = cat([latents]*2)`, `scale_model_input`.
   - `denoising_unet(x, t, encoder_hidden_states=image_prompt_embeds, pose_cond_fea=pose_fea)`.
   - CFG: `uncond + scale*(text - uncond)` (`:328-332`).
   - `scheduler.step`.
9. `reference_control_reader.clear()` + `writer.clear()` (`:347-348`).
10. Decode: `latents / 0.18215`, `b c f h w -> (b f) c h w`, per-frame `vae.decode`, `(x/2+0.5).clamp(0,1)` (`:102-115`).

### `src/pipelines/pipeline_pose2vid_long.py` (video, the one `scripts/pose2vid.py` uses)
Same structure plus sliding-window context:
- Latents `(1,4,F,H/8,W/8)`; pose guider run **once** on the whole `(1,3,F,H,W)` clip (no CFG dup at this stage).
- Per timestep: build `context_queue` via `uniform` scheduler from `src/pipelines/context.py:15` (`context_frames=24`, `context_stride=1`, `context_overlap=4`, `context_batch_size=1`, `closed_loop=True`, index wrap `e % num_frames`, jitter via `ordered_halving(step)` — note `step` is passed as `0` at both call sites, so the pad is 0).
- For each window: gather `latents[:,:,c]`, repeat ×2 for CFG, slice `pose_fea[:,:,c]`, run `denoising_unet`, accumulate into `noise_pred` and `counter`.
- CFG applied to `noise_pred / counter` (averaged overlaps), then `scheduler.step`.
- Reference UNet forward: identical, `if i == 0`, `torch.zeros_like(t)` (the alternative `t` is commented out).
- `interpolate_latents` optional slerp/lerp frame interpolation (`:290-333`, `pipelines/utils.py`).
- Note: this file also carries a dead `_encode_prompt` with a text tokenizer/text_encoder (`:185-289`) — never called.

### Scheduler / VAE
- Scheduler: `DDIMScheduler(**infer_config.noise_scheduler_kwargs)` (`scripts/pose2vid.py:77-78`).
  - **v2** (`configs/inference/inference_v2.yaml`): `beta_start 0.00085`, `beta_end 0.012`, `beta_schedule "linear"`, `clip_sample false`, `steps_offset 1`, `prediction_type "v_prediction"`, `rescale_betas_zero_snr True`, `timestep_spacing "trailing"`.
  - **v1**: same betas, `beta_schedule "linear"`, `steps_offset 1`, `clip_sample false`, **epsilon** prediction, no zero-SNR.
  - Training (`stage1.yaml`) uses `beta_schedule "scaled_linear"` with `num_train_timesteps 1000`; `train_stage_1.py` also forces `scaled_linear`.
- VAE: `AutoencoderKL.from_pretrained('./pretrained_weights/sd-vae-ft-mse')` (`stabilityai/sd-vae-ft-mse`), scaling factor hardcoded `0.18215`.
- Base UNet weights: `runwayml/stable-diffusion-v1-5` (`tools/download_weights.py:22`).
- Image encoder: `lambdalabs/sd-image-variations-diffusers` → `image_encoder/` (`tools/download_weights.py:39`). Config confirmed at `sprite-sheet-diffusion/ModelTraining/pretrained_model/image_encoder/config.json`: `CLIPVisionModelWithProjection`, hidden 1024, 24 layers, 16 heads, intermediate 4096, patch 14, image 224, `quick_gelu`, **projection_dim 768**, layer_norm_eps 1e-5.
- Finetuned AnimateAnyone weights: HF `patrolli/AnimateAnyone` → `denoising_unet.pth`, `motion_module.pth`, `pose_guider.pth`, `reference_unet.pth` (`tools/download_weights.py:88-97`).

---

## 6. Sprite-Sheet-Diffusion `ModelTraining` — deltas vs Moore

### File-level diff (models/)
`motion_module.py`, `transformer_2d.py`, `unet_2d_blocks.py`, `unet_2d_condition.py` are **byte-identical** to Moore's. `attention.py`, `mutual_self_attention.py`, `transformer_3d.py`, `unet_3d_blocks.py`, `unet_3d.py` differ only by S **lacking** Moore's later face-reenact additions (`name=`, `self_attention_additional_feats`, `mode`, `task_type`) — S is a fork of an **earlier** Moore commit. So Moore is a superset; port Moore.

### The one real architectural change: PoseGuider
`/data/sdcpp-pixel-refs/sprite-sheet-diffusion/ModelTraining/models/pose_guider.py:13` — a **completely different** module (AniPortrait-style, `PoseGuider(noise_latent_channels=320, use_ca=True)`). Moore's original is preserved verbatim as `models/pose_guider_org.py::PoseGuiderOrg`.

New PoseGuider:
- `conv_layers`: 8 × `Conv2d` + `BatchNorm2d` + `ReLU`, 3→3(k3s1) →16(k4**s2**) →16(k3) →32(k4s2) →32(k3) →64(k4s2) →64(k3) →128(k3s1). Stride 8 total.
- `final_proj`: `Conv2d(128, 320, k=1)`, **zero-initialized** (`:117-119`); other convs He-normal (`:108-116`).
- Learnable `self.scale = nn.Parameter(torch.ones(1) * 2)` (`:93`), multiplied after `final_proj`.
- `conv_layers_1` (320→320, s2), `conv_layers_2` (320→640, s2), `conv_layers_3` (640→1280, s2), `conv_layers_4` (1280→1280, s1) — each BN+ReLU.
- `cross_attn1..4`: local `Transformer2DModel` (defined in the same file, `:180`) with default `num_attention_heads=16, attention_head_dim=88` → inner_dim 1408, `cross_attention_dim=None`, conv proj_in/proj_out k1. Cross-attends pose features against **reference-pose** features.
- Forward `(x, ref_x)` returns a **list of 5 pyramid features** at 1/8, 1/16, 1/32, 1/64, 1/64 resolution with 320/320/640/1280/1280 channels.

Consumption changed in `models/unet_3d.py`: `sample = sample + pose_cond_fea[0]` after `conv_in` (`:487`), then `sample = sample + pose_cond_fea[block_count]` **after each down block** (`:509-511`). So it's a multi-scale ControlNet-like injection instead of Moore's single injection.

`ModelTraining/pipelines/pipeline_pose2img.py` and `pipeline_pose2vid_long_backup.py` add a `ref_pose_image` argument, call `self.pose_guider(pose_cond_tensor, ref_pose_tensor)`, and CFG-duplicate each pyramid level in a loop. They also flip `cond_image_processor` to **`do_normalize=True`** (pose images become [-1,1], unlike Moore).

Caveat: `ModelTraining/inference_img.py:103` instantiates `PoseGuiderOrg` but passes it to `Pose2ImagePipeline`, which calls it with two args — that path is broken as committed. The working entry point is `inference.py` (uses `PoseGuider(noise_latent_channels=320)` + `pipeline_pose2vid_long_backup`), and `run.sh` runs `pose2image.py` (stage-1 training).

### IP-Adapter
**Not integrated into the AnimateAnyone pipeline at all.** `ModelTraining/IP-Adapter/` is a vendored copy of tencent-ailab/IP-Adapter. A grep for `ip_adapter|IPAdapter|ImageProjModel|Resampler|set_ip_adapter` across `ModelTraining/` outside that directory returns **zero hits** — no reference in `models/`, `pipelines/`, `inference*.py`, `main.py`, `pose2image.py`, or any yaml.

The only project-authored file is `IP-Adapter/tutorial_train_adapted.py`, a **separate baseline** (their "ipadaptor" ablation in `experiment/results/`): stock SD1.5 UNet + `ControlNetModel` (openpose) + `ImageProjModel(cross_attention_dim=unet.cross_attention_dim, clip_embeddings_dim=image_encoder.projection_dim, clip_extra_context_tokens=4)` (`:133-137`); `IPAttnProcessor(hidden_size, cross_attention_dim, scale=1.0, num_tokens=4)` swapped into every `attn2` while `attn1` gets plain `AttnProcessor` (`:139-162`); pose fed as `controlnet_cond`, CLIP ref embeds as `encoder_hidden_states`. Saves `{"image_proj": ..., "ip_adapter": ...}` → `ip_adapter.bin` (`:243-246`). Defaults: resolution 512, lr 5e-5, wd 1e-2, bs 1, fp16. **You can ignore this for the port.**

### `configs/prompts/inference.yaml` (verbatim)
```yaml
pretrained_base_model_path: './pretrained_model/stable-diffusion-v1-5'
pretrained_vae_path: './pretrained_model/sd-vae-ft-mse'
image_encoder_path: './pretrained_model/image_encoder'
denoising_unet_path: "./pretrained_model/denoising_unet.pth"
reference_unet_path: "./pretrained_model/reference_unet.pth"
pose_guider_path: "./pretrained_model/pose_guider.pth"
motion_module_path: "./pretrained_model/motion_module.pth"
inference_config: "./configs/inference/inference_v2.yaml"
weight_dtype: 'fp16'
test_dir: 'data/custom/characters'
output_format: 'video'   # video or image
```
`inference.py` CLI defaults (`:36-47`): **W=512, H=512, L=None (=all pose frames), seed 42, cfg 3.5, steps 25, fps 8, fi_step 3**. Their `configs/inference/inference_v2.yaml` is byte-identical to Moore's v2 (v_prediction + zero-SNR + trailing).

### Pose format
OpenPose **COCO-18** skeleton images rendered as RGB PNGs, not DWPose-134. Evidence:
- `Dataprocessing/handlabel.py:12-49` — a Tkinter hand-labeling tool with 18 keypoints (`nose, neck, right_shoulder, …, left_ear`), 17 limb connections, fixed per-joint BGR colors, `point_diameter = 8`; writes `humanpose_N.png` next to `frame_N.png`.
- `ModelTraining/openpose/` and `Dataprocessing/openpose/` are ONNX OpenPose/DWPose detectors (`body.py`, `hand.py`, `face.py`, `animalpose.py`, `cv_ox_det.py`, `cv_ox_pose.py`).
- Directory layout consumed by `inference.py:148-173`:
  `data/custom/characters/<character>/motions/<motion>/{ground_truth/*.png, poses/*.png}`. Reference image = **first frame of `ground_truth/`**; reference pose = **first image of `poses/`**; all files sorted lexically, `Image.open(...).convert("RGB")`.
- Example config paths (`configs/prompts/animation.yaml`): `./configs/inference/ref_images/dog.png` + `./configs/inference/pose_images/dog_fall` (a directory of frames, not an mp4 — unlike Moore's `*_kps.mp4`). Other commented cases: `guile`, `Alice`, `kawa`, `Freeknight`, `William`, `cute_girl_jump`, `adventure_girl_dead`.
- Dataset JSON schema (`dataset/dataset_game.py:60-83`): `{characters: [{name, main_reference, main_reference_pose, motions: [{motion_name, poses: [...], ground_truth: [...]}]}]}`; sample_size `[512, 512]`; normalize mean/std 0.5.

### Google Drive weight link (readme.md, "Pretrained weight" section)
```
https://drive.google.com/drive/folders/1VxbOv5PE441NsNStQlmqbIw0iyY9Mn9L?usp=sharing
```
Folder ID: **`1VxbOv5PE441NsNStQlmqbIw0iyY9Mn9L`**

Additionally, `ModelTraining/pretrained_model/download_gdrive.sh` pulls three individual files by ID:
```
gdown 1vSexqqHmqRE5lXSxS_nOpJVtHswmxqwU
gdown 1wab2SnWKznqtgEgnoICbI_iioeHHajb9
gdown 1SYJj4IJTlYqNzodA2avbleBFbviUkXR7
```
(almost certainly denoising_unet / reference_unet / pose_guider). `download_animateanyone.sh` fetches the four Moore baseline `.pth` from `huggingface.co/patrolli/AnimateAnyone`; `download.sh` fetches SD1.5, sd-vae-ft-mse, `lambdalabs/sd-image-variations-diffusers/image_encoder`, wav2vec2-base-960h.

---

## 7. Checkpoint tensor naming (for the converter)

All four are **raw `torch.save(model.state_dict())` dicts — no wrapper key, no `state_dict`/`module.` prefix.** Written by `save_checkpoint()` in `train_stage_1.py:715-743` / `train_stage_2.py:715-743`, filenames `{prefix}-{global_step}.pth`.

| Checkpoint | Load target | Strict | Top-level key patterns |
|---|---|---|---|
| **`reference_unet.pth`** | `UNet2DConditionModel` (2D) — `scripts/pose2vid.py:88-90`, `strict=True` (default) | yes | `conv_in.{weight,bias}`, `time_embedding.linear_{1,2}.*`, `down_blocks.{0..3}.{resnets,attentions,downsamplers}.*`, `mid_block.{resnets,attentions}.*`, `up_blocks.{0..3}.{resnets,attentions,upsamplers}.*`. Attentions: `...attentions.{j}.{norm,proj_in,proj_out,transformer_blocks.0.{attn1,attn2,norm1,norm2,norm3,ff}}.*`. **No `conv_norm_out.*`, no `conv_out.*`** (module doesn't define them → strict load would fail if present; the published checkpoint omits them). Note `up_blocks.3.*` params were frozen in stage 1 (`train_stage_1.py:311-314`) but are still saved. |
| **`denoising_unet.pth`** | `UNet3DConditionModel` — `scripts/pose2vid.py:84-87`, **`strict=False`** | no | Same tree as above but 3D-block names: `conv_in.*`, `time_embedding.*`, `down_blocks.{0..3}.{resnets,attentions,motion_modules,downsamplers}.*`, `mid_block.{resnets,attentions,motion_modules}.*`, `up_blocks.{0..3}.{resnets,attentions,motion_modules,upsamplers}.*`, plus **`conv_norm_out.*` and `conv_out.*`** (present here). Attentions are `Transformer3DModel` → `...attentions.{j}.transformer_blocks.{k}.{attn1,attn2,attn_temp?,norm1,norm2,norm3,norm_temp?,ff}.*`. `strict=False` because the stage-1 checkpoint has no `motion_modules.*` and inflated-groupnorm/motion keys may mismatch. |
| **`pose_guider.pth`** | `PoseGuider` — `scripts/pose2vid.py:91-93`, strict | yes | `conv_in.{weight,bias}`, `blocks.{0..5}.{weight,bias}`, `conv_out.{weight,bias}`. Weights are 4-D Conv2d tensors (InflatedConv3d subclasses Conv2d): `conv_in.weight (16,3,3,3)`, `blocks.0 (16,16,3,3)`, `blocks.1 (32,16,3,3)`, `blocks.2 (32,32,3,3)`, `blocks.3 (96,32,3,3)`, `blocks.4 (96,96,3,3)`, `blocks.5 (256,96,3,3)`, `conv_out.weight (320,256,3,3)`. For **S**'s new PoseGuider instead: `conv_layers.{0,1,3,4,6,7,9,10,12,13,15,16,18,19}.*` (Conv2d/BatchNorm alternating in an `nn.Sequential`), `final_proj.{weight,bias}`, `conv_layers_{1,2,3,4}.{0,1,3,4}.*`, `cross_attn{1,2,3,4}.{norm,proj_in,proj_out,transformer_blocks.0.*}.*`, `scale`. |
| **`motion_module.pth`** | merged into the 3D UNet state dict inside `from_pretrained_2d` (`unet_3d.py:645-668`), then `load_state_dict(strict=False)` | no | Filtered by `if "motion_module" in key` (`train_stage_2.py:737-741`), so **every key contains the literal substring `motion_modules`**: `down_blocks.{i}.motion_modules.{j}.temporal_transformer.{norm,proj_in,proj_out}.*` and `...temporal_transformer.transformer_blocks.{k}.{attention_blocks.{0,1}.{to_q,to_k,to_v,to_out.0,pos_encoder.pe},norms.{0,1},ff.net.{0.proj,2},ff_norm}.*`, plus `mid_block.motion_modules.0.*` and `up_blocks.{i}.motion_modules.{j}.*`. The upstream AnimateDiff `mm_sd_v15_v2.ckpt` uses these exact key names — that's why the plain `state_dict.update()` merge works. `pos_encoder.pe` is a registered **buffer** of shape `(1, 32, C)` (v2) / `(1, 24, C)` (v1) and will be present in the file. |

Special-case for the C++ converter: **`pose_guider.pth` in ControlNet form.** `train_stage_1.py:289-295` maps `control_v11p_sd15_openpose`'s `controlnet_cond_embedding.<rest>` → `<rest>`, dropping any key containing `conv_out`. If you ever want to bootstrap from a ControlNet checkpoint, apply the same prefix strip.

---

## 8. Exact configs (consolidated)

**UNet (both ReferenceNet 2D and denoising 3D), from SD1.5 `unet/config.json`:**
- `in_channels 4`, `out_channels 4`, `sample_size 64`
- `block_out_channels [320, 640, 1280, 1280]`
- `layers_per_block 2`, `downsample_padding 1`, `mid_block_scale_factor 1`
- `down_block_types ["CrossAttnDownBlock2D"×3, "DownBlock2D"]` → 3D variants for the denoising UNet
- `up_block_types ["UpBlock2D", "CrossAttnUpBlock2D"×3]`
- `mid_block_type "UNetMidBlock2DCrossAttn"` → `"UNetMidBlock3DCrossAttn"`
- **`cross_attention_dim 768`**
- **`attention_head_dim 8`** (→ diffusers interprets as `num_attention_heads = 8`, so head_dim = C/8: 40/80/160/160)
- `norm_num_groups 32`, `norm_eps 1e-5`, `act_fn "silu"`
- `flip_sin_to_cos true`, `freq_shift 0`, `center_input_sample false`
- `use_linear_projection` absent → **false** (conv proj_in/proj_out in spatial transformers)
- `time_embed_dim = 320*4 = 1280`
- (rev_animated, S's alternate base at `pretrained_model/rev_animated/unet/config.json`, has identical dims.)

**3D-specific (`configs/inference/inference_v2.yaml`):**
- `use_inflated_groupnorm: true`
- `unet_use_cross_frame_attention: false`, `unet_use_temporal_attention: false`
- `use_motion_module: true`, `motion_module_resolutions: [1,2,4,8]` (= all four stages), `motion_module_mid_block: true`, `motion_module_decoder_only: false`
- `motion_module_type: Vanilla`
- `motion_module_kwargs`: `num_attention_heads 8`, `num_transformer_block 1`, `attention_block_types ["Temporal_Self","Temporal_Self"]`, `temporal_position_encoding true`, `temporal_position_encoding_max_len 32`, `temporal_attention_dim_div 1`
  - → temporal `attention_head_dim = C/8` (40/80/160/160), `inner_dim = C`, temporal GroupNorm 32 groups eps 1e-6.

**PoseGuider:** `conditioning_embedding_channels=320`, `conditioning_channels=3`, `block_out_channels=(16,32,96,256)`.

**CLIP image encoder:** `lambdalabs/sd-image-variations-diffusers/image_encoder`, `CLIPVisionModelWithProjection`, ViT-L/14-336-arch-at-224: hidden 1024, 24 layers, 16 heads, intermediate 4096, patch 14, image 224, `quick_gelu`, ln_eps 1e-5, **projection_dim 768**. Preprocess: `CLIPImageProcessor()` defaults (resize 224, center-crop 224, CLIP mean/std), fed `ref_image.resize((224,224))`. Output used: `.image_embeds` → `(b, 768)` → `unsqueeze(1)` → `(b, 1, 768)`. **Sequence length 1.** Uncond = zeros.

**VAE:** `stabilityai/sd-vae-ft-mse`, `vae_scale_factor 8`, scaling `0.18215` hardcoded, ref image encoded with `.latent_dist.mean`.

**Resolutions & frame counts:**
- Moore inference default: `-W 512 -H 784 -L 24` (`scripts/pose2vid.py:30-33`); README example `-W 512 -H 784 -L 64`; gradio defaults `512×768`, 24 frames, 25 steps, cfg 3.5 (`app.py:37-41`); README results at 512×768 and 512×512.
- Moore stage 1 train: **768×768**, bs 4, 30k steps, lr 1e-5, `sample_margin 30`.
- Moore stage 2 train: **512×512**, bs 1, `n_sample_frames 24`, `sample_rate 4`, 10k steps, 8-bit Adam.
- Sprite inference: **512×512**, L = all pose frames, 25 steps, cfg 3.5, fps 8.
- Sprite stage 1/2 train: `sample_size [512, 512]`, bs 4 / 1, 300k / 40k steps.
- Context window (video): `context_frames 24`, `context_stride 1`, `context_overlap 4`, `context_batch_size 1`, schedule `"uniform"`.
- Common training: `num_train_timesteps 1000`, `uncond_ratio 0.1` (0.0 in Sprite stage 2), `noise_offset 0.05`, `snr_gamma 5.0`, `enable_zero_snr true`, seed 12580.

---

## Porting notes / gotchas

1. **ReferenceNet has no output head.** Don't allocate `conv_norm_out`/`conv_out` for it; its "output" is the 16 banked `norm1` activations.
2. **Reference forward runs exactly once, at t=0**, with the CFG-doubled ref latent and the CFG-doubled CLIP embedding. Both halves of the bank are identical in content except that the ReferenceNet's own cross-attn saw zeros for the uncond half — so the two halves genuinely differ. The bank is then broadcast across frames on read.
3. **Read-mode attention is concat-KV with doubled sequence length**, and the uncond half of the batch is computed with *plain* self-attention (no reference). If your C++ attention kernel batches, you need this split.
4. **Block pairing is by descending `norm1` width**, not by module path. With `fusion_blocks="full"` and SD1.5 that's 16 blocks; a stable sort matters (Python's `sorted` is stable, so ties break by DFS order. CORRECTION 2026-08-27, verified during the C++ port: torch_dfs order is down_blocks -> up_blocks -> mid_block, because diffusers registers the up_blocks ModuleList at unet_2d_condition.py:456 BEFORE mid_block at :531; confirmed empirically by ref_bank_05 = (2, 64, 1280), the only 8x8 mid bank, in the fixture manifest).
5. **Pose images are NOT normalized to [-1,1]** in Moore (`do_normalize=False`); they are in Sprite (`do_normalize=True`). Getting this wrong silently degrades output.
6. **v_prediction + rescale_betas_zero_snr + trailing timestep spacing** is the v2 default — not the usual SD1.5 epsilon/leading. This changes both the scheduler math and the `set_timesteps` grid.
7. **Everything 3D is Conv2d-over-(b·f)** except the motion modules. A C++ port can keep 2D kernels throughout and only add a temporal-axis transpose for the motion modules.
8. `unet_3d.py`'s `mode` / `task_type` / `self_attention_additional_feats` machinery is dead code for pose-driven animation (`task_type="action"` default) — skip it.
