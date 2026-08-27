#!/usr/bin/env python
"""Run the Moore-AnimateAnyone reference pose2img flow end-to-end on the task-1
fixture inputs (ref.png + pose.png), dumping the image and the per-step
trajectory (initial latent, per-step CFG'd noise_pred, per-step latents) so the
C++ port's generation loop (task 9) can be compared step by step.

Usage: /data/sdcpp-pixel-refs/venv/bin/python tools/aa/ref_generate.py [outdir]
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch

WEIGHTS_DIR = Path(os.environ.get("SDCPP_AA_WEIGHTS", "/data/sdcpp-pixel-refs/weights"))
FIXTURES_DIR = Path(os.environ.get("SDCPP_AA_FIXTURES", "/data/sdcpp-pixel-refs/fixtures"))
REF_REPO = Path(os.environ.get("SDCPP_AA_REF", "/data/sdcpp-pixel-refs/moore-animate-anyone"))
OUT_DIR = Path(sys.argv[1] if len(sys.argv) > 1 else "/data/sdcpp-pixel-refs/outputs/task9/ref_pipeline")
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(REF_REPO))

from omegaconf import OmegaConf  # noqa: E402
from PIL import Image  # noqa: E402
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection  # noqa: E402
from diffusers import AutoencoderKL, DDIMScheduler  # noqa: E402
from diffusers.image_processor import VaeImageProcessor  # noqa: E402

from src.models.unet_2d_condition import UNet2DConditionModel  # noqa: E402
from src.models.unet_3d import UNet3DConditionModel  # noqa: E402
from src.models.pose_guider import PoseGuider  # noqa: E402
from src.models.mutual_self_attention import ReferenceAttentionControl  # noqa: E402

torch.set_grad_enabled(False)
DEVICE = "cpu"
DTYPE = torch.float32
SEED = 42
STEPS = 25
CFG = 3.5
W = H = 512

sd15_unet_dir = WEIGHTS_DIR / "stable-diffusion-v1-5"
vae_dir = WEIGHTS_DIR / "sd-vae-ft-mse"
image_encoder_dir = WEIGHTS_DIR / "image_encoder"
aa_dir = WEIGHTS_DIR / "AnimateAnyone"

infer_config = OmegaConf.load(REF_REPO / "configs" / "inference" / "inference_v2.yaml")

vae = AutoencoderKL.from_pretrained(str(vae_dir)).to(DEVICE, dtype=DTYPE)
reference_unet = UNet2DConditionModel.from_pretrained(str(sd15_unet_dir), subfolder="unet").to(dtype=DTYPE, device=DEVICE)
denoising_unet = UNet3DConditionModel.from_pretrained_2d(
    str(sd15_unet_dir), str(aa_dir / "motion_module.pth"), subfolder="unet",
    unet_additional_kwargs=OmegaConf.to_container(infer_config.unet_additional_kwargs),
).to(dtype=DTYPE, device=DEVICE)
pose_guider = PoseGuider(320, block_out_channels=(16, 32, 96, 256)).to(dtype=DTYPE, device=DEVICE)
image_enc = CLIPVisionModelWithProjection.from_pretrained(str(image_encoder_dir)).to(dtype=DTYPE, device=DEVICE)
scheduler = DDIMScheduler(**OmegaConf.to_container(infer_config.noise_scheduler_kwargs))

denoising_unet.load_state_dict(torch.load(aa_dir / "denoising_unet.pth", map_location="cpu"), strict=False)
reference_unet.load_state_dict(torch.load(aa_dir / "reference_unet.pth", map_location="cpu"), strict=True)
pose_guider.load_state_dict(torch.load(aa_dir / "pose_guider.pth", map_location="cpu"), strict=True)
for m in (vae, reference_unet, denoising_unet, pose_guider, image_enc):
    m.eval()
print("models loaded")

ref_image = Image.open(FIXTURES_DIR / "ref.png").convert("RGB")
pose_image = Image.open(FIXTURES_DIR / "pose.png").convert("RGB")

# CLIP embeds (P-map section 5 step 3)
clip_image = CLIPImageProcessor().preprocess(ref_image.resize((224, 224)), return_tensors="pt").pixel_values
image_prompt_embeds = image_enc(clip_image.to(DEVICE)).image_embeds.unsqueeze(1)
uncond = torch.zeros_like(image_prompt_embeds)
encoder_hidden_states = torch.cat([uncond, image_prompt_embeds], dim=0)  # (2,1,768)

# Reference latents: VAE mean * 0.18215
ref_image_tensor = VaeImageProcessor(vae_scale_factor=8, do_convert_rgb=True).preprocess(ref_image, height=H, width=W)
ref_image_latents = vae.encode(ref_image_tensor.to(DEVICE)).latent_dist.mean * 0.18215  # (1,4,64,64)

# Pose feature
pose_np = np.asarray(pose_image).astype(np.float32) / 255.0
pose_t = torch.from_numpy(pose_np).permute(2, 0, 1).unsqueeze(0).unsqueeze(2)  # (1,3,1,H,W)
pose_fea = pose_guider(pose_t.to(DEVICE))  # (1,320,1,64,64)
pose_fea = torch.cat([pose_fea] * 2)

# Reference control (write/read, fusion_blocks full)
reference_control_writer = ReferenceAttentionControl(reference_unet, do_classifier_free_guidance=True, mode="write", batch_size=1, fusion_blocks="full")
reference_control_reader = ReferenceAttentionControl(denoising_unet, do_classifier_free_guidance=True, mode="read", batch_size=1, fusion_blocks="full")

scheduler.set_timesteps(STEPS)
timesteps = scheduler.timesteps
print("timesteps:", timesteps.tolist())

gen = torch.Generator(device="cpu").manual_seed(SEED)
latents = torch.randn((1, 4, H // 8, W // 8), generator=gen, device=DEVICE, dtype=DTYPE)
latents = latents * scheduler.init_noise_sigma
latents = latents.unsqueeze(2)  # (1,4,1,64,64)
np.save(OUT_DIR / "init_latent.npy", latents.numpy())

for i, t in enumerate(timesteps):
    if i == 0:
        reference_unet(
            ref_image_latents.repeat(2, 1, 1, 1),
            torch.zeros_like(t),
            encoder_hidden_states=encoder_hidden_states,
            return_dict=False,
        )
        reference_control_reader.update(reference_control_writer)

    latent_model_input = torch.cat([latents] * 2)
    latent_model_input = scheduler.scale_model_input(latent_model_input, t)
    noise_pred = denoising_unet(
        latent_model_input,
        t,
        encoder_hidden_states=encoder_hidden_states,
        pose_cond_fea=pose_fea,
        return_dict=False,
    )[0]
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    noise_pred_cfg = noise_pred_uncond + CFG * (noise_pred_text - noise_pred_uncond)
    latents = scheduler.step(noise_pred_cfg, t, latents, return_dict=False)[0]
    np.save(OUT_DIR / f"step{i:02d}_noise_pred.npy", noise_pred_cfg.numpy())
    np.save(OUT_DIR / f"step{i:02d}_latents.npy", latents.numpy())
    print(f"step {i} t={int(t)}: x mean {latents.mean():.5f} std {latents.std():.5f} "
          f"v mean {noise_pred_cfg.mean():.5f} std {noise_pred_cfg.std():.5f}", flush=True)

reference_control_reader.clear()
reference_control_writer.clear()

image = vae.decode(latents.squeeze(2) / 0.18215).sample
image = (image / 2 + 0.5).clamp(0, 1)
arr = (image[0].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
Image.fromarray(arr).save(OUT_DIR / "ref_pipeline_out.png")
print("saved", OUT_DIR / "ref_pipeline_out.png")
