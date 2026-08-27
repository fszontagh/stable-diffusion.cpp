#!/usr/bin/env python
"""Dump PyTorch golden fixtures for the sd.cpp AnimateAnyone port.

Runs the Moore-AnimateAnyone reference implementation (loaded exactly as
scripts/pose2vid.py does) on CPU, fp32, with a fixed seed, and saves every
intermediate tensor the C++ port needs to check itself against.

See docs/superpowers/notes/animate-anyone-pytorch-map.md (the "P-map"),
sections 2, 3, 5, 7, 8, for the exact reference behavior this script
reproduces.

Usage (inside the venv created per tools/aa/README.md):

    /data/sdcpp-pixel-refs/venv/bin/python tools/aa/dump_fixtures.py

Env overrides: SDCPP_AA_WEIGHTS, SDCPP_AA_FIXTURES, SDCPP_AA_REF,
SDCPP_AA_SPRITE_REF (see README.md).
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

SEED = 12580

WEIGHTS_DIR = Path(os.environ.get("SDCPP_AA_WEIGHTS", "/data/sdcpp-pixel-refs/weights"))
FIXTURES_DIR = Path(os.environ.get("SDCPP_AA_FIXTURES", "/data/sdcpp-pixel-refs/fixtures"))
REF_REPO = Path(os.environ.get("SDCPP_AA_REF", "/data/sdcpp-pixel-refs/moore-animate-anyone"))
SPRITE_REPO = Path(
    os.environ.get("SDCPP_AA_SPRITE_REF", "/data/sdcpp-pixel-refs/sprite-sheet-diffusion")
)

sys.path.insert(0, str(REF_REPO))

from omegaconf import OmegaConf  # noqa: E402
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection  # noqa: E402
from diffusers import AutoencoderKL, DDIMScheduler  # noqa: E402

from src.models.unet_2d_condition import UNet2DConditionModel  # noqa: E402
from src.models.unet_3d import UNet3DConditionModel  # noqa: E402
from src.models.pose_guider import PoseGuider  # noqa: E402
from src.models.mutual_self_attention import ReferenceAttentionControl  # noqa: E402

FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

torch.set_grad_enabled(False)
DEVICE = "cpu"
DTYPE = torch.float32

manifest = {}


def dump_context_windows_only():
    """--context-windows: dump context_windows.json only, from the reference
    src.pipelines.context.uniform() (context.py:15), and update just that
    entry in manifest.json. Does NOT load any model weights - this fixture is
    pure scheduling math (Task 10), independent of the heavy checkpoints the
    rest of this script needs.

    Matches exactly how pipeline_pose2vid_long.py's __call__ invokes the
    context scheduler for its real (non-discarded) context_queue: step=0,
    num_frames=latents.shape[2], context_frames=24, context_stride=1,
    context_overlap=4, closed_loop=True (the v2 defaults)."""
    from src.pipelines.context import uniform  # noqa: E402

    context_frames = 24
    context_stride = 1
    context_overlap = 4
    closed_loop = True
    step = 0
    num_steps = 25

    windows = {}
    for num_frames in (64, 32):
        windows[str(num_frames)] = list(
            uniform(
                step,
                num_steps,
                num_frames,
                context_frames,
                context_stride,
                context_overlap,
                closed_loop,
            )
        )

    out = {
        "params": {
            "step": step,
            "context_frames": context_frames,
            "context_stride": context_stride,
            "context_overlap": context_overlap,
            "closed_loop": closed_loop,
        },
        "windows": windows,
    }
    path = FIXTURES_DIR / "context_windows.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  wrote {path}")
    for nf, wins in windows.items():
        print(f"    num_frames={nf}: {len(wins)} window(s)")

    manifest_path = FIXTURES_DIR / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            full_manifest = json.load(f)
    else:
        full_manifest = {"other_files": {}}
    full_manifest.setdefault("other_files", {})["context_windows.json"] = (
        "context.py uniform() window lists for num_frames in {64,32}, step=0, "
        "v2 defaults (context_frames=24, context_stride=1, context_overlap=4, "
        "closed_loop=True)"
    )
    with open(manifest_path, "w") as f:
        json.dump(full_manifest, f, indent=2)
    print(f"  updated {manifest_path} (context_windows.json entry only)")


if "--context-windows" in sys.argv:
    sys.path.insert(0, str(REF_REPO))
    dump_context_windows_only()
    sys.exit(0)


def save_npy(name, tensor):
    arr = tensor.detach().to(torch.float32).cpu().numpy()
    path = FIXTURES_DIR / f"{name}.npy"
    np.save(path, arr)
    manifest[name] = {"shape": list(arr.shape), "dtype": str(arr.dtype)}
    print(f"  wrote {path}  shape={arr.shape}  dtype={arr.dtype}")
    return arr


# ---------------------------------------------------------------------------
# 0. Sample images (ref / pose / ref_pose)
# ---------------------------------------------------------------------------

# COCO-18 keypoint names, BGR colors, and limb connections, copied verbatim
# from Dataprocessing/handlabel.py (P-map section 6).
KEYPOINTS = [
    "nose", "neck", "right_shoulder", "right_elbow", "right_wrist",
    "left_shoulder", "left_elbow", "left_wrist", "right_hip",
    "right_knee", "right_ankle", "left_hip", "left_knee", "left_ankle",
    "right_eye", "left_eye", "right_ear", "left_ear",
]
KEYPOINT_COLORS_BGR = [
    (0, 0, 255), (1, 85, 255), (1, 170, 255), (0, 255, 255),
    (3, 255, 170), (0, 255, 86), (0, 255, 3), (85, 255, 3),
    (171, 255, 3), (255, 255, 3), (255, 170, 5), (255, 85, 0),
    (255, 0, 0), (255, 0, 84), (255, 0, 170), (255, 0, 255),
    (169, 0, 255), (85, 0, 255),
]
SKELETON_WITH_COLORS_BGR = [
    ((0, 1), (154, 0, 0)), ((1, 2), (0, 0, 153)), ((1, 5), (1, 51, 153)),
    ((1, 8), (153, 153, 0)), ((1, 11), (153, 153, 0)), ((2, 3), (0, 101, 153)),
    ((3, 4), (0, 153, 153)), ((5, 6), (0, 153, 101)), ((6, 7), (0, 153, 51)),
    ((8, 9), (51, 153, 0)), ((9, 10), (102, 153, 0)), ((11, 12), (153, 102, 0)),
    ((12, 13), (153, 51, 0)), ((0, 14), (153, 0, 51)), ((0, 15), (153, 0, 153)),
    ((14, 16), (102, 0, 153)), ((15, 17), (102, 0, 153)),
]
POINT_DIAMETER = 8


def bgr_to_rgb(c):
    b, g, r = c
    return (r, g, b)


def synth_pose_image(size=512, cx=256, cy=256, scale=1.0, seed_offset=0):
    """Render a simple standing COCO-18 skeleton on a black background,
    using the exact color scheme from handlabel.py. Not a real detector
    output - used only because the sprite-sheet-diffusion sample checkout
    carries no ground_truth/poses character data (see README.md)."""
    img = Image.new("RGB", (size, size), (0, 0, 0))
    draw = ImageDraw.Draw(img)

    rng = np.random.default_rng(SEED + seed_offset)
    jitter = rng.normal(0, 2.0, size=(18, 2))

    # A rough standing pose, in a 512x512 canvas, scaled/offset.
    base = {
        "nose": (0, -160), "neck": (0, -140),
        "right_shoulder": (-35, -140), "right_elbow": (-55, -90), "right_wrist": (-60, -40),
        "left_shoulder": (35, -140), "left_elbow": (55, -90), "left_wrist": (60, -40),
        "right_hip": (-20, 0), "right_knee": (-22, 80), "right_ankle": (-24, 160),
        "left_hip": (20, 0), "left_knee": (22, 80), "left_ankle": (24, 160),
        "right_eye": (-8, -170), "left_eye": (8, -170),
        "right_ear": (-16, -165), "left_ear": (16, -165),
    }
    pts = []
    for i, name in enumerate(KEYPOINTS):
        dx, dy = base[name]
        x = cx + dx * scale + jitter[i, 0]
        y = cy + dy * scale + jitter[i, 1]
        pts.append((x, y))

    for (a, b), color_bgr in SKELETON_WITH_COLORS_BGR:
        draw.line([pts[a], pts[b]], fill=bgr_to_rgb(color_bgr), width=POINT_DIAMETER)
    for i, (x, y) in enumerate(pts):
        r = POINT_DIAMETER / 2
        draw.ellipse(
            [x - r, y - r, x + r, y + r], fill=bgr_to_rgb(KEYPOINT_COLORS_BGR[i])
        )
    return img


def find_sample_character_data():
    """Look for real character/pose data under the sprite-sheet-diffusion
    sample checkout (P-map section 6 layout:
    data/custom/characters/<char>/motions/<motion>/{ground_truth,poses}/*.png).
    Returns (ref_img, pose_img, ref_pose_img) or None."""
    candidates = list(SPRITE_REPO.glob("**/ground_truth/*.png")) + list(
        SPRITE_REPO.glob("**/ground_truth/*.jpg")
    )
    if not candidates:
        return None
    gt_dir = sorted(candidates)[0].parent
    motion_dir = gt_dir.parent
    poses_dir = motion_dir / "poses"
    gt_files = sorted(gt_dir.glob("*"))
    pose_files = sorted(poses_dir.glob("*")) if poses_dir.exists() else []
    if not gt_files or not pose_files:
        return None
    ref_img = Image.open(gt_files[0]).convert("RGB").resize((512, 512))
    pose_img = Image.open(pose_files[0]).convert("RGB").resize((512, 512))
    ref_pose_img = pose_img
    return ref_img, pose_img, ref_pose_img


print("=== 0. Preparing sample images ===")
sample = find_sample_character_data()
if sample is not None:
    ref_image, pose_image, ref_pose_image = sample
    print("  using real sample data from sprite-sheet-diffusion checkout")
else:
    print("  NOTE: no ground_truth/poses sample data found under "
          f"{SPRITE_REPO} - synthesizing a placeholder character + COCO-18 "
          "skeleton pose per handlabel.py's color scheme instead.")
    ref_image = Image.new("RGB", (512, 512), (120, 140, 200))
    d = ImageDraw.Draw(ref_image)
    d.ellipse([176, 96, 336, 256], fill=(230, 200, 170))  # head
    d.rectangle([196, 256, 316, 420], fill=(90, 90, 160))  # torso
    pose_image = synth_pose_image(seed_offset=0)
    ref_pose_image = synth_pose_image(seed_offset=1)

ref_image.save(FIXTURES_DIR / "ref.png")
pose_image.save(FIXTURES_DIR / "pose.png")
ref_pose_image.save(FIXTURES_DIR / "ref_pose.png")
print(f"  wrote {FIXTURES_DIR / 'ref.png'}")
print(f"  wrote {FIXTURES_DIR / 'pose.png'}")
print(f"  wrote {FIXTURES_DIR / 'ref_pose.png'}")

# ---------------------------------------------------------------------------
# 1. Load models exactly like scripts/pose2vid.py (P-map section 5)
# ---------------------------------------------------------------------------

print("=== 1. Loading models ===")

sd15_unet_dir = WEIGHTS_DIR / "stable-diffusion-v1-5"
vae_dir = WEIGHTS_DIR / "sd-vae-ft-mse"
image_encoder_dir = WEIGHTS_DIR / "image_encoder"
aa_dir = WEIGHTS_DIR / "AnimateAnyone"

infer_config_path = REF_REPO / "configs" / "inference" / "inference_v2.yaml"
infer_config = OmegaConf.load(infer_config_path)

vae = AutoencoderKL.from_pretrained(str(vae_dir)).to(DEVICE, dtype=DTYPE)

reference_unet = UNet2DConditionModel.from_pretrained(
    str(sd15_unet_dir), subfolder="unet"
).to(dtype=DTYPE, device=DEVICE)

denoising_unet = UNet3DConditionModel.from_pretrained_2d(
    str(sd15_unet_dir),
    str(aa_dir / "motion_module.pth"),
    subfolder="unet",
    unet_additional_kwargs=OmegaConf.to_container(infer_config.unet_additional_kwargs),
).to(dtype=DTYPE, device=DEVICE)

pose_guider = PoseGuider(320, block_out_channels=(16, 32, 96, 256)).to(
    dtype=DTYPE, device=DEVICE
)

image_enc = CLIPVisionModelWithProjection.from_pretrained(str(image_encoder_dir)).to(
    dtype=DTYPE, device=DEVICE
)

sched_kwargs = OmegaConf.to_container(infer_config.noise_scheduler_kwargs)
scheduler = DDIMScheduler(**sched_kwargs)

# P-map section 7: strict flags per checkpoint.
denoising_unet.load_state_dict(
    torch.load(aa_dir / "denoising_unet.pth", map_location="cpu"), strict=False
)
reference_unet.load_state_dict(
    torch.load(aa_dir / "reference_unet.pth", map_location="cpu"), strict=True
)
pose_guider.load_state_dict(
    torch.load(aa_dir / "pose_guider.pth", map_location="cpu"), strict=True
)

reference_unet.eval()
denoising_unet.eval()
pose_guider.eval()
image_enc.eval()
vae.eval()

print("  models loaded")

# ---------------------------------------------------------------------------
# 2. pose_guider_a_out.npy
# ---------------------------------------------------------------------------

print("=== 2. PoseGuider forward ===")
torch.manual_seed(SEED)
# do_normalize=False (Moore) -> [0,1], see P-map section 3 / porting note 5.
pose_np = np.asarray(pose_image).astype(np.float32) / 255.0  # (H, W, 3)
pose_t = torch.from_numpy(pose_np).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
pose_t = pose_t.unsqueeze(2)  # (1, 3, 1, H, W) - InflatedConv3d expects b c f h w
pose_guider_a_out = pose_guider(pose_t.to(DEVICE, dtype=DTYPE))
save_npy("pose_guider_a_out", pose_guider_a_out)
assert pose_guider_a_out.shape == (1, 320, 1, 64, 64), pose_guider_a_out.shape

# ---------------------------------------------------------------------------
# 3. clip_embeds.npy
# ---------------------------------------------------------------------------

print("=== 3. CLIP image embeds ===")
clip_image_processor = CLIPImageProcessor()
clip_image = clip_image_processor.preprocess(
    ref_image.resize((224, 224)), return_tensors="pt"
).pixel_values
# Save the exact preprocessed [1,3,224,224] fp32 tensor fed to the encoder, so the
# C++ port's model-correctness check (sd-aa-test clip-embeds, STRICT check) can be
# gated purely on the model, independent of how closely its own C++
# image->resize->normalize preprocessing chain matches PIL/CLIPImageProcessor
# bit-for-bit (that chain gets its own, looser PREPROCESS CHAIN check).
save_npy("pixel_values", clip_image)
clip_image_embeds = image_enc(clip_image.to(DEVICE, dtype=image_enc.dtype)).image_embeds
image_prompt_embeds = clip_image_embeds.unsqueeze(1)  # (1, 1, 768)
uncond_image_prompt_embeds = torch.zeros_like(image_prompt_embeds)
cfg_image_prompt_embeds = torch.cat(
    [uncond_image_prompt_embeds, image_prompt_embeds], dim=0
)  # (2, 1, 768)
save_npy("clip_embeds", cfg_image_prompt_embeds)
assert cfg_image_prompt_embeds.shape == (2, 1, 768)

# ---------------------------------------------------------------------------
# 4. ref_latents.npy + ref_bank_00..15.npy
# ---------------------------------------------------------------------------

print("=== 4. Reference latents + write-mode bank ===")
ref_image_processor_mean = None  # not needed; VaeImageProcessor replicated inline below

from diffusers.image_processor import VaeImageProcessor  # noqa: E402

vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
ref_image_processor = VaeImageProcessor(
    vae_scale_factor=vae_scale_factor, do_convert_rgb=True
)
ref_image_tensor = ref_image_processor.preprocess(ref_image, height=512, width=512)
ref_image_tensor = ref_image_tensor.to(dtype=vae.dtype, device=vae.device)
ref_image_latents = vae.encode(ref_image_tensor).latent_dist.mean
ref_image_latents = ref_image_latents * 0.18215  # (1, 4, 64, 64)
save_npy("ref_latents", ref_image_latents)

reference_control_writer = ReferenceAttentionControl(
    reference_unet,
    do_classifier_free_guidance=True,
    mode="write",
    batch_size=1,
    fusion_blocks="full",
)
reference_control_reader = ReferenceAttentionControl(
    denoising_unet,
    do_classifier_free_guidance=True,
    mode="read",
    batch_size=1,
    fusion_blocks="full",
)

torch.manual_seed(SEED)
t_zero = torch.zeros((), dtype=torch.long)
reference_unet(
    ref_image_latents.repeat(2, 1, 1, 1),
    t_zero,
    encoder_hidden_states=cfg_image_prompt_embeds,
    return_dict=False,
)
reference_control_reader.update(reference_control_writer, dtype=DTYPE)

# Harvest the bank in the same descending-norm1-width order the hooking code
# uses (P-map section 2 / porting note 4).
from src.models.mutual_self_attention import torch_dfs  # noqa: E402
from src.models.attention import BasicTransformerBlock  # noqa: E402

writer_blocks = [
    m for m in torch_dfs(reference_unet) if isinstance(m, BasicTransformerBlock)
]
writer_blocks = sorted(writer_blocks, key=lambda x: -x.norm1.normalized_shape[0])
assert len(writer_blocks) == 16, len(writer_blocks)
for i, block in enumerate(writer_blocks):
    assert len(block.bank) == 1, f"block {i} bank has {len(block.bank)} entries"
    save_npy(f"ref_bank_{i:02d}", block.bank[0])

# ---------------------------------------------------------------------------
# 5. unet_step_f1 / unet_step_f8
# ---------------------------------------------------------------------------


def run_unet_step(video_length, tag):
    print(f"=== 5. Denoising UNet forward, F={video_length} ({tag}) ===")
    torch.manual_seed(SEED)
    latents = torch.randn(
        (1, 4, video_length, 64, 64), dtype=DTYPE, device=DEVICE
    )
    latent_model_input = torch.cat([latents] * 2, dim=0)  # CFG-doubled, (2,4,F,64,64)
    save_npy(f"unet_step_{tag}_in", latent_model_input)

    t = torch.tensor(999, dtype=torch.long)

    # pose_cond_fea: PoseGuider output, CFG-doubled, broadcast across frames.
    pose_fea_1frame = pose_guider_a_out  # (1, 320, 1, 64, 64)
    pose_fea = pose_fea_1frame.repeat(1, 1, video_length, 1, 1)
    pose_fea = torch.cat([pose_fea] * 2, dim=0)  # (2, 320, F, 64, 64)

    encoder_hidden_states = cfg_image_prompt_embeds  # (2, 1, 768)

    out = denoising_unet(
        latent_model_input,
        t,
        encoder_hidden_states=encoder_hidden_states,
        pose_cond_fea=pose_fea,
        return_dict=False,
    )[0]
    save_npy(f"unet_step_{tag}", out)
    assert out.shape == (2, 4, video_length, 64, 64), out.shape


run_unet_step(1, "f1")
run_unet_step(8, "f8")

reference_control_reader.clear()
reference_control_writer.clear()

# ---------------------------------------------------------------------------
# 6. sched_v2.json
# ---------------------------------------------------------------------------

print("=== 6. DDIM v-pred / zero-SNR / trailing scheduler ===")
sched2 = DDIMScheduler(**sched_kwargs)
sched2.set_timesteps(25)
timesteps = sched2.timesteps.tolist()
alphas_cumprod = sched2.alphas_cumprod.tolist()

torch.manual_seed(SEED)
x_t = torch.randn((1, 4, 64, 64), dtype=DTYPE)
torch.manual_seed(SEED + 1)
v = torch.randn((1, 4, 64, 64), dtype=DTYPE)
t0 = sched2.timesteps[0]
step_out = sched2.step(v, t0, x_t, return_dict=False)[0]

sched_dump = {
    "num_inference_steps": 25,
    "timesteps": [int(t) for t in timesteps],
    "alphas_cumprod": alphas_cumprod,
    "step_example": {
        "t": int(t0),
        "seed_x_t": SEED,
        "seed_v": SEED + 1,
        "x_t": x_t.numpy().tolist(),
        "v": v.numpy().tolist(),
        "prev_sample": step_out.numpy().tolist(),
    },
}
with open(FIXTURES_DIR / "sched_v2.json", "w") as f:
    json.dump(sched_dump, f)
print(f"  wrote {FIXTURES_DIR / 'sched_v2.json'}")
print(f"  {len(timesteps)} timesteps, {len(alphas_cumprod)} alphas_cumprod entries")

# ---------------------------------------------------------------------------
# 7. manifest.json
# ---------------------------------------------------------------------------

print("=== 7. manifest.json ===")
import diffusers  # noqa: E402
import transformers  # noqa: E402

full_manifest = {
    "seed": SEED,
    "versions": {
        "torch": torch.__version__,
        "diffusers": diffusers.__version__,
        "transformers": transformers.__version__,
    },
    "npy_files": manifest,
    "other_files": {
        "pose.png": "512x512 RGB, [0,255] uint8 PNG",
        "ref.png": "512x512 RGB, [0,255] uint8 PNG",
        "ref_pose.png": "512x512 RGB, [0,255] uint8 PNG",
        "sched_v2.json": "DDIM v-pred/zero-SNR/trailing scheduler dump",
    },
    "notes": {
        "sample_data": (
            "real"
            if sample is not None
            else "synthesized (no ground_truth/poses sample data found under "
            f"{SPRITE_REPO})"
        ),
    },
}
with open(FIXTURES_DIR / "manifest.json", "w") as f:
    json.dump(full_manifest, f, indent=2)
print(f"  wrote {FIXTURES_DIR / 'manifest.json'}")

print("\n=== Shape table ===")
for name, info in manifest.items():
    print(f"  {name:24s} {str(info['shape']):24s} {info['dtype']}")
print("\nDone.")
