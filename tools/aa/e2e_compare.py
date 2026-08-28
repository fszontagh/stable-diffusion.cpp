#!/usr/bin/env python
"""Task 11: end-to-end parity harness for AnimateAnyone vs the PyTorch
(Moore-AnimateAnyone) reference.

Drives the PyTorch Pose2VideoPipeline (pipeline_pose2vid_long.py, v2 config
== zero-SNR trailing DDIM v-prediction, motion modules ON) over a short real
pose clip, dumps its initial latent (torch.randn draw, pre-scheduler-scale)
as a .npy the C++ CLI can consume via SDCPP_AA_DEBUG_INIT_LATENT, invokes the
C++ CLI (sd-cli -M vid_gen) with the identical ref image / pose frames /
seed / steps / cfg, and reports per-frame SSIM between the two outputs.

Bit-parity is NOT the bar (FA precision on GPU, float accumulation order
differ from CPU torch) - the per-module fixtures (tools/aa/dump_fixtures.py
+ examples/aa_test) already pin exactness at the module level. This script
checks *orchestration* parity: same weights, same schedule, same initial
noise, does the whole pipeline converge to visually the same clip. The bar
is mean per-frame SSIM >= 0.85.

Usage:
    SDCPP_AA_WEIGHTS=/data/sdcpp-pixel-refs/weights \\
    /data/sdcpp-pixel-refs/venv/bin/python tools/aa/e2e_compare.py \\
        [--frames 6] [--steps 10] [--seed 42] [--cfg 3.5] [--width 512] [--height 512] \\
        [--skip-python] [--skip-cpp]

Both stages are skippable (--skip-python / --skip-cpp) so a prior run's
frames can be re-scored, or the C++ side re-run alone after a rebuild,
without repeating the (slow, CPU-only) PyTorch pass.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REF_REPO = Path(os.environ.get("SDCPP_AA_REF", "/data/sdcpp-pixel-refs/moore-animate-anyone"))
WEIGHTS_DIR = Path(os.environ.get("SDCPP_AA_WEIGHTS", "/data/sdcpp-pixel-refs/weights"))
REPO_ROOT = Path(__file__).resolve().parents[2]
SD_CLI = REPO_ROOT / "build" / "bin" / "sd-cli"

DEFAULT_REF_IMAGE = Path("/data/sdcpp-pixel-refs/e2e/inputs/ref.png")
DEFAULT_POSE_DIR = Path("/data/sdcpp-pixel-refs/e2e/inputs/pose_512_f6")
DEFAULT_OUT_DIR = Path("/data/sdcpp-pixel-refs/outputs/task11")

VRAM_LIMIT_MIB = 1500
VRAM_POLL_SECS = 60


def gpu_gate():
    """GPU rule: block (polling every 60s, in-turn) while foreign VRAM use
    exceeds 1.5 GB, so this run doesn't contend for VRAM with another job."""
    while True:
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-compute-apps=used_memory", "--format=csv,noheader,nounits"],
                text=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            print(f"[gpu_gate] nvidia-smi unavailable ({exc}); proceeding without the gate")
            return
        used = sum(int(x) for x in out.split() if x.strip().isdigit())
        if used <= VRAM_LIMIT_MIB:
            print(f"[gpu_gate] foreign VRAM use {used} MiB <= {VRAM_LIMIT_MIB} MiB, proceeding")
            return
        print(f"[gpu_gate] foreign VRAM use {used} MiB > {VRAM_LIMIT_MIB} MiB, waiting {VRAM_POLL_SECS}s...")
        time.sleep(VRAM_POLL_SECS)


def run_python_reference(args, out_dir: Path) -> dict:
    sys.path.insert(0, str(REF_REPO))
    import torch
    from omegaconf import OmegaConf
    from PIL import Image
    from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
    from diffusers import AutoencoderKL, DDIMScheduler

    from src.models.unet_2d_condition import UNet2DConditionModel
    from src.models.unet_3d import UNet3DConditionModel
    from src.models.pose_guider import PoseGuider
    from src.pipelines.pipeline_pose2vid_long import Pose2VideoPipeline

    torch.set_grad_enabled(False)
    device = "cpu"
    dtype = torch.float32

    sd15_unet_dir = WEIGHTS_DIR / "stable-diffusion-v1-5"
    vae_dir = WEIGHTS_DIR / "sd-vae-ft-mse"
    image_encoder_dir = WEIGHTS_DIR / "image_encoder"
    aa_dir = WEIGHTS_DIR / "AnimateAnyone"
    infer_config = OmegaConf.load(REF_REPO / "configs" / "inference" / "inference_v2.yaml")

    print("[python] loading models (CPU, fp32)...")
    t0 = time.time()
    vae = AutoencoderKL.from_pretrained(str(vae_dir)).to(device, dtype=dtype)
    reference_unet = UNet2DConditionModel.from_pretrained(str(sd15_unet_dir), subfolder="unet").to(dtype=dtype, device=device)
    denoising_unet = UNet3DConditionModel.from_pretrained_2d(
        str(sd15_unet_dir), str(aa_dir / "motion_module.pth"), subfolder="unet",
        unet_additional_kwargs=OmegaConf.to_container(infer_config.unet_additional_kwargs),
    ).to(dtype=dtype, device=device)
    pose_guider = PoseGuider(320, block_out_channels=(16, 32, 96, 256)).to(dtype=dtype, device=device)
    image_enc = CLIPVisionModelWithProjection.from_pretrained(str(image_encoder_dir)).to(dtype=dtype, device=device)
    scheduler = DDIMScheduler(**OmegaConf.to_container(infer_config.noise_scheduler_kwargs))

    denoising_unet.load_state_dict(torch.load(aa_dir / "denoising_unet.pth", map_location="cpu"), strict=False)
    reference_unet.load_state_dict(torch.load(aa_dir / "reference_unet.pth", map_location="cpu"), strict=True)
    pose_guider.load_state_dict(torch.load(aa_dir / "pose_guider.pth", map_location="cpu"), strict=True)
    for m in (vae, reference_unet, denoising_unet, pose_guider, image_enc):
        m.eval()
    print(f"[python] models loaded in {time.time() - t0:.1f}s")

    pipe = Pose2VideoPipeline(
        vae=vae, image_encoder=image_enc, reference_unet=reference_unet,
        denoising_unet=denoising_unet, pose_guider=pose_guider, scheduler=scheduler,
    ).to(device, dtype=dtype)

    # Capture the exact initial latent (post torch.randn, pre/post
    # scheduler.init_noise_sigma scale - v2's init_noise_sigma is 1.0 so
    # this IS the raw randn draw) that prepare_latents() draws inside
    # __call__(), so the C++ run can be seeded with the identical tensor
    # rather than relying on two different RNGs to agree.
    captured = {}
    orig_prepare_latents = pipe.prepare_latents

    def capturing_prepare_latents(*a, **kw):
        latents = orig_prepare_latents(*a, **kw)
        captured["init_latent"] = latents.detach().clone().cpu().numpy()
        return latents

    pipe.prepare_latents = capturing_prepare_latents

    # Optional: dump the per-step trajectory (the CFG'd v-pred model output
    # fed into scheduler.step, and the resulting prev_sample latents) for
    # step-by-step divergence analysis against the C++ side's
    # SDCPP_AA_DEBUG_STEP_LATENTS dump. File naming matches
    # ref_generate.py's stepNN_{noise_pred,latents}.npy convention and the
    # C++ writer's stepNN_{vpred,latents}.npy.
    step_dir = None
    if args.dump_steps:
        step_dir = out_dir / "python_steps"
        step_dir.mkdir(parents=True, exist_ok=True)
        step_counter = {"i": 0}
        orig_scheduler_step = pipe.scheduler.step

        def capturing_scheduler_step(model_output, timestep, sample, *a, **kw):
            i = step_counter["i"]
            np.save(step_dir / f"step{i:02d}_vpred.npy", model_output.detach().cpu().numpy().astype(np.float32))
            result = orig_scheduler_step(model_output, timestep, sample, *a, **kw)
            np.save(step_dir / f"step{i:02d}_latents.npy", result.prev_sample.detach().cpu().numpy().astype(np.float32))
            step_counter["i"] += 1
            return result

        pipe.scheduler.step = capturing_scheduler_step

    ref_image = Image.open(args.ref_image).convert("RGB")
    pose_files = sorted(Path(args.pose_dir).glob("*.png"))[: args.frames]
    if len(pose_files) < args.frames:
        raise SystemExit(f"only {len(pose_files)} pose frames in {args.pose_dir}, need {args.frames}")
    pose_images = [Image.open(p).convert("RGB") for p in pose_files]

    generator = torch.Generator(device="cpu").manual_seed(args.seed)

    print(f"[python] running pipeline: frames={args.frames} steps={args.steps} "
          f"cfg={args.cfg} size={args.width}x{args.height} seed={args.seed}")
    t0 = time.time()
    videos = pipe(
        ref_image,
        pose_images,
        width=args.width,
        height=args.height,
        video_length=args.frames,
        num_inference_steps=args.steps,
        guidance_scale=args.cfg,
        generator=generator,
    ).videos  # (1, 3, F, H, W) float in [0,1], torch tensor (output_type="tensor")
    elapsed = time.time() - t0
    print(f"[python] pipeline done in {elapsed:.1f}s")

    frames_dir = out_dir / "python_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    arr = videos[0].cpu().numpy() if hasattr(videos, "cpu") else videos[0]  # (3, F, H, W)
    for f in range(args.frames):
        frame = (arr[:, f].transpose(1, 2, 0) * 255.0).round().clip(0, 255).astype(np.uint8)
        Image.fromarray(frame).save(frames_dir / f"frame_{f:02d}.png")

    init_latent_path = out_dir / "init_latent.npy"
    np.save(init_latent_path, captured["init_latent"].astype(np.float32))
    print(f"[python] saved {args.frames} frames to {frames_dir}, init latent to {init_latent_path} "
          f"(shape {captured['init_latent'].shape})")

    result = {
        "wall_time_s": elapsed,
        "frames_dir": str(frames_dir),
        "init_latent_path": str(init_latent_path),
        "init_latent_shape": list(captured["init_latent"].shape),
    }
    if step_dir is not None:
        result["step_dir"] = str(step_dir)
    return result


def run_cpp(args, out_dir: Path, init_latent_path: Path, step_dir: Path = None) -> dict:
    if not SD_CLI.exists():
        raise SystemExit(f"C++ CLI not found at {SD_CLI}; build it first (ninja sd-cli)")

    gpu_gate()

    cpp_frames_dir = out_dir / "cpp_frames"
    cpp_frames_dir.mkdir(parents=True, exist_ok=True)
    out_pattern = cpp_frames_dir / "frame_%02d.png"

    aa_dir = WEIGHTS_DIR / "AnimateAnyone"
    cmd = [
        str(SD_CLI), "-M", "vid_gen",
        "--diffusion-model", str(aa_dir / "denoising_unet.pth"),
        "--motion-module", str(aa_dir / "motion_module.pth"),
        "--reference-net", str(aa_dir / "reference_unet.pth"),
        "--pose-guider", str(aa_dir / "pose_guider.pth"),
        "--clip_vision", str(WEIGHTS_DIR / "image_encoder" / "pytorch_model.bin"),
        "--vae", str(WEIGHTS_DIR / "sd-vae-ft-mse" / "diffusion_pytorch_model.safetensors"),
        "--type", "f16", "--offload-to-cpu", "--diffusion-fa",
        "-r", str(args.ref_image),
        "--pose-dir", str(args.pose_dir),
        "-W", str(args.width), "-H", str(args.height),
        "--steps", str(args.steps), "--cfg-scale", str(args.cfg),
        "-s", str(args.seed), "--fps", "8",
        "-o", str(out_pattern),
    ]
    env = dict(os.environ)
    env["SDCPP_AA_DEBUG_INIT_LATENT"] = str(init_latent_path)
    if step_dir is not None:
        step_dir.mkdir(parents=True, exist_ok=True)
        env["SDCPP_AA_DEBUG_STEP_LATENTS"] = str(step_dir)

    print("[cpp] command:")
    print("  " + " ".join(cmd))
    print(f"[cpp] SDCPP_AA_DEBUG_INIT_LATENT={init_latent_path}")
    if step_dir is not None:
        print(f"[cpp] SDCPP_AA_DEBUG_STEP_LATENTS={step_dir}")

    t0 = time.time()
    log_path = out_dir / "cpp_run.log"
    with open(log_path, "w") as log_f:
        proc = subprocess.run(cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT)
    elapsed = time.time() - t0
    print(f"[cpp] exit code {proc.returncode} in {elapsed:.1f}s (log: {log_path})")
    if proc.returncode != 0:
        raise SystemExit(f"C++ CLI run failed (exit {proc.returncode}); see {log_path}")

    result = {
        "wall_time_s": elapsed,
        "frames_dir": str(cpp_frames_dir),
        "command": cmd,
        "log_path": str(log_path),
    }
    if step_dir is not None:
        result["step_dir"] = str(step_dir)
    return result


def compute_step_divergence(args, python_step_dir: Path, cpp_step_dir: Path) -> dict:
    """Rel-L2 divergence per denoising step, for both the CFG'd v-pred model
    output (pre-scheduler-step) and the resulting latents (post-step). Also
    reports the step-0 vpred divergence separately since that isolates
    per-step INPUT-assembly bugs from accumulation: with the injected
    identical initial latent, step 0's v-pred is a function of ONLY the
    (supposedly identical) t=999 inputs - banks, pose feature, clip embeds,
    latent - so a large step-0 vpred delta means the divergence is already
    present before any DDIM accumulation happens."""

    def rel_l2(a, b):
        a = a.astype(np.float64).ravel()
        b = b.astype(np.float64).ravel()
        denom = np.linalg.norm(a)
        if denom == 0:
            denom = 1.0
        return float(np.linalg.norm(a - b) / denom)

    rows = []
    for i in range(args.steps):
        py_vpred_p = python_step_dir / f"step{i:02d}_vpred.npy"
        cp_vpred_p = cpp_step_dir / f"step{i:02d}_vpred.npy"
        py_lat_p = python_step_dir / f"step{i:02d}_latents.npy"
        cp_lat_p = cpp_step_dir / f"step{i:02d}_latents.npy"
        if not (py_vpred_p.exists() and cp_vpred_p.exists() and py_lat_p.exists() and cp_lat_p.exists()):
            print(f"[step-divergence] missing dump(s) for step {i}, stopping at {i} steps")
            break
        py_vpred = np.load(py_vpred_p)
        cp_vpred = np.load(cp_vpred_p)
        py_lat = np.load(py_lat_p)
        cp_lat = np.load(cp_lat_p)
        row = {
            "step": i,
            "vpred_rel_l2": rel_l2(py_vpred, cp_vpred),
            "latents_rel_l2": rel_l2(py_lat, cp_lat),
        }
        rows.append(row)
        print(f"[step-divergence] step {i}: vpred rel L2 {row['vpred_rel_l2']:.6e}, "
              f"latents rel L2 {row['latents_rel_l2']:.6e}")

    return {"rows": rows}


def compute_ssim(args, python_frames_dir: Path, cpp_frames_dir: Path, out_dir: Path) -> dict:
    from PIL import Image
    from skimage.metrics import structural_similarity as ssim

    per_frame = []
    for f in range(args.frames):
        py_path = python_frames_dir / f"frame_{f:02d}.png"
        cpp_path = cpp_frames_dir / f"frame_{f:02d}.png"
        if not py_path.exists() or not cpp_path.exists():
            raise SystemExit(f"missing frame {f}: {py_path} exists={py_path.exists()}, "
                              f"{cpp_path} exists={cpp_path.exists()}")
        py_im = np.asarray(Image.open(py_path).convert("RGB"))
        cpp_im = np.asarray(Image.open(cpp_path).convert("RGB"))
        if py_im.shape != cpp_im.shape:
            cpp_im = np.asarray(Image.open(cpp_path).convert("RGB").resize(
                (py_im.shape[1], py_im.shape[0])))
        score = ssim(py_im, cpp_im, channel_axis=-1, data_range=255)
        per_frame.append(score)
        print(f"[ssim] frame {f}: {score:.4f}")

    mean_ssim = float(np.mean(per_frame))
    worst_idx = int(np.argmin(per_frame))
    print(f"[ssim] mean over {args.frames} frames: {mean_ssim:.4f} (worst: frame {worst_idx} = {per_frame[worst_idx]:.4f})")
    return {
        "per_frame_ssim": [float(x) for x in per_frame],
        "mean_ssim": mean_ssim,
        "worst_frame": worst_idx,
        "worst_ssim": float(per_frame[worst_idx]),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cfg", type=float, default=3.5)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--ref-image", type=Path, default=DEFAULT_REF_IMAGE)
    ap.add_argument("--pose-dir", type=Path, default=DEFAULT_POSE_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--skip-python", action="store_true", help="reuse existing python_frames/init_latent.npy")
    ap.add_argument("--skip-cpp", action="store_true", help="reuse existing cpp_frames/")
    ap.add_argument("--dump-steps", action="store_true",
                     help="dump per-step v-pred + latents on both sides and report per-step rel L2 "
                          "divergence (discriminates chaotic-amplification vs a systematic orchestration delta)")
    args = ap.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "settings": {
            "frames": args.frames, "steps": args.steps, "seed": args.seed, "cfg": args.cfg,
            "width": args.width, "height": args.height,
            "ref_image": str(args.ref_image), "pose_dir": str(args.pose_dir),
        },
    }

    if args.skip_python:
        python_frames_dir = out_dir / "python_frames"
        init_latent_path = out_dir / "init_latent.npy"
        if not init_latent_path.exists():
            raise SystemExit(f"--skip-python given but {init_latent_path} missing")
        report["python"] = {"skipped": True, "frames_dir": str(python_frames_dir),
                             "init_latent_path": str(init_latent_path)}
    else:
        py_result = run_python_reference(args, out_dir)
        report["python"] = py_result
        python_frames_dir = Path(py_result["frames_dir"])
        init_latent_path = Path(py_result["init_latent_path"])

    cpp_step_dir = out_dir / "cpp_steps" if args.dump_steps else None
    if args.skip_cpp:
        cpp_frames_dir = out_dir / "cpp_frames"
        report["cpp"] = {"skipped": True, "frames_dir": str(cpp_frames_dir)}
    else:
        cpp_result = run_cpp(args, out_dir, init_latent_path, step_dir=cpp_step_dir)
        report["cpp"] = cpp_result
        cpp_frames_dir = Path(cpp_result["frames_dir"])

    ssim_result = compute_ssim(args, python_frames_dir, cpp_frames_dir, out_dir)
    report["ssim"] = ssim_result
    report["pass"] = ssim_result["mean_ssim"] >= 0.85

    if args.dump_steps:
        python_step_dir = out_dir / "python_steps"
        if not python_step_dir.exists() or (cpp_step_dir is None or not cpp_step_dir.exists()):
            print(f"[step-divergence] skipped: missing python_steps ({python_step_dir.exists()}) "
                  f"or cpp_steps ({cpp_step_dir is not None and cpp_step_dir.exists()})")
        else:
            report["step_divergence"] = compute_step_divergence(args, python_step_dir, cpp_step_dir)

    json_path = out_dir / "report.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)

    md_path = out_dir / "report.md"
    with open(md_path, "w") as f:
        f.write("# Task 11 E2E parity report\n\n")
        f.write(f"Mean SSIM: **{ssim_result['mean_ssim']:.4f}** "
                f"({'PASS' if report['pass'] else 'BELOW BAR'}, bar >= 0.85)\n\n")
        f.write("Settings: frames=%d steps=%d seed=%d cfg=%s size=%dx%d\n\n" % (
            args.frames, args.steps, args.seed, args.cfg, args.width, args.height))
        f.write("| frame | ssim |\n|---|---|\n")
        for i, s in enumerate(ssim_result["per_frame_ssim"]):
            f.write(f"| {i} | {s:.4f} |\n")
        f.write(f"\nWorst frame: {ssim_result['worst_frame']} (ssim {ssim_result['worst_ssim']:.4f})\n")
        if "step_divergence" in report:
            f.write("\n## Per-step rel L2 divergence (v-pred, latents)\n\n")
            f.write("| step | vpred rel L2 | latents rel L2 |\n|---|---|---|\n")
            for row in report["step_divergence"]["rows"]:
                f.write(f"| {row['step']} | {row['vpred_rel_l2']:.4e} | {row['latents_rel_l2']:.4e} |\n")

    print(f"\nWrote {json_path} and {md_path}")
    print(f"RESULT: mean SSIM = {ssim_result['mean_ssim']:.4f} "
          f"({'PASS' if report['pass'] else 'BELOW BAR (>= 0.85 required)'})")


if __name__ == "__main__":
    main()
