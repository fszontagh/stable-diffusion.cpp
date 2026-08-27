#!/usr/bin/env bash
# Download the pretrained weights needed to run the Moore-AnimateAnyone
# reference pipeline (pose2vid) and to dump PyTorch fixtures for the
# sd.cpp AnimateAnyone port.
#
# Weights are large binary artifacts and are NOT committed to this repo.
# They are downloaded into $SDCPP_AA_WEIGHTS (default:
# /data/sdcpp-pixel-refs/weights). Only this script + the generated
# MANIFEST.txt (file list + sha256) are tracked in git.
#
# Usage:
#   tools/aa/download_weights.sh
#
# Env overrides:
#   SDCPP_AA_WEIGHTS   - destination dir (default /data/sdcpp-pixel-refs/weights)
#   HF_TOKEN            - optional HuggingFace token for higher rate limits
#
# Sources (see docs/superpowers/notes/animate-anyone-pytorch-map.md section 5,7,8):
#   patrolli/AnimateAnyone            -> denoising_unet.pth, reference_unet.pth,
#                                         pose_guider.pth, motion_module.pth
#   stabilityai/sd-vae-ft-mse         -> vae config.json + diffusion_pytorch_model.safetensors
#   lambdalabs/sd-image-variations-diffusers -> image_encoder/{config.json,pytorch_model.bin}
#   runwayml/stable-diffusion-v1-5    -> unet/config.json (UNet2D/UNet3D config only;
#                                         we do NOT need runwayml's unet weights - the
#                                         AnimateAnyone reference/denoising unets are
#                                         loaded from patrolli/AnimateAnyone's .pth files
#                                         on top of this config, per pose2vid.py:56-68)

set -euo pipefail

WEIGHTS_DIR="${SDCPP_AA_WEIGHTS:-/data/sdcpp-pixel-refs/weights}"
mkdir -p "$WEIGHTS_DIR"

AUTH_HEADER=()
if [[ -n "${HF_TOKEN:-}" ]]; then
  AUTH_HEADER=(-H "Authorization: Bearer ${HF_TOKEN}")
fi

download() {
  local url="$1"
  local out="$2"
  mkdir -p "$(dirname "$out")"
  if [[ -f "$out" ]]; then
    echo "SKIP (exists): $out"
    return 0
  fi
  echo "GET $url -> $out"
  curl -fL --retry 5 --retry-delay 3 "${AUTH_HEADER[@]}" -o "$out.part" "$url"
  mv "$out.part" "$out"
}

# --- patrolli/AnimateAnyone ---
for f in denoising_unet.pth reference_unet.pth pose_guider.pth motion_module.pth; do
  download "https://huggingface.co/patrolli/AnimateAnyone/resolve/main/${f}" \
    "$WEIGHTS_DIR/AnimateAnyone/${f}"
done

# --- stabilityai/sd-vae-ft-mse ---
download "https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/config.json" \
  "$WEIGHTS_DIR/sd-vae-ft-mse/config.json"
download "https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/diffusion_pytorch_model.safetensors" \
  "$WEIGHTS_DIR/sd-vae-ft-mse/diffusion_pytorch_model.safetensors"

# --- lambdalabs/sd-image-variations-diffusers (image_encoder subfolder) ---
download "https://huggingface.co/lambdalabs/sd-image-variations-diffusers/resolve/main/image_encoder/config.json" \
  "$WEIGHTS_DIR/image_encoder/config.json"
download "https://huggingface.co/lambdalabs/sd-image-variations-diffusers/resolve/main/image_encoder/pytorch_model.bin" \
  "$WEIGHTS_DIR/image_encoder/pytorch_model.bin"

# --- SD1.5 base unet (config + vanilla weights) ---
# NOTE: deviates from the task brief's "config.json only" - both
# UNet2DConditionModel.from_pretrained(...) (ReferenceNet) and
# UNet3DConditionModel.from_pretrained_2d(...) (denoising net) construct the
# model AND load a base weights file from this path before AnimateAnyone's
# own .pth state dicts are load_state_dict()'d on top (pose2vid.py:56-68,
# 86-92; unet_3d.py from_pretrained_2d:641-660 raises FileNotFoundError if no
# weights file is present). The base SD1.5 values are fully overwritten by
# the AnimateAnyone checkpoints afterwards, so this file only needs to exist
# with matching tensor shapes - but it must exist.
# runwayml/stable-diffusion-v1-5 now redirects to stable-diffusion-v1-5/stable-diffusion-v1-5.
download "https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/resolve/main/unet/config.json" \
  "$WEIGHTS_DIR/stable-diffusion-v1-5/unet/config.json"
download "https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/resolve/main/unet/diffusion_pytorch_model.bin" \
  "$WEIGHTS_DIR/stable-diffusion-v1-5/unet/diffusion_pytorch_model.bin"

echo "Writing MANIFEST.txt ..."
MANIFEST="$WEIGHTS_DIR/MANIFEST.txt"
{
  echo "# AnimateAnyone weight manifest - generated $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "# file  size_bytes  sha256"
  find "$WEIGHTS_DIR" -type f ! -name 'MANIFEST.txt' | sort | while read -r f; do
    rel="${f#"$WEIGHTS_DIR"/}"
    size=$(stat -c%s "$f")
    hash=$(sha256sum "$f" | awk '{print $1}')
    echo "$rel  $size  $hash"
  done
} > "$MANIFEST"

echo "Done. Manifest at $MANIFEST"
cat "$MANIFEST"
