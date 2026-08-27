#ifndef __SD_MODEL_ADAPTER_POSE_GUIDER_HPP__
#define __SD_MODEL_ADAPTER_POSE_GUIDER_HPP__

#include "core/ggml_extend.hpp"
#include "model/common/block.hpp"

/*
    =========================== AnimateAnyone Pose Guider (Moore) ===========================

    Reference: moore-animate-anyone/src/models/pose_guider.py (57 lines).

    conv_in: Conv2d(3 -> 16, k3 s1 p1)
    blocks.0: Conv2d(16 -> 16,  k3 s1 p1)
    blocks.1: Conv2d(16 -> 32,  k3 s2 p1)
    blocks.2: Conv2d(32 -> 32,  k3 s1 p1)
    blocks.3: Conv2d(32 -> 96,  k3 s2 p1)
    blocks.4: Conv2d(96 -> 96,  k3 s1 p1)
    blocks.5: Conv2d(96 -> 256, k3 s2 p1)
    conv_out: Conv2d(256 -> 320, k3 s1 p1), zero-initialized in the checkpoint

    Activation: SiLU after conv_in and after every block; none after conv_out.
    Block dict keys are the literal checkpoint tensor-name prefixes
    (conv_in, blocks.0 .. blocks.5, conv_out) so GGMLBlock::init() binds
    tensors by name with an empty/whatever prefix is passed in.

    Input: pose skeleton RGB image, ggml layout [W, H, 3, N], values in [0,1]
    (no [-1,1] rescale - matches the Moore VaeImageProcessor(do_normalize=False)).
    Output: [W/8, H/8, 320, N] - three stride-2 convs (blocks.1/.3/.5) give total
    stride 8, matching the latent spatial resolution.
*/

namespace AnimateAnyone {

struct PoseGuiderA : public GGMLBlock {
    PoseGuiderA() {
        blocks["conv_in"] = std::shared_ptr<GGMLBlock>(new Conv2d(3, 16, {3, 3}, {1, 1}, {1, 1}));

        blocks["blocks.0"] = std::shared_ptr<GGMLBlock>(new Conv2d(16, 16, {3, 3}, {1, 1}, {1, 1}));
        blocks["blocks.1"] = std::shared_ptr<GGMLBlock>(new Conv2d(16, 32, {3, 3}, {2, 2}, {1, 1}));
        blocks["blocks.2"] = std::shared_ptr<GGMLBlock>(new Conv2d(32, 32, {3, 3}, {1, 1}, {1, 1}));
        blocks["blocks.3"] = std::shared_ptr<GGMLBlock>(new Conv2d(32, 96, {3, 3}, {2, 2}, {1, 1}));
        blocks["blocks.4"] = std::shared_ptr<GGMLBlock>(new Conv2d(96, 96, {3, 3}, {1, 1}, {1, 1}));
        blocks["blocks.5"] = std::shared_ptr<GGMLBlock>(new Conv2d(96, 256, {3, 3}, {2, 2}, {1, 1}));

        blocks["conv_out"] = std::shared_ptr<GGMLBlock>(new Conv2d(256, 320, {3, 3}, {1, 1}, {1, 1}));
    }

    std::string get_desc() override {
        return "pose_guider";
    }

    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* pose_rgb) {
        // pose_rgb: [W, H, 3, N] -> [W/8, H/8, 320, N]
        auto conv_in = std::dynamic_pointer_cast<Conv2d>(blocks["conv_in"]);
        auto conv_out = std::dynamic_pointer_cast<Conv2d>(blocks["conv_out"]);

        ggml_tensor* h = conv_in->forward(ctx, pose_rgb);
        h              = ggml_silu_inplace(ctx->ggml_ctx, h);

        for (int i = 0; i < 6; i++) {
            auto block = std::dynamic_pointer_cast<Conv2d>(blocks["blocks." + std::to_string(i)]);
            h          = block->forward(ctx, h);
            h          = ggml_silu_inplace(ctx->ggml_ctx, h);
        }

        h = conv_out->forward(ctx, h);
        return h;
    }
};

}  // namespace AnimateAnyone

#endif  // __SD_MODEL_ADAPTER_POSE_GUIDER_HPP__
