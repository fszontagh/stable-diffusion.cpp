#ifndef __SD_MODEL_ADAPTER_POSE_GUIDER_SSD_HPP__
#define __SD_MODEL_ADAPTER_POSE_GUIDER_SSD_HPP__

#include "core/ggml_extend.hpp"
#include "model/common/block.hpp"

/*
    ================= AnimateAnyone Pose Guider variant B (Sprite-Sheet-Diffusion) =================

    Source of truth (transcribed verbatim, not the P-map table, per Task 12 brief):
    /data/sdcpp-pixel-refs/sprite-sheet-diffusion/ModelTraining/models/pose_guider.py
    ("PoseGuider" class + the local "Transformer2DModel"/attention.py's BasicTransformerBlock it
    instantiates with default kwargs).

    conv_layers (nn.Sequential, 8 x [Conv2d, BatchNorm2d, ReLU]):
      0/1  Conv2d(3->3,   k3 s1 p1) + BN(3)
      3/4  Conv2d(3->16,  k4 s2 p1) + BN(16)
      6/7  Conv2d(16->16, k3 s1 p1) + BN(16)
      9/10 Conv2d(16->32, k4 s2 p1) + BN(32)
      12/13 Conv2d(32->32, k3 s1 p1) + BN(32)
      15/16 Conv2d(32->64, k4 s2 p1) + BN(64)
      18/19 Conv2d(64->64, k3 s1 p1) + BN(64)
      21/22 Conv2d(64->128,k3 s1 p1) + BN(128)
      (ReLU has no weights and sits at indices 2,5,8,11,14,17,20,23 - not registered as a block)
      Net stride: 8 (three k4s2 convs).

    final_proj: Conv2d(128 -> 320, k1), zero-initialized in the reference (irrelevant once real
    weights are loaded; matters only for the random-init fallback fixture).
    scale: learnable scalar nn.Parameter(torch.ones(1) * 2), multiplies final_proj's output.

    conv_layers_1: Conv2d(320->320,k3s1p1)+BN, Conv2d(320->320,k3s2p1)+BN   (indices 0/1, 3/4)
    conv_layers_2: Conv2d(320->320,k3s1p1)+BN, Conv2d(320->640,k3s2p1)+BN
    conv_layers_3: Conv2d(640->640,k3s1p1)+BN, Conv2d(640->1280,k3s2p1)+BN
    conv_layers_4: Conv2d(1280->1280,k3s1p1)+BN                            (index 0/1 only, stride 1)

    cross_attn1..4: the file's own "Transformer2DModel" (num_attention_heads=16,
    attention_head_dim=88 -> inner_dim=1408, in_channels = 320/640/1280/1280, conv proj_in/proj_out
    k1, GroupNorm(32, in_channels) norm (ggml's group-norm helper hardcodes eps=1e-6, matching
    nn.GroupNorm(..., eps=1e-6) here), one BasicTransformerBlock with cross_attention_dim=None and
    double_self_attention=False.

    IMPORTANT WIRING QUIRK (verified by reading diffusers-style attention.py's
    BasicTransformerBlock.__init__ in the same S repo): when cross_attention_dim is None and
    double_self_attention is False, that block never allocates norm2/attn2 at all
    ("self.norm2 = None; self.attn2 = None"), and forward() only runs attn2 "if self.attn2 is not
    None". So despite pose_guider.py's forward() computing a whole ref_x branch through
    conv_layers/conv_layers_k and calling cross_attn{k}(x, ref_x), the "encoder_hidden_states=ref_x"
    argument is NEVER consumed - cross_attn{k} is pure self-attention on x. The reference-pose
    branch is dead code in the upstream implementation (almost certainly an authoring bug: they
    presumably meant to pass cross_attention_dim=inner_dim in the local Transformer2DModel
    default, or thread the ref feature into an actual attn2). We transcribe the wiring exactly:
    PoseGuiderB::forward() takes and validates a ref_pose tensor (required by the CLI, matching the
    upstream API contract) but does not run it through any weights, since doing so is
    mathematically inert to the returned features - see the .py excerpt above for the exact
    evidence. This is called out again in docs/animate_anyone.md.

    forward(pose) -> 5 pyramid features, channels/resolution (relative to the 512x512 input):
      fea[0]: 320 @ 1/8   (after conv_layers + final_proj + scale)
      fea[1]: 320 @ 1/16  (after conv_layers_1 + cross_attn1)
      fea[2]: 640 @ 1/32  (after conv_layers_2 + cross_attn2)
      fea[3]: 1280 @ 1/64 (after conv_layers_3 + cross_attn3)
      fea[4]: 1280 @ 1/64 (after conv_layers_4 + cross_attn4, stride 1 - same resolution as fea[3])

    Input: pose skeleton RGB image, ggml layout [W, H, 3, N], values in [-1,1]
    (do_normalize=True in the S pipelines - DIFFERS from Moore's PoseGuiderA [0,1], P-map porting
    note 5).

    Checkpoint tensor prefixes (P-map section 7 pose_guider row, S form):
      conv_layers.{0,1,3,4,6,7,9,10,12,13,15,16,18,19,21,22}.*, final_proj.*,
      conv_layers_{1,2,3,4}.{0,1,3,4}.* (conv_layers_4 only has {0,1}), scale,
      cross_attn{1,2,3,4}.{norm,proj_in,proj_out,transformer_blocks.0.*}.*
*/

namespace AnimateAnyone {

// nn.BatchNorm2d in eval mode: y = (x - running_mean) / sqrt(running_var + eps) * weight + bias,
// all four params are per-channel [C] vectors read straight from the checkpoint (no fork
// BatchNorm2d block existed prior to this port).
class BatchNorm2d : public UnaryBlock {
protected:
    int64_t num_channels;
    float eps;
    std::string prefix;

    void init_params(ggml_context* ctx, const String2TensorStorage& tensor_storage_map = {}, const std::string prefix = "") override {
        this->prefix              = prefix;
        params["weight"]          = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, num_channels);
        params["bias"]            = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, num_channels);
        params["running_mean"]    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, num_channels);
        params["running_var"]     = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, num_channels);
    }

public:
    BatchNorm2d(int64_t num_channels, float eps = 1e-5f)
        : num_channels(num_channels), eps(eps) {}

    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
        // x: [W, H, C, N]
        ggml_tensor* w  = ggml_reshape_4d(ctx->ggml_ctx, params["weight"], 1, 1, num_channels, 1);
        ggml_tensor* b  = ggml_reshape_4d(ctx->ggml_ctx, params["bias"], 1, 1, num_channels, 1);
        ggml_tensor* rm = ggml_reshape_4d(ctx->ggml_ctx, params["running_mean"], 1, 1, num_channels, 1);
        ggml_tensor* rv = ggml_reshape_4d(ctx->ggml_ctx, params["running_var"], 1, 1, num_channels, 1);

        ggml_tensor* denom = ggml_sqrt(ctx->ggml_ctx, ggml_scale_bias(ctx->ggml_ctx, rv, 1.0f, eps));  // [1,1,C,1]
        ggml_tensor* scale = ggml_div(ctx->ggml_ctx, w, denom);                                        // [1,1,C,1]
        ggml_tensor* shift = ggml_sub(ctx->ggml_ctx, b, ggml_mul(ctx->ggml_ctx, rm, scale));            // [1,1,C,1]

        x = ggml_mul(ctx->ggml_ctx, x, scale);   // broadcast over W,H,N
        x = ggml_add(ctx->ggml_ctx, x, shift);   // broadcast over W,H,N
        return x;
    }
};

// A run of [Conv2d, BatchNorm2d, ReLU] pairs registered under literal nn.Sequential integer
// indices, so GGMLBlock::init() binds tensors exactly as `<prefix>.<idx>.*` matches the checkpoint.
class ConvBNReLUStack : public GGMLBlock {
public:
    struct Spec {
        int64_t in_c, out_c, k, stride, pad, index;  // index = the Conv2d's slot in the Sequential
    };

private:
    std::vector<int> conv_indices;

public:
    ConvBNReLUStack(const std::vector<Spec>& specs) {
        for (const auto& s : specs) {
            blocks[std::to_string(s.index)]     = std::shared_ptr<GGMLBlock>(
                new Conv2d(s.in_c, s.out_c, {s.k, s.k}, {s.stride, s.stride}, {s.pad, s.pad}));
            blocks[std::to_string(s.index + 1)] = std::shared_ptr<GGMLBlock>(new BatchNorm2d(s.out_c));
            conv_indices.push_back((int)s.index);
        }
    }

    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
        for (int idx : conv_indices) {
            auto conv = std::dynamic_pointer_cast<Conv2d>(blocks[std::to_string(idx)]);
            auto bn   = std::dynamic_pointer_cast<BatchNorm2d>(blocks[std::to_string(idx + 1)]);
            x         = conv->forward(ctx, x);
            x         = bn->forward(ctx, x);
            x         = ggml_relu_inplace(ctx->ggml_ctx, x);
        }
        return x;
    }
};

// One cross_attn{k} module: local "Transformer2DModel" from pose_guider.py, self-attention only
// (see the header comment above for why encoder_hidden_states/ref_x is not wired in).
class PoseGuiderCrossAttnBlock : public GGMLBlock {
protected:
    int64_t in_channels;
    int64_t n_head  = 16;
    int64_t d_head  = 88;  // inner_dim = 1408

public:
    PoseGuiderCrossAttnBlock(int64_t in_channels) : in_channels(in_channels) {
        int64_t inner_dim = n_head * d_head;
        blocks["norm"]     = std::shared_ptr<GGMLBlock>(new GroupNorm(32, in_channels));
        blocks["proj_in"]  = std::shared_ptr<GGMLBlock>(new Conv2d(in_channels, inner_dim, {1, 1}));
        blocks["proj_out"] = std::shared_ptr<GGMLBlock>(new Conv2d(inner_dim, in_channels, {1, 1}));

        blocks["transformer_blocks.0.norm1"] = std::shared_ptr<GGMLBlock>(new LayerNorm(inner_dim));
        blocks["transformer_blocks.0.attn1"] = std::shared_ptr<GGMLBlock>(new CrossAttention(inner_dim, inner_dim, n_head, d_head));
        blocks["transformer_blocks.0.norm3"] = std::shared_ptr<GGMLBlock>(new LayerNorm(inner_dim));
        blocks["transformer_blocks.0.ff"]    = std::shared_ptr<GGMLBlock>(new FeedForward(inner_dim, inner_dim));
    }

    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
        // x: [W, H, in_channels, N]
        auto norm     = std::dynamic_pointer_cast<GroupNorm>(blocks["norm"]);
        auto proj_in  = std::dynamic_pointer_cast<Conv2d>(blocks["proj_in"]);
        auto proj_out = std::dynamic_pointer_cast<Conv2d>(blocks["proj_out"]);
        auto norm1    = std::dynamic_pointer_cast<LayerNorm>(blocks["transformer_blocks.0.norm1"]);
        auto attn1    = std::dynamic_pointer_cast<CrossAttention>(blocks["transformer_blocks.0.attn1"]);
        auto norm3    = std::dynamic_pointer_cast<LayerNorm>(blocks["transformer_blocks.0.norm3"]);
        auto ff       = std::dynamic_pointer_cast<FeedForward>(blocks["transformer_blocks.0.ff"]);

        ggml_tensor* x_in     = x;
        int64_t n              = x->ne[3];
        int64_t h               = x->ne[1];
        int64_t w               = x->ne[0];
        int64_t inner_dim       = n_head * d_head;

        ggml_tensor* h_ = norm->forward(ctx, x);
        h_              = proj_in->forward(ctx, h_);                                             // [W,H,inner_dim,N]
        h_              = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, h_, 1, 2, 0, 3));  // [N,H,W,inner_dim]-ish
        h_              = ggml_reshape_3d(ctx->ggml_ctx, h_, inner_dim, w * h, n);                // [N, h*w, inner_dim]

        auto r = h_;
        h_     = norm1->forward(ctx, h_);
        h_     = attn1->forward(ctx, h_, h_);  // self-attention: context == itself
        h_     = ggml_add(ctx->ggml_ctx, h_, r);
        r      = h_;
        h_     = norm3->forward(ctx, h_);
        h_     = ff->forward(ctx, h_);
        h_     = ggml_add(ctx->ggml_ctx, h_, r);

        h_ = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, h_, 1, 0, 2, 3));  // [N, inner_dim, h*w]
        h_ = ggml_reshape_4d(ctx->ggml_ctx, h_, w, h, inner_dim, n);                // [N, inner_dim, h, w]

        h_ = proj_out->forward(ctx, h_);  // [N, in_channels, h, w]
        h_ = ggml_add(ctx->ggml_ctx, h_, x_in);
        return h_;
    }
};

struct PoseGuiderB : public GGMLBlock {
    void init_params(ggml_context* ctx, const String2TensorStorage& tensor_storage_map = {}, const std::string prefix = "") override {
        params["scale"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    }

    PoseGuiderB() {
        blocks["conv_layers"] = std::shared_ptr<GGMLBlock>(new ConvBNReLUStack({
            {3, 3, 3, 1, 1, 0},
            {3, 16, 4, 2, 1, 3},
            {16, 16, 3, 1, 1, 6},
            {16, 32, 4, 2, 1, 9},
            {32, 32, 3, 1, 1, 12},
            {32, 64, 4, 2, 1, 15},
            {64, 64, 3, 1, 1, 18},
            {64, 128, 3, 1, 1, 21},
        }));
        blocks["final_proj"] = std::shared_ptr<GGMLBlock>(new Conv2d(128, 320, {1, 1}));

        blocks["conv_layers_1"] = std::shared_ptr<GGMLBlock>(new ConvBNReLUStack({
            {320, 320, 3, 1, 1, 0},
            {320, 320, 3, 2, 1, 3},
        }));
        blocks["conv_layers_2"] = std::shared_ptr<GGMLBlock>(new ConvBNReLUStack({
            {320, 320, 3, 1, 1, 0},
            {320, 640, 3, 2, 1, 3},
        }));
        blocks["conv_layers_3"] = std::shared_ptr<GGMLBlock>(new ConvBNReLUStack({
            {640, 640, 3, 1, 1, 0},
            {640, 1280, 3, 2, 1, 3},
        }));
        blocks["conv_layers_4"] = std::shared_ptr<GGMLBlock>(new ConvBNReLUStack({
            {1280, 1280, 3, 1, 1, 0},
        }));

        blocks["cross_attn1"] = std::shared_ptr<GGMLBlock>(new PoseGuiderCrossAttnBlock(320));
        blocks["cross_attn2"] = std::shared_ptr<GGMLBlock>(new PoseGuiderCrossAttnBlock(640));
        blocks["cross_attn3"] = std::shared_ptr<GGMLBlock>(new PoseGuiderCrossAttnBlock(1280));
        blocks["cross_attn4"] = std::shared_ptr<GGMLBlock>(new PoseGuiderCrossAttnBlock(1280));
    }

    std::string get_desc() override {
        return "pose_guider_b";
    }

    // pose: [W, H, 3, N] in [-1,1]. ref_pose is accepted (and shape-checked by the caller) to
    // match the upstream forward(x, ref_x) signature, but is NOT consumed - see the header
    // comment: cross_attn{k} in the source of truth is self-attention only, so any ref_pose
    // computation would be provably inert to the returned features.
    std::vector<ggml_tensor*> forward(GGMLRunnerContext* ctx, ggml_tensor* pose, ggml_tensor* ref_pose) {
        auto conv_layers   = std::dynamic_pointer_cast<ConvBNReLUStack>(blocks["conv_layers"]);
        auto final_proj    = std::dynamic_pointer_cast<Conv2d>(blocks["final_proj"]);
        auto conv_layers_1 = std::dynamic_pointer_cast<ConvBNReLUStack>(blocks["conv_layers_1"]);
        auto conv_layers_2 = std::dynamic_pointer_cast<ConvBNReLUStack>(blocks["conv_layers_2"]);
        auto conv_layers_3 = std::dynamic_pointer_cast<ConvBNReLUStack>(blocks["conv_layers_3"]);
        auto conv_layers_4 = std::dynamic_pointer_cast<ConvBNReLUStack>(blocks["conv_layers_4"]);
        auto cross_attn1    = std::dynamic_pointer_cast<PoseGuiderCrossAttnBlock>(blocks["cross_attn1"]);
        auto cross_attn2    = std::dynamic_pointer_cast<PoseGuiderCrossAttnBlock>(blocks["cross_attn2"]);
        auto cross_attn3    = std::dynamic_pointer_cast<PoseGuiderCrossAttnBlock>(blocks["cross_attn3"]);
        auto cross_attn4    = std::dynamic_pointer_cast<PoseGuiderCrossAttnBlock>(blocks["cross_attn4"]);

        std::vector<ggml_tensor*> fea;

        ggml_tensor* scale = params["scale"];

        ggml_tensor* x = conv_layers->forward(ctx, pose);
        x              = final_proj->forward(ctx, x);
        x              = ggml_mul(ctx->ggml_ctx, x, ggml_reshape_4d(ctx->ggml_ctx, scale, 1, 1, 1, 1));
        fea.push_back(x);  // fea[0]: 320 @ 1/8

        x = conv_layers_1->forward(ctx, x);
        x = cross_attn1->forward(ctx, x);
        fea.push_back(x);  // fea[1]: 320 @ 1/16

        x = conv_layers_2->forward(ctx, x);
        x = cross_attn2->forward(ctx, x);
        fea.push_back(x);  // fea[2]: 640 @ 1/32

        x = conv_layers_3->forward(ctx, x);
        x = cross_attn3->forward(ctx, x);
        fea.push_back(x);  // fea[3]: 1280 @ 1/64

        x = conv_layers_4->forward(ctx, x);
        x = cross_attn4->forward(ctx, x);
        fea.push_back(x);  // fea[4]: 1280 @ 1/64 (same resolution as fea[3], stride 1 in conv_layers_4)

        return fea;
    }
};

// GGMLRunner wrapper around PoseGuiderB, mirroring PoseGuiderRunner (pose_guider.hpp).
struct PoseGuiderBRunner : public GGMLRunner {
    PoseGuiderB model;
    std::string prefix;

    PoseGuiderBRunner(ggml_backend_t backend,
                      const String2TensorStorage& tensor_storage_map,
                      const std::string& prefix                           = "",
                      std::shared_ptr<RunnerWeightManager> weight_manager = nullptr)
        : GGMLRunner(backend, weight_manager), prefix(prefix) {
        model.init(params_ctx, tensor_storage_map, prefix);
    }

    std::string get_desc() override {
        return "pose_guider_b";
    }

    void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors) {
        model.get_param_tensors(tensors, prefix);
    }

    // Multi-output readback follows the same pattern as
    // UNetModelRunner::compute_reference_banks (model/diffusion/unet.hpp): persist every
    // pyramid feature into the runner's named tensor cache during the graph build, run one
    // ordinary single-result compute() (the last feature is the "result" tensor), then pull the
    // other four back out of the cache by name.
    std::vector<sd::Tensor<float>> compute(int n_threads,
                                           const sd::Tensor<float>& pose_tensor,
                                           const sd::Tensor<float>& ref_pose_tensor) {
        auto get_graph = [&]() -> ggml_cgraph* {
            ggml_cgraph* gf  = new_graph_custom(8192);
            ggml_tensor* x   = make_input(pose_tensor);
            ggml_tensor* ref = make_input(ref_pose_tensor);
            auto runner_ctx  = get_context();

            std::vector<ggml_tensor*> feas = model.forward(&runner_ctx, x, ref);
            GGML_ASSERT(feas.size() == 5);
            for (int i = 0; i < 5; i++) {
                ggml_tensor* feat = ggml_cont(runner_ctx.ggml_ctx, feas[i]);
                ggml_set_output(feat);
                runner_ctx.persist_cache_tensor("pg.fea." + std::to_string(i), feat);
                ggml_build_forward_expand(gf, feat);
            }
            return gf;
        };

        auto out = GGMLRunner::compute<float>(get_graph, n_threads, false, false, false);
        std::vector<sd::Tensor<float>> results;
        if (!out.has_value()) {
            return results;
        }
        results.reserve(5);
        for (int i = 0; i < 5; i++) {
            ggml_tensor* cached = get_cache_tensor_by_name("pg.fea." + std::to_string(i));
            if (cached == nullptr) {
                results.clear();
                free_cache_ctx_and_buffer();
                return results;
            }
            results.push_back(restore_trailing_singleton_dims(sd::make_sd_tensor_from_ggml<float>(cached), 4));
        }
        free_cache_ctx_and_buffer();
        return results;
    }
};

}  // namespace AnimateAnyone

#endif  // __SD_MODEL_ADAPTER_POSE_GUIDER_SSD_HPP__
