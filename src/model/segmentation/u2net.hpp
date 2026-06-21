#ifndef __SD_MODEL_SEGMENTATION_U2NET_HPP__
#define __SD_MODEL_SEGMENTATION_U2NET_HPP__

#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/ggml_extend.hpp"
#include "core/util.h"

/*
    ===================================  U^2-Net  ====================================
    Reference: https://github.com/xuebinqin/U-2-Net  (Qin et al., 2020)
    "U^2-Net: Going Deeper with Nested U-Structure for Salient Object Detection"

    The full U^2-Net has 6 encoder stages + 5 decoder stages, where each stage is
    itself a small U-shaped block (RSU = ReSidual U-block). The smallest variant
    U^2-NetP shares the same topology with fewer channels.

    The convert_u2net.py tool fuses the original Conv2d + BatchNorm2d into a
    single Conv2d with adjusted weight/bias, so the inference path here is plain
    Conv2d + ReLU.
*/

struct U2NetConfig {
    bool small_variant = false;  // u2netp uses 16 mid channels everywhere; u2net uses growing mid channels
    int in_ch          = 3;
    int out_ch         = 1;

    static U2NetConfig detect_from_weights(const String2TensorStorage& tensor_storage_map) {
        U2NetConfig config;
        auto it = tensor_storage_map.find("stage1.rebnconvin.conv_s1.weight");
        if (it != tensor_storage_map.end()) {
            // u2net's stage1.rebnconvin outputs 64 channels; u2netp outputs 64 as well but
            // the inner mid channels differ (16 for u2netp, 32+ for u2net).
            // Detect via stage1.rebnconv1: u2net = 32 channels, u2netp = 16.
            auto it_in = tensor_storage_map.find("stage1.rebnconv1.conv_s1.weight");
            if (it_in != tensor_storage_map.end() && it_in->second.ne[0] == 16) {
                config.small_variant = true;
            }
        }
        return config;
    }
};

// REBNCONV = Conv2d (with BN fused in) + ReLU.
class REBNConv : public GGMLBlock {
public:
    REBNConv(int in_ch, int out_ch, int dilation = 1) {
        // pad = dilation keeps spatial dim under a 3x3 conv with given dilation.
        const int pad = dilation;
        blocks["conv_s1"] = std::shared_ptr<GGMLBlock>(
            new Conv2d(in_ch, out_ch, {3, 3}, {1, 1}, {pad, pad}, {dilation, dilation}));
    }

    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
        auto conv = std::dynamic_pointer_cast<Conv2d>(blocks["conv_s1"]);
        x = conv->forward(ctx, x);
        x = ggml_relu_inplace(ctx->ggml_ctx, x);
        return x;
    }
};

// Bilinear upsample of `src` to the spatial size of `target`.
inline ggml_tensor* upsample_like(GGMLRunnerContext* ctx, ggml_tensor* src, ggml_tensor* target) {
    return ggml_interpolate(ctx->ggml_ctx,
                            src,
                            target->ne[0],
                            target->ne[1],
                            src->ne[2],
                            src->ne[3],
                            GGML_SCALE_MODE_BILINEAR | GGML_SCALE_FLAG_ALIGN_CORNERS);
}

// 2x max-pool with ceil_mode=false. U^2-Net uses max_pool2d(2, stride=2, ceil_mode=True)
// internally, but for the stable input sizes (320x320) ceil_mode is a no-op.
inline ggml_tensor* downsample_pool(GGMLRunnerContext* ctx, ggml_tensor* x) {
    return ggml_pool_2d(ctx->ggml_ctx, x, GGML_OP_POOL_MAX, 2, 2, 2, 2, 0, 0);
}

// RSU = ReSidual U-block. Parameterized by encoder depth N (= 7, 6, 5, 4).
// Each side has N REBNConv layers; pooling between encoder layers, bilinear
// upsample between decoder layers. A single dilated REBNConv bridges the bottom.
class RSU : public GGMLBlock {
protected:
    int depth;
    bool dilated;  // RSU4F uses dilations instead of pooling/upsampling

public:
    RSU(int depth, int in_ch, int mid_ch, int out_ch, bool dilated = false)
        : depth(depth), dilated(dilated) {
        blocks["rebnconvin"] = std::shared_ptr<GGMLBlock>(new REBNConv(in_ch, out_ch));

        // Encoder side: rebnconv1 .. rebnconv{depth}. In the dilated variant each
        // successive layer has dilation 2^(i-1); in the pooled variant they use
        // plain Conv with downsampling between layers.
        for (int i = 1; i <= depth; ++i) {
            int d = dilated ? (1 << (i - 1)) : 1;
            int ic = (i == 1) ? out_ch : mid_ch;
            blocks["rebnconv" + std::to_string(i)] = std::shared_ptr<GGMLBlock>(
                new REBNConv(ic, mid_ch, d));
        }
        // Bottom bridge has dilation = 2^depth in dilated mode, 2 in pooled mode.
        int bottom_d = dilated ? (1 << depth) : 2;
        blocks["rebnconv" + std::to_string(depth + 1)] = std::shared_ptr<GGMLBlock>(
            new REBNConv(mid_ch, mid_ch, bottom_d));

        // Decoder side: rebnconv{depth}d .. rebnconv1d. Input is concat of encoder
        // and decoder activations, so input channels are 2*mid_ch (except the last
        // which produces out_ch).
        for (int i = depth; i >= 2; --i) {
            int d = dilated ? (1 << (i - 1)) : 1;
            blocks["rebnconv" + std::to_string(i) + "d"] = std::shared_ptr<GGMLBlock>(
                new REBNConv(2 * mid_ch, mid_ch, d));
        }
        blocks["rebnconv1d"] = std::shared_ptr<GGMLBlock>(new REBNConv(2 * mid_ch, out_ch));
    }

    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
        auto get_block = [&](const std::string& name) {
            return std::dynamic_pointer_cast<REBNConv>(blocks[name]);
        };

        // Input projection (also the residual we add at the end).
        auto rebnconvin = get_block("rebnconvin");
        ggml_tensor* xin = rebnconvin->forward(ctx, x);

        // Encoder: collect activations at each level for the skip connections.
        // The last conv does not get pooled - its output feeds the bridge directly.
        std::vector<ggml_tensor*> encs(depth + 1, nullptr);
        ggml_tensor* prev = xin;
        for (int i = 1; i <= depth; ++i) {
            auto block = get_block("rebnconv" + std::to_string(i));
            ggml_tensor* e = block->forward(ctx, prev);
            encs[i] = e;
            if (dilated || i == depth) {
                prev = e;
            } else {
                prev = downsample_pool(ctx, e);
            }
        }

        // Bottom bridge.
        auto bridge = get_block("rebnconv" + std::to_string(depth + 1));
        ggml_tensor* d = bridge->forward(ctx, prev);

        // Decoder: walk back up, concat with encoder activations.
        for (int i = depth; i >= 2; --i) {
            auto block = get_block("rebnconv" + std::to_string(i) + "d");
            ggml_tensor* concat = ggml_concat(ctx->ggml_ctx, d, encs[i], 2);
            d = block->forward(ctx, concat);
            if (!dilated) {
                d = upsample_like(ctx, d, encs[i - 1]);
            }
        }

        // Last decoder layer + residual.
        auto last = get_block("rebnconv1d");
        ggml_tensor* concat = ggml_concat(ctx->ggml_ctx, d, encs[1], 2);
        ggml_tensor* dout = last->forward(ctx, concat);
        return ggml_add(ctx->ggml_ctx, dout, xin);
    }
};

// Full U^2-Net network. 6 encoder stages (RSU7/6/5/4/4F/4F), 5 decoder stages,
// and 6 side outputs fused to a single saliency map.
class U2Net : public GGMLBlock {
protected:
    U2NetConfig config;
    int channel(int u2net_ch, int u2netp_ch) const {
        return config.small_variant ? u2netp_ch : u2net_ch;
    }

public:
    explicit U2Net(U2NetConfig cfg) : config(std::move(cfg)) {
        const int in  = config.in_ch;
        const int out = config.out_ch;
        // U^2-Net channel schedule (PyTorch reference).
        //   Encoder stage N gets RSU{N} with (in, mid, out) channels.
        //   Decoder stages have asymmetric (smaller) out_ch and halved mid_ch.
        //   u2netp uses constant mid=16 / out=64 throughout.
        const auto e_m1 = channel(32,  16), e_o1 = channel(64,  64);
        const auto e_m2 = channel(32,  16), e_o2 = channel(128, 64);
        const auto e_m3 = channel(64,  16), e_o3 = channel(256, 64);
        const auto e_m4 = channel(128, 16), e_o4 = channel(512, 64);
        const auto e_m5 = channel(256, 16), e_o5 = channel(512, 64);
        const auto e_m6 = channel(256, 16), e_o6 = channel(512, 64);

        const auto d_m5 = channel(256, 16), d_o5 = channel(512, 64);
        const auto d_m4 = channel(128, 16), d_o4 = channel(256, 64);
        const auto d_m3 = channel(64,  16), d_o3 = channel(128, 64);
        const auto d_m2 = channel(32,  16), d_o2 = channel(64,  64);
        const auto d_m1 = channel(16,  16), d_o1 = channel(64,  64);

        // RSU depth parameter == number of encoder convs (= RSU name - 1).
        blocks["stage1"] = std::shared_ptr<GGMLBlock>(new RSU(6, in,    e_m1, e_o1));                       // RSU7
        blocks["stage2"] = std::shared_ptr<GGMLBlock>(new RSU(5, e_o1, e_m2, e_o2));                       // RSU6
        blocks["stage3"] = std::shared_ptr<GGMLBlock>(new RSU(4, e_o2, e_m3, e_o3));                       // RSU5
        blocks["stage4"] = std::shared_ptr<GGMLBlock>(new RSU(3, e_o3, e_m4, e_o4));                       // RSU4
        blocks["stage5"] = std::shared_ptr<GGMLBlock>(new RSU(3, e_o4, e_m5, e_o5, /*dilated=*/true));     // RSU4F
        blocks["stage6"] = std::shared_ptr<GGMLBlock>(new RSU(3, e_o5, e_m6, e_o6, /*dilated=*/true));     // RSU4F

        blocks["stage5d"] = std::shared_ptr<GGMLBlock>(new RSU(3, e_o6 + e_o5, d_m5, d_o5, /*dilated=*/true));
        blocks["stage4d"] = std::shared_ptr<GGMLBlock>(new RSU(3, d_o5 + e_o4, d_m4, d_o4));
        blocks["stage3d"] = std::shared_ptr<GGMLBlock>(new RSU(4, d_o4 + e_o3, d_m3, d_o3));
        blocks["stage2d"] = std::shared_ptr<GGMLBlock>(new RSU(5, d_o3 + e_o2, d_m2, d_o2));
        blocks["stage1d"] = std::shared_ptr<GGMLBlock>(new RSU(6, d_o2 + e_o1, d_m1, d_o1));

        // Side outputs (3x3 conv to single saliency channel).
        blocks["side1"] = std::shared_ptr<GGMLBlock>(new Conv2d(d_o1, out, {3, 3}, {1, 1}, {1, 1}));
        blocks["side2"] = std::shared_ptr<GGMLBlock>(new Conv2d(d_o2, out, {3, 3}, {1, 1}, {1, 1}));
        blocks["side3"] = std::shared_ptr<GGMLBlock>(new Conv2d(d_o3, out, {3, 3}, {1, 1}, {1, 1}));
        blocks["side4"] = std::shared_ptr<GGMLBlock>(new Conv2d(d_o4, out, {3, 3}, {1, 1}, {1, 1}));
        blocks["side5"] = std::shared_ptr<GGMLBlock>(new Conv2d(d_o5, out, {3, 3}, {1, 1}, {1, 1}));
        blocks["side6"] = std::shared_ptr<GGMLBlock>(new Conv2d(e_o6, out, {3, 3}, {1, 1}, {1, 1}));

        // Final fusion conv across the 6 side outputs.
        blocks["outconv"] = std::shared_ptr<GGMLBlock>(new Conv2d(6 * out, out, {1, 1}));
    }

    // Returns the final fused saliency map sigmoid'd to [0,1] at the input resolution.
    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
        auto stage = [&](const std::string& name) {
            return std::dynamic_pointer_cast<RSU>(blocks[name]);
        };
        auto side = [&](const std::string& name) {
            return std::dynamic_pointer_cast<Conv2d>(blocks[name]);
        };

        // Encoder path with pooling between stages.
        ggml_tensor* h1 = stage("stage1")->forward(ctx, x);
        ggml_tensor* p1 = downsample_pool(ctx, h1);
        ggml_tensor* h2 = stage("stage2")->forward(ctx, p1);
        ggml_tensor* p2 = downsample_pool(ctx, h2);
        ggml_tensor* h3 = stage("stage3")->forward(ctx, p2);
        ggml_tensor* p3 = downsample_pool(ctx, h3);
        ggml_tensor* h4 = stage("stage4")->forward(ctx, p3);
        ggml_tensor* p4 = downsample_pool(ctx, h4);
        ggml_tensor* h5 = stage("stage5")->forward(ctx, p4);
        ggml_tensor* p5 = downsample_pool(ctx, h5);
        ggml_tensor* h6 = stage("stage6")->forward(ctx, p5);

        // Decoder path: concat upsampled previous decoder + skip, run RSU.
        ggml_tensor* h6up = upsample_like(ctx, h6, h5);
        ggml_tensor* h5d  = stage("stage5d")->forward(ctx, ggml_concat(ctx->ggml_ctx, h6up, h5, 2));

        ggml_tensor* h5dup = upsample_like(ctx, h5d, h4);
        ggml_tensor* h4d   = stage("stage4d")->forward(ctx, ggml_concat(ctx->ggml_ctx, h5dup, h4, 2));

        ggml_tensor* h4dup = upsample_like(ctx, h4d, h3);
        ggml_tensor* h3d   = stage("stage3d")->forward(ctx, ggml_concat(ctx->ggml_ctx, h4dup, h3, 2));

        ggml_tensor* h3dup = upsample_like(ctx, h3d, h2);
        ggml_tensor* h2d   = stage("stage2d")->forward(ctx, ggml_concat(ctx->ggml_ctx, h3dup, h2, 2));

        ggml_tensor* h2dup = upsample_like(ctx, h2d, h1);
        ggml_tensor* h1d   = stage("stage1d")->forward(ctx, ggml_concat(ctx->ggml_ctx, h2dup, h1, 2));

        // Side outputs, each upsampled back to the input resolution.
        ggml_tensor* d1 = side("side1")->forward(ctx, h1d);
        ggml_tensor* d2 = upsample_like(ctx, side("side2")->forward(ctx, h2d), x);
        ggml_tensor* d3 = upsample_like(ctx, side("side3")->forward(ctx, h3d), x);
        ggml_tensor* d4 = upsample_like(ctx, side("side4")->forward(ctx, h4d), x);
        ggml_tensor* d5 = upsample_like(ctx, side("side5")->forward(ctx, h5d), x);
        ggml_tensor* d6 = upsample_like(ctx, side("side6")->forward(ctx, h6),  x);

        // Fuse the six side maps.
        ggml_tensor* concat = ggml_concat(ctx->ggml_ctx, d1, d2, 2);
        concat = ggml_concat(ctx->ggml_ctx, concat, d3, 2);
        concat = ggml_concat(ctx->ggml_ctx, concat, d4, 2);
        concat = ggml_concat(ctx->ggml_ctx, concat, d5, 2);
        concat = ggml_concat(ctx->ggml_ctx, concat, d6, 2);

        auto outconv = std::dynamic_pointer_cast<Conv2d>(blocks["outconv"]);
        ggml_tensor* d0 = outconv->forward(ctx, concat);
        return ggml_sigmoid_inplace(ctx->ggml_ctx, d0);
    }
};

// Runner that owns the U2Net network and computes a mask from an input image
// tensor.
struct U2NetRunner : public GGMLRunner {
    U2NetConfig config;
    std::unique_ptr<U2Net> network;

    U2NetRunner(ggml_backend_t backend,
                const String2TensorStorage& tensor_storage_map      = {},
                std::shared_ptr<RunnerWeightManager> weight_manager = nullptr)
        : GGMLRunner(backend, weight_manager),
          config(U2NetConfig::detect_from_weights(tensor_storage_map)),
          network(std::make_unique<U2Net>(config)) {
        network->init(params_ctx, tensor_storage_map, "");
    }

    std::string get_desc() override {
        return "u2net";
    }

    void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors) {
        if (network) {
            network->get_param_tensors(tensors);
        }
    }

    ggml_cgraph* build_graph(const sd::Tensor<float>& x_tensor) {
        if (!network) {
            return nullptr;
        }
        constexpr int kGraphNodes = 1 << 16;
        ggml_cgraph* gf           = new_graph_custom(kGraphNodes);
        ggml_tensor* x            = make_input(x_tensor);
        auto runner_ctx           = get_context();
        ggml_tensor* out          = network->forward(&runner_ctx, x);
        ggml_build_forward_expand(gf, out);
        return gf;
    }

    sd::Tensor<float> compute(const int n_threads, const sd::Tensor<float>& x) {
        auto get_graph = [&]() -> ggml_cgraph* { return build_graph(x); };
        auto result    = restore_trailing_singleton_dims(GGMLRunner::compute<float>(get_graph, n_threads, false, false, false), x.dim());
        return result;
    }
};

#endif  // __SD_MODEL_SEGMENTATION_U2NET_HPP__
