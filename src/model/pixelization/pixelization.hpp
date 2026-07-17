#ifndef __PIXELIZATION_HPP__
#define __PIXELIZATION_HPP__

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "core/ggml_extend.hpp"

namespace pixelization {

// Stage dumps for the PyTorch parity harness. Enabled only when SD_PIXELIZATION_DUMP
// names a directory; writes .npy in PyTorch [N, C, H, W] order.
__STATIC_INLINE__ const char* stage_dump_dir() {
    return getenv("SD_PIXELIZATION_DUMP");
}

// Copy into a dedicated output tensor so the allocator cannot reuse the buffer before the
// stage is read back, mirroring GGMLRunnerContext::capture_tensor.
__STATIC_INLINE__ ggml_tensor* snapshot_tensor(GGMLRunnerContext* ctx, ggml_tensor* x) {
    ggml_tensor* snapshot = ggml_cont(ctx->ggml_ctx, x);
    snapshot              = ggml_cpy(ctx->ggml_ctx, snapshot, ggml_dup_tensor(ctx->ggml_ctx, snapshot));
    ggml_set_output(snapshot);
    return snapshot;
}

__STATIC_INLINE__ void dump_stage(const std::string& name, const sd::Tensor<float>& tensor) {
    const char* dir = stage_dump_dir();
    if (dir == nullptr || tensor.empty()) {
        return;
    }
    FILE* f = fopen((std::string(dir) + "/" + name + ".npy").c_str(), "wb");
    if (f == nullptr) {
        LOG_WARN("pixelization: cannot open stage dump for '%s'", name.c_str());
        return;
    }
    std::string header = "{'descr': '<f4', 'fortran_order': False, 'shape': (";
    for (size_t i = tensor.shape().size(); i > 0; i--) {
        header += std::to_string(tensor.shape()[i - 1]) + ",";
    }
    header += "), }";
    while ((10 + header.size() + 1) % 64 != 0) {
        header += ' ';
    }
    header += '\n';
    const uint16_t header_len = (uint16_t)header.size();
    fwrite("\x93NUMPY\x01\x00", 1, 8, f);
    fwrite(&header_len, 2, 1, f);
    fwrite(header.data(), 1, header.size(), f);
    fwrite(tensor.data(), sizeof(float), tensor.numel(), f);
    fclose(f);
}

// Upstream's "LayerNorm" (basic_layer.py:338) reduces over the WHOLE tensor and applies
// gamma/beta per channel, so the LayerNorm block (which reduces over ne[0] only) cannot be
// reused. torch.std() is Bessel-corrected and the eps sits OUTSIDE the sqrt.
class WholeTensorNorm : public UnaryBlock {
protected:
    int64_t num_features;
    float eps;

    void init_params(ggml_context* ctx, const String2TensorStorage& tensor_storage_map, const std::string prefix = "") override {
        params["gamma"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, num_features);
        params["beta"]  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, num_features);
    }

public:
    WholeTensorNorm(int64_t num_features, float eps = 1e-5f)
        : num_features(num_features), eps(eps) {}

    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
        auto gc         = ctx->ggml_ctx;
        const int64_t n = ggml_nelements(x);

        ggml_tensor* mean = ggml_mean(gc, ggml_reshape_1d(gc, ggml_cont(gc, x), n));
        ggml_tensor* xc   = ggml_sub(gc, x, ggml_repeat(gc, ggml_reshape_4d(gc, mean, 1, 1, 1, 1), x));

        ggml_tensor* var = ggml_mean(gc, ggml_sqr(gc, ggml_reshape_1d(gc, ggml_cont(gc, xc), n)));
        var              = ggml_scale(gc, var, (float)n / (float)(n - 1));  // Bessel correction
        ggml_tensor* sd  = ggml_scale_bias(gc, ggml_sqrt(gc, var), 1.0f, eps);
        x                = ggml_div(gc, xc, ggml_repeat(gc, ggml_reshape_4d(gc, sd, 1, 1, 1, 1), xc));

        ggml_tensor* g = ggml_reshape_4d(gc, params["gamma"], 1, 1, num_features, 1);
        ggml_tensor* b = ggml_reshape_4d(gc, params["beta"], 1, 1, num_features, 1);
        return ggml_add(gc, ggml_mul(gc, x, g), b);
    }
};

// StyleGAN2 modulate + demodulate.
//
// Weights MUST stay F32: the style code is raw (absmax ~8.4e8) and Conv2d forces F16 weights,
// which would overflow. Hence raw params + a direct ggml_ext_conv_2d call.
//
// Upstream (basic_layer.py:31) reinterprets the (out, in, k, k) weight buffer as
// (k, k, in, out) with a bare .view(), which is a reindex, not a transpose. Modulation and
// demodulation therefore act on the reinterpreted axes, and the buffer is permuted back into
// conv layout afterwards. This is faithful to the trained weights; do not "simplify" it.
class ModulationConvBlock : public GGMLBlock {
protected:
    int64_t in_c, out_c;
    int ksize;
    float wscale;

    void init_params(ggml_context* ctx, const String2TensorStorage& tensor_storage_map, const std::string prefix = "") override {
        params["weight"] = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ksize, ksize, in_c, out_c);
        params["bias"]   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, out_c);
    }

public:
    ModulationConvBlock(int64_t in_c, int64_t out_c, int ksize)
        : in_c(in_c), out_c(out_c), ksize(ksize) {
        wscale = 1.0f / std::sqrt((float)(ksize * ksize * in_c));
    }

    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x, ggml_tensor* code) {
        auto gc = ctx->ggml_ctx;

        ggml_tensor* w = ggml_scale(gc, params["weight"], wscale);
        // [kw, kh, in, out] -> [out, in, kw', kh']; matches upstream's .view(1, k, k, in, out)
        w = ggml_reshape_4d(gc, w, out_c, in_c, ksize, ksize);
        w = ggml_mul(gc, w, ggml_reshape_4d(gc, code, 1, in_c, 1, 1));

        // eps is INSIDE the sqrt here (basic_layer.py:35), unlike WholeTensorNorm. This is why
        // the style code must stay raw: shrinking it shrinks this sum until 1e-8 matters.
        ggml_tensor* sq   = ggml_reshape_2d(gc, ggml_cont(gc, ggml_sqr(gc, w)), out_c, ksize * ksize * in_c);
        ggml_tensor* norm = ggml_sum_rows(gc, ggml_cont(gc, ggml_transpose(gc, sq)));
        norm              = ggml_sqrt(gc, ggml_scale_bias(gc, norm, 1.0f, 1e-8f));
        w                 = ggml_div(gc, w, ggml_reshape_4d(gc, norm, out_c, 1, 1, 1));

        w = ggml_cont(gc, ggml_permute(gc, w, 3, 2, 0, 1));

        // ZERO padding: basic_layer.py:47 passes padding= to F.conv2d directly. Only ConvBlock
        // reflect-pads, despite the shared pad_type ctor argument.
        x = ggml_ext_conv_2d(gc, x, w, nullptr, 1, 1, ksize / 2, ksize / 2, 1, 1);
        x = ggml_add(gc, x, ggml_reshape_4d(gc, params["bias"], 1, 1, out_c, 1));
        x = ggml_leaky_relu(gc, x, 0.2f, true);
        return ggml_scale(gc, x, std::sqrt(2.0f));
    }
};

// basic_layer.py ConvBlock: reflect-pad -> Conv2d(padding=0) -> norm -> activation.
//
// Weights are raw F32 params rather than a Conv2d sub-block because Conv2d forces F16, which
// costs more than the 1e-4 parity budget once ~30 of these are chained.
//
// norm='in' is nn.InstanceNorm2d, whose default is affine=False, so there are no norm params in
// the checkpoint. ggml_ext_group_norm handles groups == channels correctly, but hardcodes
// eps=1e-6; ggml_group_norm is called directly to get PyTorch's 1e-5.
class ConvBlockPx : public UnaryBlock {
protected:
    int64_t in_c, out_c;
    int ksize, stride, padding;
    std::string norm, activation;

    void init_params(ggml_context* ctx, const String2TensorStorage& tensor_storage_map, const std::string prefix = "") override {
        params["conv.weight"] = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ksize, ksize, in_c, out_c);
        params["conv.bias"]   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, out_c);
    }

public:
    ConvBlockPx(int64_t in_c,
                int64_t out_c,
                int ksize,
                int stride,
                int padding,
                const std::string& norm       = "none",
                const std::string& activation = "relu")
        : in_c(in_c), out_c(out_c), ksize(ksize), stride(stride), padding(padding), norm(norm), activation(activation) {
        if (norm == "ln") {
            blocks["norm"] = std::shared_ptr<GGMLBlock>(new WholeTensorNorm(out_c));
        }
    }

    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
        auto gc = ctx->ggml_ctx;

        x = ggml_ext_pad_reflect_2d(gc, x, padding, padding);
        x = ggml_ext_conv_2d(gc, x, params["conv.weight"], params["conv.bias"], stride, stride, 0, 0, 1, 1);

        if (norm == "in") {
            x = ggml_group_norm(gc, x, (int)out_c, 1e-5f);
        } else if (norm == "ln") {
            x = std::dynamic_pointer_cast<WholeTensorNorm>(blocks["norm"])->forward(ctx, x);
        }

        if (activation == "relu") {
            x = ggml_relu_inplace(gc, x);
        } else if (activation == "tanh") {
            x = ggml_tanh_inplace(gc, x);
        }
        return x;
    }
};

class ResBlockPx : public UnaryBlock {
public:
    explicit ResBlockPx(int64_t dim) {
        blocks["model.0"] = std::shared_ptr<GGMLBlock>(new ConvBlockPx(dim, dim, 3, 1, 1, "in", "relu"));
        blocks["model.1"] = std::shared_ptr<GGMLBlock>(new ConvBlockPx(dim, dim, 3, 1, 1, "in", "none"));
    }

    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
        ggml_tensor* residual = x;
        ggml_tensor* out      = std::dynamic_pointer_cast<ConvBlockPx>(blocks["model.0"])->forward(ctx, x);
        out                   = std::dynamic_pointer_cast<ConvBlockPx>(blocks["model.1"])->forward(ctx, out);
        return ggml_add(ctx->ggml_ctx, out, residual);
    }
};

class ResBlocksPx : public UnaryBlock {
protected:
    int num_blocks;
    std::string cut_group;

public:
    ResBlocksPx(int num_blocks, int64_t dim, const std::string& cut_group)
        : num_blocks(num_blocks), cut_group(cut_group) {
        for (int i = 0; i < num_blocks; i++) {
            blocks["model." + std::to_string(i)] = std::shared_ptr<GGMLBlock>(new ResBlockPx(dim));
        }
    }

    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
        for (int i = 0; i < num_blocks; i++) {
            x = std::dynamic_pointer_cast<ResBlockPx>(blocks["model." + std::to_string(i)])->forward(ctx, x);
            sd::ggml_graph_cut::mark_graph_cut(x, cut_group + "." + std::to_string(i), "x");
        }
        return x;
    }
};

class RGBEncoderPx : public UnaryBlock {
protected:
    int n_downsample;

public:
    RGBEncoderPx(int64_t in_dim, int64_t dim, int n_downsample, int n_res, const std::string& cut_group)
        : n_downsample(n_downsample) {
        blocks["model.0"] = std::shared_ptr<GGMLBlock>(new ConvBlockPx(in_dim, dim, 7, 1, 3, "in", "relu"));
        for (int i = 0; i < n_downsample; i++) {
            blocks["model." + std::to_string(i + 1)] = std::shared_ptr<GGMLBlock>(new ConvBlockPx(dim, 2 * dim, 4, 2, 1, "in", "relu"));
            dim *= 2;
        }
        blocks["model." + std::to_string(n_downsample + 1)] = std::shared_ptr<GGMLBlock>(new ResBlocksPx(n_res, dim, cut_group + ".res"));
    }

    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
        for (int i = 0; i <= n_downsample; i++) {
            x = std::dynamic_pointer_cast<ConvBlockPx>(blocks["model." + std::to_string(i)])->forward(ctx, x);
        }
        return std::dynamic_pointer_cast<ResBlocksPx>(blocks["model." + std::to_string(n_downsample + 1)])->forward(ctx, x);
    }
};

// Shared upsample tail of both decoders (c2pGen.py:254-262 and :63-71). A base class rather
// than a sub-block: upstream holds conv_1..3 directly on each decoder, so an extra block level
// would not match the checkpoint names.
class RGBDecoderTailPx : public GGMLBlock {
public:
    RGBDecoderTailPx(int64_t dim, int64_t out_dim) {
        blocks["conv_1"] = std::shared_ptr<GGMLBlock>(new ConvBlockPx(dim, dim / 2, 5, 1, 2, "ln", "relu"));
        dim /= 2;
        blocks["conv_2"] = std::shared_ptr<GGMLBlock>(new ConvBlockPx(dim, dim / 2, 5, 1, 2, "ln", "relu"));
        dim /= 2;
        blocks["conv_3"] = std::shared_ptr<GGMLBlock>(new ConvBlockPx(dim, out_dim, 7, 1, 3, "none", "tanh"));
    }

    ggml_tensor* forward_tail(GGMLRunnerContext* ctx, ggml_tensor* x) {
        auto gc = ctx->ggml_ctx;
        x       = std::dynamic_pointer_cast<ConvBlockPx>(blocks["conv_1"])->forward(ctx, ggml_upscale(gc, x, 2, GGML_SCALE_MODE_NEAREST));
        x       = std::dynamic_pointer_cast<ConvBlockPx>(blocks["conv_2"])->forward(ctx, ggml_upscale(gc, x, 2, GGML_SCALE_MODE_NEAREST));
        return std::dynamic_pointer_cast<ConvBlockPx>(blocks["conv_3"])->forward(ctx, x);
    }
};

class RGBDecoderPx : public RGBDecoderTailPx {
public:
    RGBDecoderPx(int64_t dim, int64_t out_dim)
        : RGBDecoderTailPx(dim, out_dim) {
        blocks["mod_conv_1"] = std::shared_ptr<GGMLBlock>(new ModulationConvBlock(dim, dim, 3));
        blocks["mod_conv_2"] = std::shared_ptr<GGMLBlock>(new ModulationConvBlock(dim, dim, 3));
    }

    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x, ggml_tensor* code) {
        auto gc = ctx->ggml_ctx;

        auto slice_code = [&](int i) {
            return ggml_view_1d(gc, code, 256, (size_t)i * 256 * sizeof(float));
        };
        auto mc1 = std::dynamic_pointer_cast<ModulationConvBlock>(blocks["mod_conv_1"]);
        auto mc2 = std::dynamic_pointer_cast<ModulationConvBlock>(blocks["mod_conv_2"]);

        // Blocks 2..4 reuse mod_conv_2 rather than mod_conv_3..8 (c2pGen.py:242-251). This
        // matches the released weights, which were trained against this graph; mod_conv_3..8
        // never saw gradients and hold N(0, 0.02) noise. Not a typo -- do not "fix".
        ggml_tensor* residual = x;
        x                     = mc1->forward(ctx, x, slice_code(0));
        if (stage_dump_enabled_) {
            capture(ctx, "mod1", x);
        }
        x = mc2->forward(ctx, x, slice_code(1));
        if (stage_dump_enabled_) {
            capture(ctx, "mod2", x);
        }
        x = ggml_add(gc, x, residual);
        sd::ggml_graph_cut::mark_graph_cut(x, "pixelization.c2p.dec.mod.0", "x");
        for (int pair = 1; pair < 4; pair++) {
            residual = x;
            x        = mc2->forward(ctx, x, slice_code(pair * 2));
            x        = mc2->forward(ctx, x, slice_code(pair * 2 + 1));
            x        = ggml_add(gc, x, residual);
            sd::ggml_graph_cut::mark_graph_cut(x, "pixelization.c2p.dec.mod." + std::to_string(pair), "x");
        }
        return forward_tail(ctx, x);
    }

    // Stage captures are owned by the runner, which reads them back after compute.
    void set_stage_sink(std::vector<std::pair<std::string, ggml_tensor*>>* sink) {
        stage_sink_         = sink;
        stage_dump_enabled_ = sink != nullptr;
    }

protected:
    std::vector<std::pair<std::string, ggml_tensor*>>* stage_sink_ = nullptr;
    bool stage_dump_enabled_                                       = false;

    void capture(GGMLRunnerContext* ctx, const std::string& name, ggml_tensor* x) {
        stage_sink_->push_back({name, snapshot_tensor(ctx, x)});
    }
};

class AliasRGBDecoderPx : public RGBDecoderTailPx {
public:
    AliasRGBDecoderPx(int64_t dim, int64_t out_dim, int n_res)
        : RGBDecoderTailPx(dim, out_dim) {
        blocks["Res_Blocks"] = std::shared_ptr<GGMLBlock>(new ResBlocksPx(n_res, dim, "pixelization.alias.dec.res"));
    }

    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
        x = std::dynamic_pointer_cast<ResBlocksPx>(blocks["Res_Blocks"])->forward(ctx, x);
        return forward_tail(ctx, x);
    }
};

class C2PGenNet : public GGMLBlock {
public:
    C2PGenNet(int64_t in_dim, int64_t out_dim, int64_t dim, int n_downsample, int n_res) {
        blocks["rgb_enc"] = std::shared_ptr<GGMLBlock>(new RGBEncoderPx(in_dim, dim, n_downsample, n_res, "pixelization.c2p.enc"));
        blocks["rgb_dec"] = std::shared_ptr<GGMLBlock>(new RGBDecoderPx(dim << n_downsample, out_dim));
    }

    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x, ggml_tensor* code, std::vector<std::pair<std::string, ggml_tensor*>>* stage_sink) {
        auto dec = std::dynamic_pointer_cast<RGBDecoderPx>(blocks["rgb_dec"]);
        dec->set_stage_sink(stage_sink);

        x = std::dynamic_pointer_cast<RGBEncoderPx>(blocks["rgb_enc"])->forward(ctx, x);
        if (stage_sink != nullptr) {
            stage_sink->push_back({"rgb_enc", snapshot_tensor(ctx, x)});
        }
        return dec->forward(ctx, x, code);
    }
};

class AliasNetPx : public UnaryBlock {
public:
    AliasNetPx(int64_t in_dim, int64_t out_dim, int64_t dim, int n_downsample, int n_res) {
        blocks["rgb_enc"] = std::shared_ptr<GGMLBlock>(new RGBEncoderPx(in_dim, dim, n_downsample, n_res, "pixelization.alias.enc"));
        blocks["rgb_dec"] = std::shared_ptr<GGMLBlock>(new AliasRGBDecoderPx(dim << n_downsample, out_dim, n_res));
    }

    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
        x = std::dynamic_pointer_cast<RGBEncoderPx>(blocks["rgb_enc"])->forward(ctx, x);
        return std::dynamic_pointer_cast<AliasRGBDecoderPx>(blocks["rgb_dec"])->forward(ctx, x);
    }
};

// torchvision VGG19 `features`, truncated after layer 19: PixelBlockEncoder taps conv1_1/2_1/3_1
// /4_1 and never reads past conv4_1, so layers 21..36 are not instantiated.
//
// Taps are POST-ReLU despite indices 0/5/10/19 naming convs. get_features (c2pGen.py:152) stashes
// a reference to x after the conv runs, but torchvision's VGG19 uses nn.ReLU(inplace=True), so the
// next layer (1/6/11/20) rewrites that same storage. The dict therefore observes relu(conv(x)).
// Verified against the fixture: every PyTorch tap is non-negative.
//
// These weights come from `c2p.pb_enc.vgg.*`, NOT the standalone `vgg.features.*` group. C2PGen
// initializes this VGG from pixelart_vgg19.pth but load_state_dict then overwrites it with the
// trained copy in 160_net_G_A.pth; the two differ (~0.3 absolute), so the group is not redundant.
class VGG19TapsPx : public GGMLBlock {
protected:
    // conv index -> (in, out). Zero padding: torchvision uses nn.Conv2d(padding=1) directly.
    static const std::vector<std::pair<int, std::pair<int64_t, int64_t>>>& conv_specs() {
        static const std::vector<std::pair<int, std::pair<int64_t, int64_t>>> specs = {
            {0, {3, 64}},
            {2, {64, 64}},
            {5, {64, 128}},
            {7, {128, 128}},
            {10, {128, 256}},
            {12, {256, 256}},
            {14, {256, 256}},
            {16, {256, 256}},
            {19, {256, 512}},
        };
        return specs;
    }

public:
    VGG19TapsPx() {
        for (const auto& spec : conv_specs()) {
            blocks[std::to_string(spec.first)] = std::shared_ptr<GGMLBlock>(
                new Conv2d(spec.second.first, spec.second.second, {3, 3}, {1, 1}, {1, 1}));
        }
    }

    struct Taps {
        ggml_tensor* conv1_1;
        ggml_tensor* conv2_1;
        ggml_tensor* conv3_1;
        ggml_tensor* conv4_1;
    };

    Taps forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
        auto gc      = ctx->ggml_ctx;
        auto conv    = [&](int i, ggml_tensor* t) {
            return std::dynamic_pointer_cast<Conv2d>(blocks[std::to_string(i)])->forward(ctx, t);
        };
        auto maxpool = [&](ggml_tensor* t) {
            return ggml_pool_2d(gc, t, GGML_OP_POOL_MAX, 2, 2, 2, 2, 0, 0);
        };

        Taps taps;
        taps.conv1_1 = ggml_relu(gc, conv(0, x));
        x            = ggml_relu(gc, conv(2, taps.conv1_1));
        x            = maxpool(x);

        taps.conv2_1 = ggml_relu(gc, conv(5, x));
        x            = ggml_relu(gc, conv(7, taps.conv2_1));
        x            = maxpool(x);

        taps.conv3_1 = ggml_relu(gc, conv(10, x));
        x            = ggml_relu(gc, conv(12, taps.conv3_1));
        x            = ggml_relu(gc, conv(14, x));
        x            = ggml_relu(gc, conv(16, x));
        x            = maxpool(x);

        taps.conv4_1 = ggml_relu(gc, conv(19, x));
        return taps;
    }
};

class PixelBlockEncoderPx : public UnaryBlock {
public:
    PixelBlockEncoderPx(int64_t in_dim, int64_t dim, int64_t style_dim) {
        blocks["vgg"] = std::shared_ptr<GGMLBlock>(new VGG19TapsPx());
        // norm='none' throughout (c2pGen.py:79), unlike RGBEncoder's 'in'.
        blocks["conv1"] = std::shared_ptr<GGMLBlock>(new ConvBlockPx(in_dim, dim, 7, 1, 3, "none", "relu"));
        dim *= 2;
        blocks["conv2"] = std::shared_ptr<GGMLBlock>(new ConvBlockPx(dim, dim, 4, 2, 1, "none", "relu"));
        dim *= 2;
        blocks["conv3"] = std::shared_ptr<GGMLBlock>(new ConvBlockPx(dim, dim, 4, 2, 1, "none", "relu"));
        dim *= 2;
        blocks["conv4"] = std::shared_ptr<GGMLBlock>(new ConvBlockPx(dim, dim, 4, 2, 1, "none", "relu"));
        dim *= 2;
        blocks["model.1"] = std::shared_ptr<GGMLBlock>(new Conv2d(dim, style_dim, {1, 1}));
    }

    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
        auto gc   = ctx->ggml_ctx;
        auto conv = [&](const std::string& name, ggml_tensor* t) {
            return std::dynamic_pointer_cast<ConvBlockPx>(blocks[name])->forward(ctx, t);
        };

        auto taps = std::dynamic_pointer_cast<VGG19TapsPx>(blocks["vgg"])->forward(ctx, x);

        x = ggml_concat(gc, conv("conv1", x), taps.conv1_1, 2);
        x = ggml_concat(gc, conv("conv2", x), taps.conv2_1, 2);
        x = ggml_concat(gc, conv("conv3", x), taps.conv3_1, 2);
        x = ggml_concat(gc, conv("conv4", x), taps.conv4_1, 2);

        // AdaptiveAvgPool2d(1) == mean over H and W. ggml_mean reduces ne[0] only, so fold the
        // [W, H, C, 1] spatial plane into one row per channel first.
        x = ggml_cont(gc, x);
        x = ggml_mean(gc, ggml_reshape_2d(gc, x, x->ne[0] * x->ne[1], x->ne[2]));
        x = ggml_reshape_4d(gc, x, 1, 1, x->ne[1], 1);
        return std::dynamic_pointer_cast<Conv2d>(blocks["model.1"])->forward(ctx, x);
    }
};

// basic_layer.py:172 sets style1 = style0 and blends with a=0, so the interpolation collapses to
// model[3](model[0:3](style0)). The dead branch is upstream's, not ours.
class MLPPx : public UnaryBlock {
public:
    MLPPx(int64_t in_dim, int64_t out_dim, int64_t dim) {
        blocks["model.0.fc"] = std::shared_ptr<GGMLBlock>(new Linear(in_dim, in_dim));
        blocks["model.1.fc"] = std::shared_ptr<GGMLBlock>(new Linear(in_dim, dim));
        blocks["model.2.fc"] = std::shared_ptr<GGMLBlock>(new Linear(dim, dim));
        blocks["model.3.fc"] = std::shared_ptr<GGMLBlock>(new Linear(dim, out_dim));
    }

    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
        auto gc = ctx->ggml_ctx;
        auto fc = [&](const std::string& name, ggml_tensor* t) {
            return std::dynamic_pointer_cast<Linear>(blocks[name])->forward(ctx, t);
        };
        for (int i = 0; i < 3; i++) {
            x = ggml_relu(gc, fc("model." + std::to_string(i) + ".fc", x));
        }
        return fc("model.3.fc", x);
    }
};

class StyleEncoderNet : public UnaryBlock {
public:
    StyleEncoderNet(int64_t in_dim, int64_t dim, int64_t style_dim, int64_t out_dim, int64_t mlp_dim) {
        blocks["pb_enc"] = std::shared_ptr<GGMLBlock>(new PixelBlockEncoderPx(in_dim, dim, style_dim));
        blocks["mlp"]    = std::shared_ptr<GGMLBlock>(new MLPPx(style_dim, out_dim, mlp_dim));
    }

    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
        ggml_tensor* code = std::dynamic_pointer_cast<PixelBlockEncoderPx>(blocks["pb_enc"])->forward(ctx, x);
        code              = ggml_reshape_2d(ctx->ggml_ctx, code, code->ne[2], 1);
        return std::dynamic_pointer_cast<MLPPx>(blocks["mlp"])->forward(ctx, code);
    }
};

// Runs once at load when --pixelization-ref overrides the baked default_style_code, never per
// tile. The returned code is RAW (absmax ~8.4e8) to match Task 1's baked convention: normalizing
// it shrinks ModulationConvBlock's demodulation sum until its internal 1e-8 eps matters.
struct StyleEncoderRunner : public GGMLRunner {
    std::unique_ptr<StyleEncoderNet> net;

    StyleEncoderRunner(ggml_backend_t backend,
                       const String2TensorStorage& tensor_storage_map      = {},
                       std::shared_ptr<RunnerWeightManager> weight_manager = nullptr)
        : GGMLRunner(backend, weight_manager),
          net(std::make_unique<StyleEncoderNet>(3, 64, 256, 2048, 256)) {
        net->init(params_ctx, tensor_storage_map, "c2p");
    }

    std::string get_desc() override {
        return "style_encoder";
    }

    void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors) {
        net->get_param_tensors(tensors, "c2p");
    }

    ggml_cgraph* build_graph(const sd::Tensor<float>& x_tensor) {
        constexpr int kGraphNodes = 1 << 12;
        ggml_cgraph* gf           = new_graph_custom(kGraphNodes);
        ggml_tensor* x            = make_input(x_tensor);

        auto runner_ctx  = get_context();
        ggml_tensor* out = net->forward(&runner_ctx, x);
        ggml_build_forward_expand(gf, out);
        return gf;
    }

    // ref_image must be greyscale replicated across 3 channels (pixelization.py:46), not RGB.
    std::vector<float> compute(const int n_threads, const sd::Tensor<float>& ref_image) {
        auto get_graph = [&]() -> ggml_cgraph* { return build_graph(ref_image); };
        auto result    = take_or_empty(GGMLRunner::compute<float>(get_graph, n_threads, false, false, false));
        if (result.empty()) {
            return {};
        }
        std::vector<float> code(result.data(), result.data() + result.numel());
        dump_stage("style_code", result);
        return code;
    }
};

struct C2PGenRunner : public GGMLRunner {
    std::unique_ptr<C2PGenNet> net;
    std::vector<std::pair<std::string, ggml_tensor*>> stage_tensors;

    C2PGenRunner(ggml_backend_t backend,
                 const String2TensorStorage& tensor_storage_map      = {},
                 std::shared_ptr<RunnerWeightManager> weight_manager = nullptr)
        : GGMLRunner(backend, weight_manager),
          net(std::make_unique<C2PGenNet>(3, 3, 64, 2, 4)) {
        net->init(params_ctx, tensor_storage_map, "c2p");
    }

    std::string get_desc() override {
        return "c2pgen";
    }

    void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors) {
        net->get_param_tensors(tensors, "c2p");
    }

    ggml_cgraph* build_graph(const sd::Tensor<float>& x_tensor, const sd::Tensor<float>& code_tensor) {
        constexpr int kGraphNodes = 1 << 16;  // 65k
        ggml_cgraph* gf           = new_graph_custom(kGraphNodes);
        ggml_tensor* x            = make_input(x_tensor);
        ggml_tensor* code         = make_input(code_tensor);

        stage_tensors.clear();
        auto runner_ctx  = get_context();
        ggml_tensor* out = net->forward(&runner_ctx, x, code, stage_dump_dir() != nullptr ? &stage_tensors : nullptr);
        for (const auto& stage : stage_tensors) {
            ggml_build_forward_expand(gf, stage.second);
        }
        ggml_build_forward_expand(gf, out);
        return gf;
    }

    sd::Tensor<float> compute(const int n_threads,
                              const sd::Tensor<float>& x,
                              const std::vector<float>& code) {
        GGML_ASSERT(code.size() == 2048);
        sd::Tensor<float> code_tensor({(int64_t)code.size()}, code);
        auto get_graph = [&]() -> ggml_cgraph* { return build_graph(x, code_tensor); };
        auto result    = restore_trailing_singleton_dims(GGMLRunner::compute<float>(get_graph, n_threads, false, false, false), x.dim());
        for (const auto& stage : stage_tensors) {
            // ggml_n_dims drops the trailing batch of 1, which the [N, C, H, W] dump needs back.
            dump_stage(stage.first, restore_trailing_singleton_dims(sd::make_sd_tensor_from_ggml<float>(stage.second), 4));
        }
        dump_stage("dec_out", result);
        return result;
    }
};

struct AliasNetRunner : public GGMLRunner {
    std::unique_ptr<AliasNetPx> net;

    AliasNetRunner(ggml_backend_t backend,
                   const String2TensorStorage& tensor_storage_map      = {},
                   std::shared_ptr<RunnerWeightManager> weight_manager = nullptr)
        : GGMLRunner(backend, weight_manager),
          net(std::make_unique<AliasNetPx>(3, 3, 64, 2, 3)) {
        net->init(params_ctx, tensor_storage_map, "alias");
    }

    std::string get_desc() override {
        return "aliasnet";
    }

    void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors) {
        net->get_param_tensors(tensors, "alias");
    }

    ggml_cgraph* build_graph(const sd::Tensor<float>& x_tensor) {
        constexpr int kGraphNodes = 1 << 16;  // 65k
        ggml_cgraph* gf           = new_graph_custom(kGraphNodes);
        ggml_tensor* x            = make_input(x_tensor);

        auto runner_ctx  = get_context();
        ggml_tensor* out = net->forward(&runner_ctx, x);
        ggml_build_forward_expand(gf, out);
        return gf;
    }

    sd::Tensor<float> compute(const int n_threads,
                              const sd::Tensor<float>& x) {
        auto get_graph = [&]() -> ggml_cgraph* { return build_graph(x); };
        auto result    = restore_trailing_singleton_dims(GGMLRunner::compute<float>(get_graph, n_threads, false, false, false), x.dim());
        dump_stage("alias_out", result);
        return result;
    }
};

}  // namespace pixelization

#endif  // __PIXELIZATION_HPP__
