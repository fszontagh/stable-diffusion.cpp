#ifndef __PIXELIZATION_HPP__
#define __PIXELIZATION_HPP__

#include "core/ggml_extend.hpp"

namespace pixelization {

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

}  // namespace pixelization

#endif  // __PIXELIZATION_HPP__
