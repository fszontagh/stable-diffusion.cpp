#ifndef __SD_MODEL_TE_AA_CLIP_PREPROCESS_HPP__
#define __SD_MODEL_TE_AA_CLIP_PREPROCESS_HPP__

// AnimateAnyone CLIP-vision preprocessing: a PIL-exact bicubic (a=-0.5)
// antialiased resize + CLIP mean/std normalization.
//
// The reference pipeline does `ref_image.resize((224,224))` (plain PIL resize,
// default resample bicubic in modern Pillow, squashing to the target size and
// ignoring aspect ratio) and then feeds the result through CLIPImageProcessor()
// defaults, whose own resize/crop become a no-op once the image is already
// exactly target-sized. Neither of the fork's existing resize paths reproduces
// that within the aa_test fixture's 1e-3 rel-L2 embedding tolerance:
// - the shared clip_preprocess() (core/util.cpp) resizes nearest-neighbor with
//   an aspect-preserving crop;
// - core/tensor.hpp's InterpolateMode::Bicubic uses PyTorch's a=-0.75 cubic
//   kernel with no intermediate 8-bit rounding, while Pillow uses a=-0.5 AND
//   rounds/clips each separable pass's output to uint8 (libImaging/Resample.c
//   materializes the horizontal pass as an 8-bit image before the vertical
//   pass runs). Both differences were measured to matter (task 4): the kernel
//   coefficient alone costs ~9.7% on the projected embedding, and skipping the
//   intermediate uint8 roundings still leaves ~0.5%.
// This header reproduces Pillow's exact chain. It was validated against the
// reference pipeline's own pixel_values in aa_test's clip-embeds mode (task 4)
// and promoted here (task 9) so the generation-time conditioner
// (AnimateAnyoneVisionConditioner) uses the same preprocessing. Deliberately
// NOT merged into core/tensor.hpp's Bicubic mode, which existing families
// (IP-Adapter/PhotoMaker/SVD) depend on for its PyTorch-matching behavior.

#include <algorithm>
#include <cmath>
#include <vector>

#include "core/tensor.hpp"

namespace AnimateAnyone {

// Pillow's bicubic kernel: cubic convolution with a=-0.5 (Catmull-Rom family),
// NOT PyTorch's a=-0.75.
inline double aa_pil_bicubic_weight(double x) {
    constexpr double a = -0.5;
    x                  = std::fabs(x);
    if (x <= 1.0) {
        return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0;
    }
    if (x < 2.0) {
        return ((a * x - 5.0 * a) * x + 8.0 * a) * x - 4.0 * a;
    }
    return 0.0;
}

struct AaResizeContributor {
    int64_t index;
    double weight;
};

// Antialias-widened (filter support scales with the downsample ratio)
// contributor list per output index, for one axis.
inline std::vector<std::vector<AaResizeContributor>> aa_make_resize_contributors(int64_t in_size, int64_t out_size) {
    std::vector<std::vector<AaResizeContributor>> contributors(static_cast<size_t>(out_size));
    const double scale        = static_cast<double>(in_size) / static_cast<double>(out_size);
    const double filter_scale = std::max(1.0, scale);
    const double support      = 2.0 * filter_scale;

    for (int64_t out = 0; out < out_size; ++out) {
        const double center = (static_cast<double>(out) + 0.5) * scale - 0.5;
        int64_t start       = static_cast<int64_t>(std::ceil(center - support));
        int64_t end         = static_cast<int64_t>(std::floor(center + support));
        double weight_sum   = 0.0;
        auto& axis_contribs = contributors[static_cast<size_t>(out)];
        for (int64_t in = start; in <= end; ++in) {
            double weight = aa_pil_bicubic_weight((center - static_cast<double>(in)) / filter_scale);
            if (weight == 0.0) {
                continue;
            }
            int64_t clamped = std::min(std::max<int64_t>(in, 0), in_size - 1);
            axis_contribs.push_back({clamped, weight});
            weight_sum += weight;
        }
        if (std::fabs(weight_sum) > 1e-12) {
            for (auto& c : axis_contribs) {
                c.weight /= weight_sum;
            }
        }
    }
    return contributors;
}

// Round-half-to-even (std::nearbyint under the default FE_TONEAREST rounding
// mode), not round-half-up: empirically matches Pillow's actual output one tie
// case closer on the task-4 fixture (4 differing pixels out of 150528 vs 7,
// all off by exactly 1/255) than floor(v+0.5) round-half-up.
inline float aa_round_clip_u8(double v) {
    double rounded = std::nearbyint(v);
    return static_cast<float>(std::min(std::max(rounded, 0.0), 255.0));
}

// Separable two-pass resize of `image_0_255` ([W,H,C,1], values in [0,255]) to
// `size`x`size`, matching PIL's bicubic (a=-0.5) antialiased downsampling,
// including the round/clip to a uint8-equivalent image after EACH pass (Pillow
// materializes the horizontal pass as an 8-bit image before the vertical pass).
inline sd::Tensor<float> aa_pil_bicubic_resize(const sd::Tensor<float>& image_0_255, int size) {
    int64_t W = image_0_255.shape()[0], H = image_0_255.shape()[1], C = image_0_255.shape()[2];
    auto row_contribs = aa_make_resize_contributors(W, size);  // horizontal axis
    auto col_contribs = aa_make_resize_contributors(H, size);  // vertical axis

    sd::Tensor<float> tmp({size, H, C, 1});
    for (int64_t y = 0; y < H; ++y) {
        for (int64_t c = 0; c < C; ++c) {
            for (int64_t ox = 0; ox < size; ++ox) {
                double acc = 0.0;
                for (const auto& con : row_contribs[static_cast<size_t>(ox)]) {
                    acc += con.weight * image_0_255.index(con.index, y, c, 0);
                }
                tmp.index(ox, y, c, 0) = aa_round_clip_u8(acc);
            }
        }
    }

    sd::Tensor<float> out({size, size, C, 1});
    for (int64_t oy = 0; oy < size; ++oy) {
        for (int64_t c = 0; c < C; ++c) {
            for (int64_t ox = 0; ox < size; ++ox) {
                double acc = 0.0;
                for (const auto& con : col_contribs[static_cast<size_t>(oy)]) {
                    acc += con.weight * tmp.index(ox, con.index, c, 0);
                }
                out.index(ox, oy, c, 0) = aa_round_clip_u8(acc);
            }
        }
    }
    return out;
}

// CLIP-preprocesses `image_0_255` ([W,H,3,1], values in [0,255] - raw byte
// values, NOT pre-divided by 255: the resize rounds/clips to uint8 between
// passes, matching PIL) to a `size`x`size` tile with CLIP's mean/std
// normalization.
inline sd::Tensor<float> aa_clip_preprocess(const sd::Tensor<float>& image_0_255, int size) {
    sd::Tensor<float> resized_0_255 = aa_pil_bicubic_resize(image_0_255, size);
    sd::Tensor<float> scaled        = resized_0_255 / 255.0f;  // [0,255] -> [0,1]

    // Same constants as core/util.cpp's clip_preprocess (CLIP's fixed mean/std).
    sd::Tensor<float> mean({1, 1, 3, 1}, {0.48145466f, 0.4578275f, 0.40821073f});
    sd::Tensor<float> std_dev({1, 1, 3, 1}, {0.26862954f, 0.26130258f, 0.27577711f});
    return (scaled - mean) / std_dev;
}

}  // namespace AnimateAnyone

#endif  // __SD_MODEL_TE_AA_CLIP_PREPROCESS_HPP__
