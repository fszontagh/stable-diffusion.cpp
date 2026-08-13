#ifndef __REGIONAL_PROMPT_HPP__
#define __REGIONAL_PROMPT_HPP__

#include <algorithm>
#include <cmath>
#include <vector>

#include "core/tensor.hpp"
#include "stable-diffusion.h"

namespace sd {
    namespace regional {

        // Rasterizes one region rectangle into a latent-resolution weight plane.
        // The feather ramp is applied inside the rectangle so neighbouring regions
        // cross-fade instead of meeting at a hard edge.
        inline Tensor<float> rasterize_region(const sd_region_t& region,
                                              float feather,
                                              int64_t latent_w,
                                              int64_t latent_h) {
            Tensor<float> mask = Tensor<float>::zeros({latent_w, latent_h, 1, 1});

            float x0 = region.x * static_cast<float>(latent_w);
            float y0 = region.y * static_cast<float>(latent_h);
            float x1 = (region.x + region.width) * static_cast<float>(latent_w);
            float y1 = (region.y + region.height) * static_cast<float>(latent_h);

            auto smootherstep = [](float t) -> float {
                t = std::max(0.f, std::min(1.f, t));
                return t * t * t * (t * (6.0f * t - 15.0f) + 10.0f);
            };
            auto axis_falloff = [&](float p, float lo, float hi) -> float {
                if (p < lo || p >= hi) {
                    return 0.f;
                }
                if (feather <= 0.f) {
                    return 1.f;
                }
                return std::min(smootherstep((p - lo) / feather),
                                smootherstep((hi - p) / feather));
            };

            for (int64_t py = 0; py < latent_h; ++py) {
                float fy = axis_falloff(static_cast<float>(py) + 0.5f, y0, y1);
                if (fy <= 0.f) {
                    continue;
                }
                for (int64_t px = 0; px < latent_w; ++px) {
                    float fx = axis_falloff(static_cast<float>(px) + 0.5f, x0, x1);
                    if (fx <= 0.f) {
                        continue;
                    }
                    mask[py * latent_w + px] = region.weight * fx * fy;
                }
            }
            return mask;
        }

        // Turns the raw per-region weight planes into blend weights used to mix the
        // per-region model predictions.
        //
        // `region_masks` holds one plane per region, already scaled by that region's
        // weight. `base_mask` is the plane for the main prompt and starts out
        // uniformly `base_weight`. Both are modified in place.
        //
        // TODO(user): decide the combination policy. See the notes in the chat.
        inline void normalize_region_weights(std::vector<Tensor<float>>& region_masks,
                                             Tensor<float>& base_mask) {
            (void)region_masks;
            (void)base_mask;
        }

    }  // namespace regional
}  // namespace sd

#endif  // __REGIONAL_PROMPT_HPP__
