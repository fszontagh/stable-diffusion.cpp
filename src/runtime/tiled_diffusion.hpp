#ifndef __TILED_DIFFUSION_HPP__
#define __TILED_DIFFUSION_HPP__

#include <algorithm>
#include <cmath>

#include "stable-diffusion.h"

namespace sd {
    namespace tiling {

        // Default window size for tiled diffusion, in latent pixels (64 -> 512px at a
        // scale factor of 8). Larger than the VAE default because a diffusion window
        // that is far below the model's native resolution composes badly.
        constexpr int kDefaultDiffusionTileSize = 64;

        // Shared by the VAE tiling path and the MultiDiffusion sampling path so the
        // meaning of tile_size_* / rel_size_* stays identical between them.
        inline void resolve_tile_sizes(int& tile_size_x,
                                       int& tile_size_y,
                                       float& tile_overlap,
                                       const sd_tiling_params_t& params,
                                       int64_t latent_x,
                                       int64_t latent_y,
                                       float encoding_factor = 1.0f,
                                       int default_tile_size = 32) {
            tile_overlap       = std::max(std::min(params.target_overlap, 0.5f), 0.0f);
            auto get_tile_size = [&](int requested_size, float factor, int64_t latent_size) {
                const int min_tile_dimension = 4;
                int tile_size                = default_tile_size;
                // factor <= 1 means simple fraction of the latent dimension
                // factor > 1 means number of tiles across that dimension
                if (factor > 0.f) {
                    if (factor > 1.0)
                        factor = 1 / (factor - factor * tile_overlap + tile_overlap);
                    tile_size = static_cast<int>(std::round(latent_size * factor));
                } else if (requested_size >= min_tile_dimension) {
                    tile_size = requested_size;
                }
                tile_size = static_cast<int>(tile_size * encoding_factor);
                return std::max(std::min(tile_size, static_cast<int>(latent_size)), min_tile_dimension);
            };

            tile_size_x = get_tile_size(params.tile_size_x, params.rel_size_x, latent_x);
            tile_size_y = get_tile_size(params.tile_size_y, params.rel_size_y, latent_y);
        }

    }  // namespace tiling
}  // namespace sd

#endif  // __TILED_DIFFUSION_HPP__
