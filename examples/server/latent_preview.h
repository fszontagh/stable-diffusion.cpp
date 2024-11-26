#ifndef EXAMPLES_SERVER_LATENT_PREVIEW_H
#define EXAMPLES_SERVER_LATENT_PREVIEW_H

#include "ggml.h"
#include "model.h"

const float flux_latent_rgb_proj[16][3] = {
    {-0.0346, 0.0244, 0.0681},
    {0.0034, 0.0210, 0.0687},
    {0.0275, -0.0668, -0.0433},
    {-0.0174, 0.0160, 0.0617},
    {0.0859, 0.0721, 0.0329},
    {0.0004, 0.0383, 0.0115},
    {0.0405, 0.0861, 0.0915},
    {-0.0236, -0.0185, -0.0259},
    {-0.0245, 0.0250, 0.1180},
    {0.1008, 0.0755, -0.0421},
    {-0.0515, 0.0201, 0.0011},
    {0.0428, -0.0012, -0.0036},
    {0.0817, 0.0765, 0.0749},
    {-0.1264, -0.0522, -0.1103},
    {-0.0280, -0.0881, -0.0499},
    {-0.1262, -0.0982, -0.0778}};

// https://github.com/Stability-AI/sd3.5/blob/main/sd3_impls.py#L228-L246
const float sd3_latent_rgb_proj[16][3] = {
    {-0.0645, 0.0177, 0.1052},
    {0.0028, 0.0312, 0.0650},
    {0.1848, 0.0762, 0.0360},
    {0.0944, 0.0360, 0.0889},
    {0.0897, 0.0506, -0.0364},
    {-0.0020, 0.1203, 0.0284},
    {0.0855, 0.0118, 0.0283},
    {-0.0539, 0.0658, 0.1047},
    {-0.0057, 0.0116, 0.0700},
    {-0.0412, 0.0281, -0.0039},
    {0.1106, 0.1171, 0.1220},
    {-0.0248, 0.0682, -0.0481},
    {0.0815, 0.0846, 0.1207},
    {-0.0120, -0.0055, -0.0867},
    {-0.0749, -0.0634, -0.0456},
    {-0.1418, -0.1457, -0.1259},
};

// https://github.com/comfyanonymous/ComfyUI/blob/master/comfy/latent_formats.py#L32-L38
const float sdxl_latent_rgb_proj[4][3] = {
    {0.3651, 0.4232, 0.4341},
    {-0.2533, -0.0042, 0.1068},
    {0.1076, 0.1111, -0.0362},
    {-0.3165, -0.2492, -0.2188}};

// https://github.com/comfyanonymous/ComfyUI/blob/master/comfy/latent_formats.py#L32-L38
const float sd_latent_rgb_proj[4][3]{
    {0.3512, 0.2297, 0.3227},
    {0.3250, 0.4974, 0.2350},
    {-0.2829, 0.1762, 0.2721},
    {-0.2120, -0.2616, -0.7177}};

inline void proj_latents(struct ggml_tensor* latents, enum SDVersion version, uint8_t* data) {
    const int channel = 3;
    int width         = latents->ne[0];
    int height        = latents->ne[1];
    int dim           = latents->ne[2];

    const float(*latent_rgb_proj)[channel];

    if (dim == 16) {
        // 16 channels VAE -> Flux or SD3

        if (version == VERSION_SD3_5_2B || version == VERSION_SD3_5_8B || version == VERSION_SD3_2B) {
            latent_rgb_proj = sd3_latent_rgb_proj;
        } else if (version == VERSION_FLUX_DEV || version == VERSION_FLUX_SCHNELL || version == VERSION_FLUX_LITE) {
            latent_rgb_proj = flux_latent_rgb_proj;
        } else {
            // unknown model
            return;
        }

    } else if (dim == 4) {
        // 4 channels VAE
        if (version == VERSION_SDXL) {
            latent_rgb_proj = sdxl_latent_rgb_proj;
        } else if (version == VERSION_SD1 || version == VERSION_SD2) {
            latent_rgb_proj = sd_latent_rgb_proj;
        } else {
            // unknown model
            return;
        }
    } else {
        // unknown latent space
        return;
    }
    int data_head = 0;
    for (int j = 0; j < height; j++) {
        for (int i = 0; i < width; i++) {
            int latent_id = (i * latents->nb[0] + j * latents->nb[1]);
            float r = 0, g = 0, b = 0;
            for (int d = 0; d < dim; d++) {
                float value = *(float*)((char*)latents->data + latent_id + d * latents->nb[2]);
                r += value * latent_rgb_proj[d][0];
                g += value * latent_rgb_proj[d][1];
                b += value * latent_rgb_proj[d][2];
            }

            // change range
            r = r * .5 + .5;
            g = g * .5 + .5;
            b = b * .5 + .5;

            // clamp rgb values to [0,1] range
            r = r >= 0 ? r <= 1 ? r : 1 : 0;
            g = g >= 0 ? g <= 1 ? g : 1 : 0;
            b = b >= 0 ? b <= 1 ? b : 1 : 0;

            data[data_head++] = (uint8_t)(r * 255.);
            data[data_head++] = (uint8_t)(g * 255.);
            data[data_head++] = (uint8_t)(b * 255.);
        }
    }
}



#endif