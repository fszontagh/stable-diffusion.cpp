#ifndef __SD_MODEL_DIFFUSION_JOYAI_IMAGE_HPP__
#define __SD_MODEL_DIFFUSION_JOYAI_IMAGE_HPP__

#include <memory>

#include "core/util.h"
#include "model/common/block.hpp"
#include "model/common/rope.hpp"
#include "model/diffusion/dit.hpp"
#include "model/diffusion/flux.hpp"
#include "model/diffusion/model.hpp"
#include "model_loader.h"

// JoyAI-Image-Edit(-Plus) (JD.com). Double-stream MMDiT with WAN-style
// modulation tables and 3-axis RoPE, WAN 2.1 VAE latents, Qwen3-VL conditioning.
// Diffusers reference: JoyImageEditPlusTransformer3DModel. The I/O prelude
// (Conv3d patch embed + time/text condition embedder + time_projection) mirrors
// WAN; the joint-attention block mirrors Qwen-Image with fused QKV.
namespace JoyImage {
    constexpr int JOYAI_IMAGE_GRAPH_SIZE = 20480;

    struct JoyImageConfig {
        std::tuple<int, int, int> patch_size = {1, 2, 2};
        int64_t in_channels                  = 16;
        int64_t out_channels                 = 16;
        int64_t hidden_size                  = 4096;
        int num_layers                       = 40;
        int64_t num_attention_heads          = 32;
        int64_t attention_head_dim           = 128;
        int64_t text_dim                     = 4096;
        int time_freq_dim                    = 256;
        float mlp_width_ratio                = 4.0f;
        int theta                            = 10000;
        std::vector<int> axes_dim            = {16, 56, 56};
        int axes_dim_sum                     = 128;

        static JoyImageConfig detect_from_weights(const String2TensorStorage& tensor_storage_map, const std::string& prefix) {
            JoyImageConfig config;
            config.num_layers = 0;
            for (const auto& [name, tensor_storage] : tensor_storage_map) {
                if (!starts_with(name, prefix)) {
                    continue;
                }
                if (ends_with(name, "proj_out.weight")) {
                    config.hidden_size = tensor_storage.ne[0];
                    config.out_channels = tensor_storage.ne[1] /
                                          (std::get<0>(config.patch_size) * std::get<1>(config.patch_size) * std::get<2>(config.patch_size));
                }
                if (ends_with(name, "condition_embedder.text_embedder.linear_1.weight")) {
                    config.text_dim = tensor_storage.ne[0];
                }
                if (ends_with(name, "double_blocks.0.attn.img_attn_q_norm.weight")) {
                    config.attention_head_dim = tensor_storage.ne[0];
                }
                size_t pos = name.find("double_blocks.");
                if (pos == std::string::npos) {
                    continue;
                }
                auto items = split_string(name.substr(pos), '.');
                if (items.size() > 1) {
                    int block_index = atoi(items[1].c_str());
                    if (block_index + 1 > config.num_layers) {
                        config.num_layers = block_index + 1;
                    }
                }
            }
            if (config.attention_head_dim > 0) {
                config.num_attention_heads = config.hidden_size / config.attention_head_dim;
            }
            config.in_channels = config.out_channels;
            LOG_DEBUG("joyai_image: num_layers = %d, hidden_size = %" PRId64 ", heads = %" PRId64 ", head_dim = %" PRId64 ", in/out ch = %" PRId64 ", text_dim = %" PRId64,
                      config.num_layers, config.hidden_size, config.num_attention_heads,
                      config.attention_head_dim, config.out_channels, config.text_dim);
            return config;
        }
    };

    // Joint double-stream attention with fused QKV per stream (img_attn_qkv /
    // txt_attn_qkv). Joint order is [img, txt] (img first), matching diffusers.
    struct JoyImageAttention : public GGMLBlock {
    protected:
        int64_t dim_head;
        int64_t num_heads;

    public:
        JoyImageAttention(int64_t dim,
                          int64_t num_attention_heads,
                          int64_t attention_head_dim,
                          float eps = 1e-6f)
            : dim_head(attention_head_dim), num_heads(num_attention_heads) {
            int64_t inner_dim = num_attention_heads * attention_head_dim;

            blocks["img_attn_qkv"]  = std::shared_ptr<GGMLBlock>(new Linear(dim, inner_dim * 3, true));
            blocks["img_attn_q_norm"] = std::shared_ptr<GGMLBlock>(new RMSNorm(attention_head_dim, eps));
            blocks["img_attn_k_norm"] = std::shared_ptr<GGMLBlock>(new RMSNorm(attention_head_dim, eps));
            blocks["img_attn_proj"] = std::shared_ptr<GGMLBlock>(new Linear(inner_dim, dim, true));

            blocks["txt_attn_qkv"]  = std::shared_ptr<GGMLBlock>(new Linear(dim, inner_dim * 3, true));
            blocks["txt_attn_q_norm"] = std::shared_ptr<GGMLBlock>(new RMSNorm(attention_head_dim, eps));
            blocks["txt_attn_k_norm"] = std::shared_ptr<GGMLBlock>(new RMSNorm(attention_head_dim, eps));
            blocks["txt_attn_proj"] = std::shared_ptr<GGMLBlock>(new Linear(inner_dim, dim, true));
        }

        std::pair<ggml_tensor*, ggml_tensor*> forward(GGMLRunnerContext* ctx,
                                                      ggml_tensor* img,
                                                      ggml_tensor* txt,
                                                      ggml_tensor* pe,
                                                      ggml_tensor* mask = nullptr) {
            // img: [N, n_img_token, hidden_size]
            // txt: [N, n_txt_token, hidden_size]
            // pe covers [img tokens, txt tokens] in that order.
            auto img_qkv    = std::dynamic_pointer_cast<Linear>(blocks["img_attn_qkv"]);
            auto img_q_norm = std::dynamic_pointer_cast<UnaryBlock>(blocks["img_attn_q_norm"]);
            auto img_k_norm = std::dynamic_pointer_cast<UnaryBlock>(blocks["img_attn_k_norm"]);
            auto img_proj   = std::dynamic_pointer_cast<Linear>(blocks["img_attn_proj"]);
            auto txt_qkv    = std::dynamic_pointer_cast<Linear>(blocks["txt_attn_qkv"]);
            auto txt_q_norm = std::dynamic_pointer_cast<UnaryBlock>(blocks["txt_attn_q_norm"]);
            auto txt_k_norm = std::dynamic_pointer_cast<UnaryBlock>(blocks["txt_attn_k_norm"]);
            auto txt_proj   = std::dynamic_pointer_cast<Linear>(blocks["txt_attn_proj"]);

            int64_t N           = img->ne[2];
            int64_t n_img_token = img->ne[1];
            int64_t n_txt_token = txt->ne[1];

            auto split_qkv = [&](ggml_tensor* qkv, int64_t n_token) {
                auto parts = ggml_ext_chunk(ctx->ggml_ctx, qkv, 3, 0);  // each [N, n_token, inner_dim]
                auto q     = ggml_reshape_4d(ctx->ggml_ctx, ggml_cont(ctx->ggml_ctx, parts[0]), dim_head, num_heads, n_token, N);
                auto k     = ggml_reshape_4d(ctx->ggml_ctx, ggml_cont(ctx->ggml_ctx, parts[1]), dim_head, num_heads, n_token, N);
                auto v     = ggml_reshape_4d(ctx->ggml_ctx, ggml_cont(ctx->ggml_ctx, parts[2]), dim_head, num_heads, n_token, N);
                return std::make_tuple(q, k, v);
            };

            auto [img_q, img_k, img_v] = split_qkv(img_qkv->forward(ctx, img), n_img_token);
            img_q                      = img_q_norm->forward(ctx, img_q);
            img_k                      = img_k_norm->forward(ctx, img_k);

            auto [txt_q, txt_k, txt_v] = split_qkv(txt_qkv->forward(ctx, txt), n_txt_token);
            txt_q                      = txt_q_norm->forward(ctx, txt_q);
            txt_k                      = txt_k_norm->forward(ctx, txt_k);

            // joint attention, img first
            auto q = ggml_concat(ctx->ggml_ctx, img_q, txt_q, 2);
            auto k = ggml_concat(ctx->ggml_ctx, img_k, txt_k, 2);
            auto v = ggml_concat(ctx->ggml_ctx, img_v, txt_v, 2);

            auto attn = Rope::attention(ctx, q, k, v, pe, mask, 1.0f / 128.f);  // [N, n_img+n_txt, inner_dim]

            auto img_attn_out = ggml_view_3d(ctx->ggml_ctx, attn, attn->ne[0], n_img_token, attn->ne[2],
                                             attn->nb[1], attn->nb[2], 0);
            auto txt_attn_out = ggml_view_3d(ctx->ggml_ctx, attn, attn->ne[0], n_txt_token, attn->ne[2],
                                             attn->nb[1], attn->nb[2], n_img_token * attn->nb[1]);
            img_attn_out      = ggml_cont(ctx->ggml_ctx, img_attn_out);
            txt_attn_out      = ggml_cont(ctx->ggml_ctx, txt_attn_out);

            img_attn_out = img_proj->forward(ctx, img_attn_out);
            txt_attn_out = txt_proj->forward(ctx, txt_attn_out);

            return {img_attn_out, txt_attn_out};
        }
    };

    // Double-stream block: WAN-style modulate_table modulation (table + global
    // vec, chunked into 6), Qwen-style joint attention and per-stream GELU MLP.
    class JoyImageTransformerBlock : public GGMLBlock {
    protected:
        int64_t dim;

        void init_params(ggml_context* ctx, const String2TensorStorage& tensor_storage_map = {}, const std::string prefix = "") override {
            // Kept in F32: these are added to the conditioning vector directly,
            // and ggml's binary broadcast ops reject quantized operands.
            params["img_mod.modulate_table"] = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, dim, 6, 1);
            params["txt_mod.modulate_table"] = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, dim, 6, 1);
        }

    public:
        JoyImageTransformerBlock(int64_t dim,
                                 int64_t num_attention_heads,
                                 int64_t attention_head_dim,
                                 float mlp_width_ratio = 4.0f,
                                 float eps             = 1e-6f)
            : dim(dim) {
            int mlp_ratio = static_cast<int>(mlp_width_ratio);

            blocks["img_norm1"] = std::shared_ptr<GGMLBlock>(new LayerNorm(dim, eps, false));
            blocks["img_norm2"] = std::shared_ptr<GGMLBlock>(new LayerNorm(dim, eps, false));
            blocks["img_mlp"]   = std::shared_ptr<GGMLBlock>(new FeedForward(dim, dim, mlp_ratio, FeedForward::Activation::GELU, true));

            blocks["txt_norm1"] = std::shared_ptr<GGMLBlock>(new LayerNorm(dim, eps, false));
            blocks["txt_norm2"] = std::shared_ptr<GGMLBlock>(new LayerNorm(dim, eps, false));
            blocks["txt_mlp"]   = std::shared_ptr<GGMLBlock>(new FeedForward(dim, dim, mlp_ratio, FeedForward::Activation::GELU, true));

            blocks["attn"] = std::shared_ptr<GGMLBlock>(new JoyImageAttention(dim, num_attention_heads, attention_head_dim, eps));
        }

        // es = chunk(modulate_table + vec, 6): shift1,scale1,gate1,shift2,scale2,gate2
        std::vector<ggml_tensor*> mod_vecs(ggml_context* ctx, ggml_tensor* table, ggml_tensor* vec) {
            auto e = ggml_add(ctx, vec, table);  // [dim, 6, N]  (table [dim,6,1] broadcasts over N)
            return ggml_ext_chunk(ctx, e, 6, 1);  // each [dim, 1, N]
        }

        std::pair<ggml_tensor*, ggml_tensor*> forward(GGMLRunnerContext* ctx,
                                                      ggml_tensor* img,
                                                      ggml_tensor* txt,
                                                      ggml_tensor* vec,
                                                      ggml_tensor* pe,
                                                      ggml_tensor* mask = nullptr) {
            // img: [N, n_img_token, dim], txt: [N, n_txt_token, dim]
            // vec: [N, 6, dim]  (global condition, same for all blocks)
            auto img_norm1 = std::dynamic_pointer_cast<LayerNorm>(blocks["img_norm1"]);
            auto img_norm2 = std::dynamic_pointer_cast<LayerNorm>(blocks["img_norm2"]);
            auto img_mlp   = std::dynamic_pointer_cast<FeedForward>(blocks["img_mlp"]);
            auto txt_norm1 = std::dynamic_pointer_cast<LayerNorm>(blocks["txt_norm1"]);
            auto txt_norm2 = std::dynamic_pointer_cast<LayerNorm>(blocks["txt_norm2"]);
            auto txt_mlp   = std::dynamic_pointer_cast<FeedForward>(blocks["txt_mlp"]);
            auto attn      = std::dynamic_pointer_cast<JoyImageAttention>(blocks["attn"]);

            auto img_es = mod_vecs(ctx->ggml_ctx, params["img_mod.modulate_table"], vec);
            auto txt_es = mod_vecs(ctx->ggml_ctx, params["txt_mod.modulate_table"], vec);

            auto flat = [&](ggml_tensor* m) { return ggml_reshape_2d(ctx->ggml_ctx, m, m->ne[0], m->ne[2]); };  // [dim,1,N] -> [dim,N]

            auto img_modulated = Flux::modulate(ctx->ggml_ctx, img_norm1->forward(ctx, img), flat(img_es[0]), flat(img_es[1]));
            auto txt_modulated = Flux::modulate(ctx->ggml_ctx, txt_norm1->forward(ctx, txt), flat(txt_es[0]), flat(txt_es[1]));

            auto [img_attn, txt_attn] = attn->forward(ctx, img_modulated, txt_modulated, pe, mask);

            img = ggml_add(ctx->ggml_ctx, img, ggml_mul(ctx->ggml_ctx, img_attn, ggml_reshape_3d(ctx->ggml_ctx, flat(img_es[2]), dim, 1, img->ne[2])));
            txt = ggml_add(ctx->ggml_ctx, txt, ggml_mul(ctx->ggml_ctx, txt_attn, ggml_reshape_3d(ctx->ggml_ctx, flat(txt_es[2]), dim, 1, txt->ne[2])));

            auto img_ffn = img_mlp->forward(ctx, Flux::modulate(ctx->ggml_ctx, img_norm2->forward(ctx, img), flat(img_es[3]), flat(img_es[4])));
            auto txt_ffn = txt_mlp->forward(ctx, Flux::modulate(ctx->ggml_ctx, txt_norm2->forward(ctx, txt), flat(txt_es[3]), flat(txt_es[4])));

            img = ggml_add(ctx->ggml_ctx, img, ggml_mul(ctx->ggml_ctx, img_ffn, ggml_reshape_3d(ctx->ggml_ctx, flat(img_es[5]), dim, 1, img->ne[2])));
            txt = ggml_add(ctx->ggml_ctx, txt, ggml_mul(ctx->ggml_ctx, txt_ffn, ggml_reshape_3d(ctx->ggml_ctx, flat(txt_es[5]), dim, 1, txt->ne[2])));

            return {img, txt};
        }
    };

    class JoyImageModel : public GGMLBlock {
    protected:
        JoyImageConfig config;

    public:
        JoyImageModel() {}
        JoyImageModel(JoyImageConfig config)
            : config(config) {
            int pt = std::get<0>(config.patch_size);
            int ph = std::get<1>(config.patch_size);
            int pw = std::get<2>(config.patch_size);

            blocks["img_in"] = std::shared_ptr<GGMLBlock>(new Conv3d(config.in_channels, config.hidden_size, {pt, ph, pw}, {pt, ph, pw}));

            // condition embedder
            blocks["condition_embedder.time_embedder.linear_1"] = std::shared_ptr<GGMLBlock>(new Linear(config.time_freq_dim, config.hidden_size));
            blocks["condition_embedder.time_embedder.linear_2"] = std::shared_ptr<GGMLBlock>(new Linear(config.hidden_size, config.hidden_size));
            blocks["condition_embedder.time_proj"]              = std::shared_ptr<GGMLBlock>(new Linear(config.hidden_size, config.hidden_size * 6));
            blocks["condition_embedder.text_embedder.linear_1"] = std::shared_ptr<GGMLBlock>(new Linear(config.text_dim, config.hidden_size));
            blocks["condition_embedder.text_embedder.linear_2"] = std::shared_ptr<GGMLBlock>(new Linear(config.hidden_size, config.hidden_size));

            for (int i = 0; i < config.num_layers; i++) {
                blocks["double_blocks." + std::to_string(i)] = std::shared_ptr<GGMLBlock>(
                    new JoyImageTransformerBlock(config.hidden_size, config.num_attention_heads, config.attention_head_dim, config.mlp_width_ratio));
            }

            blocks["norm_out"] = std::shared_ptr<GGMLBlock>(new LayerNorm(config.hidden_size, 1e-6f, false));
            blocks["proj_out"] = std::shared_ptr<GGMLBlock>(new Linear(config.hidden_size, config.out_channels * pt * ph * pw));
        }

        // Conv3d patch embed then flatten to tokens: mirrors WAN patch_embedding.
        ggml_tensor* patchify(GGMLRunnerContext* ctx, ggml_tensor* x, int64_t N) {
            auto img_in = std::dynamic_pointer_cast<Conv3d>(blocks["img_in"]);
            x           = img_in->forward(ctx, x);  // [N*dim, t_len, h_len, w_len]
            x           = ggml_reshape_3d(ctx->ggml_ctx, x, x->ne[0] * x->ne[1] * x->ne[2], x->ne[3] / N, N);  // [N, dim, tokens]
            x           = ggml_ext_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, x, 1, 0, 2, 3));  // [N, tokens, dim]
            return x;
        }

        ggml_tensor* condition(GGMLRunnerContext* ctx, ggml_tensor* timestep, ggml_tensor* context, ggml_tensor** vec_out) {
            auto time_1  = std::dynamic_pointer_cast<Linear>(blocks["condition_embedder.time_embedder.linear_1"]);
            auto time_2  = std::dynamic_pointer_cast<Linear>(blocks["condition_embedder.time_embedder.linear_2"]);
            auto time_pr = std::dynamic_pointer_cast<Linear>(blocks["condition_embedder.time_proj"]);
            auto text_1  = std::dynamic_pointer_cast<Linear>(blocks["condition_embedder.text_embedder.linear_1"]);
            auto text_2  = std::dynamic_pointer_cast<Linear>(blocks["condition_embedder.text_embedder.linear_2"]);

            auto temb = ggml_ext_timestep_embedding(ctx->ggml_ctx, timestep, config.time_freq_dim);  // [N, freq_dim]
            temb      = time_1->forward(ctx, temb);
            temb      = ggml_silu_inplace(ctx->ggml_ctx, temb);
            temb      = time_2->forward(ctx, temb);  // [N, dim]

            auto vec = ggml_silu(ctx->ggml_ctx, temb);
            vec      = time_pr->forward(ctx, vec);                                                            // [N, 6*dim]
            vec      = ggml_reshape_3d(ctx->ggml_ctx, vec, config.hidden_size, 6, vec->ne[1]);                // [N, 6, dim]
            *vec_out = vec;

            auto txt = text_1->forward(ctx, context);
            txt      = ggml_ext_gelu(ctx->ggml_ctx, txt);
            txt      = text_2->forward(ctx, txt);  // [N, L, dim]
            return txt;
        }

        // x: [W, H, T, N*C]  (Conv3d layout). Returns [N, tokens, ph*pw*pt*C_out].
        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             ggml_tensor* timestep,
                             ggml_tensor* context,
                             ggml_tensor* pe,
                             int64_t N,
                             ggml_tensor* mask = nullptr) {
            ggml_tensor* vec = nullptr;
            auto txt         = condition(ctx, timestep, context, &vec);
            auto img         = patchify(ctx, x, N);

            if (getenv("SDCPP_JOYAI_DEBUG")) {
                LOG_DEBUG("joyai_image dbg: vec ne=[%" PRId64 ",%" PRId64 ",%" PRId64 "] txt ne=[%" PRId64 ",%" PRId64 ",%" PRId64 "] img ne=[%" PRId64 ",%" PRId64 ",%" PRId64 "]",
                          vec->ne[0], vec->ne[1], vec->ne[2],
                          txt->ne[0], txt->ne[1], txt->ne[2],
                          img->ne[0], img->ne[1], img->ne[2]);
            }

            // Cut points let the streaming planner segment the graph; without
            // them the whole model has to be resident at once.
            sd::ggml_graph_cut::mark_graph_cut(img, "joyai_image.prelude", "img");
            sd::ggml_graph_cut::mark_graph_cut(txt, "joyai_image.prelude", "txt");

            // Debug bisection: emit an intermediate stage as the graph output so it
            // can be diffed against the reference implementation.
            const char* stop = getenv("SDCPP_JOYAI_STOP");
            if (stop != nullptr && std::string(stop) == "img_in") {
                return img;
            }

            for (int i = 0; i < config.num_layers; i++) {
                auto block = std::dynamic_pointer_cast<JoyImageTransformerBlock>(blocks["double_blocks." + std::to_string(i)]);
                auto res   = block->forward(ctx, img, txt, vec, pe, mask);
                img        = res.first;
                txt        = res.second;
                sd::ggml_graph_cut::mark_graph_cut(img, "joyai_image.double_blocks." + std::to_string(i), "img");
                sd::ggml_graph_cut::mark_graph_cut(txt, "joyai_image.double_blocks." + std::to_string(i), "txt");
                if (stop != nullptr && std::string(stop) == "block" + std::to_string(i)) {
                    return img;
                }
            }

            auto norm_out = std::dynamic_pointer_cast<LayerNorm>(blocks["norm_out"]);
            auto proj_out = std::dynamic_pointer_cast<Linear>(blocks["proj_out"]);
            img           = norm_out->forward(ctx, img);
            img           = proj_out->forward(ctx, img);  // [N, tokens, C_out*pt*ph*pw]
            return img;
        }
    };

    struct JoyImageRunner : public DiffusionModelRunner {
    public:
        JoyImageConfig config;
        JoyImageModel joyai_image;
        std::vector<float> pe_vec;
        SDVersion version;

        JoyImageRunner(ggml_backend_t backend,
                       const String2TensorStorage& tensor_storage_map      = {},
                       const std::string prefix                            = "",
                       SDVersion version                                   = VERSION_JOYAI_IMAGE_EDIT,
                       std::shared_ptr<RunnerWeightManager> weight_manager = nullptr,
                       const char* model_args                              = nullptr)
            : DiffusionModelRunner(backend, prefix, weight_manager),
              config(JoyImageConfig::detect_from_weights(tensor_storage_map, prefix)),
              version(version) {
            joyai_image = JoyImageModel(config);
            joyai_image.init(params_ctx, tensor_storage_map, prefix);
        }

        std::string get_desc() override {
            return "joyai_image";
        }

        void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors, const std::string& prefix) override {
            joyai_image.get_param_tensors(tensors, prefix);
        }

        // Build RoPE positions: image tokens get 3D (t,h,w) positions per
        // component (temporal band per reference), text tokens get identity
        // (position 0 -> no rotation).
        std::vector<float> gen_pe(int t_len, int h_len, int w_len,
                                  const std::vector<ggml_tensor*>& ref_latents,
                                  int txt_len, int bs) {
            std::vector<std::vector<float>> ids;

            int t_off = 0;
            auto add_component = [&](int t, int h, int w) {
                for (int ti = 0; ti < t; ++ti) {
                    for (int hi = 0; hi < h; ++hi) {
                        for (int wi = 0; wi < w; ++wi) {
                            ids.push_back({static_cast<float>(t_off + ti), static_cast<float>(hi), static_cast<float>(wi)});
                        }
                    }
                }
                t_off += t;
            };

            // Upstream packs the sequence as [refs..., target] (pipeline.py:
            // `latents = cat([ref_vae, noise], dim=1)`), so references take the
            // leading temporal bands and the target takes the last one.
            int ph = std::get<1>(config.patch_size);
            int pw = std::get<2>(config.patch_size);
            for (ggml_tensor* ref : ref_latents) {
                // Reference latents are [W, H, C, N]; ne[2] is channels, not
                // time. Each still image contributes exactly one temporal slice.
                int rh = static_cast<int>(ref->ne[1]) / ph;
                int rw = static_cast<int>(ref->ne[0]) / pw;
                add_component(1, rh, rw);
            }
            add_component(t_len, h_len, w_len);

            for (int i = 0; i < txt_len; ++i) {
                ids.push_back({0.f, 0.f, 0.f});  // identity rotation for text tokens
            }

            std::vector<std::vector<float>> ids_bs;
            ids_bs.reserve(ids.size() * bs);
            for (int b = 0; b < bs; ++b) {
                for (auto& id : ids) {
                    ids_bs.push_back(id);
                }
            }
            return Rope::embed_nd(ids_bs, bs, static_cast<float>(config.theta), config.axes_dim);
        }

        ggml_cgraph* build_graph(const sd::Tensor<float>& x_tensor,
                                 const sd::Tensor<float>& timesteps_tensor,
                                 const sd::Tensor<float>& context_tensor,
                                 const std::vector<sd::Tensor<float>>& ref_latents_tensor = {}) {
            ggml_cgraph* gf        = new_graph_custom(JOYAI_IMAGE_GRAPH_SIZE);
            ggml_tensor* x         = make_input(x_tensor);
            ggml_tensor* timesteps = make_input(timesteps_tensor);
            GGML_ASSERT(!context_tensor.empty());
            ggml_tensor* context = make_input(context_tensor);

            std::vector<ggml_tensor*> ref_latents;
            ref_latents.reserve(ref_latents_tensor.size());
            for (const auto& ref_latent_tensor : ref_latents_tensor) {
                ref_latents.push_back(make_input(ref_latent_tensor));
            }

            int64_t W = x->ne[0];
            int64_t H = x->ne[1];
            int64_t C = x->ne[2];
            int64_t N = x->ne[3];

            int pt = std::get<0>(config.patch_size);
            int ph = std::get<1>(config.patch_size);
            int pw = std::get<2>(config.patch_size);

            // reshape image latent [W,H,C,N] -> [W,H,T=1,C] for Conv3d (N==1)
            GGML_ASSERT(N == 1);
            ggml_tensor* x3d = ggml_reshape_4d(compute_ctx, x, W, H, 1, C);

            // Upstream order is [refs..., target] along the time axis, each with
            // its own temporal RoPE band (see gen_pe). This equal-resolution path
            // mirrors the non-Plus model; mixed resolutions need the padded
            // per-component packing.
            if (!ref_latents.empty()) {
                ggml_tensor* packed = nullptr;
                for (ggml_tensor* ref : ref_latents) {
                    GGML_ASSERT(ref->ne[0] == W && ref->ne[1] == H);
                    auto ref3d = ggml_reshape_4d(compute_ctx, ref, ref->ne[0], ref->ne[1], 1, ref->ne[2]);
                    packed     = packed == nullptr ? ref3d : ggml_concat(compute_ctx, packed, ref3d, 2);
                }
                x3d = ggml_concat(compute_ctx, packed, x3d, 2);
            }

            int h_len = static_cast<int>(H) / ph;
            int w_len = static_cast<int>(W) / pw;
            int t_len = 1;

            pe_vec      = gen_pe(t_len, h_len, w_len, ref_latents, static_cast<int>(context->ne[1]), static_cast<int>(N));
            int pos_len = static_cast<int>(pe_vec.size() / config.axes_dim_sum / 2);

            int64_t expected_tokens = static_cast<int64_t>(1 + ref_latents.size()) * h_len * w_len + context->ne[1];
            LOG_DEBUG("joyai_image: %d img components (%dx%d each) + %" PRId64 " txt tokens, pe positions = %d",
                      1 + static_cast<int>(ref_latents.size()), h_len, w_len, context->ne[1], pos_len);
            GGML_ASSERT(pos_len == expected_tokens);
            auto pe     = ggml_new_tensor_4d(compute_ctx, GGML_TYPE_F32, 2, 2, config.axes_dim_sum / 2, pos_len);
            set_backend_tensor_data(pe, pe_vec.data());

            // Debug: dump the conditioning tensor so it can be compared against a
            // reference text-encoder run.
            if (const char* dump = getenv("SDCPP_JOYAI_DUMP_CONTEXT")) {
                static bool written = false;
                if (!written) {
                    written = true;
                    FILE* f = fopen(dump, "wb");
                    if (f != nullptr) {
                        fwrite(context_tensor.data(), sizeof(float), static_cast<size_t>(context_tensor.numel()), f);
                        fclose(f);
                        LOG_INFO("joyai_image: dumped context [%" PRId64 ",%" PRId64 "] to %s",
                                 context->ne[0], context->ne[1], dump);
                    }
                }
            }

            auto runner_ctx = get_context();
            ggml_tensor* out = joyai_image.forward(&runner_ctx, x3d, timesteps, context, pe, N);

            // Debug bisection emits a raw intermediate; skip the output reshaping.
            if (getenv("SDCPP_JOYAI_STOP") != nullptr) {
                ggml_build_forward_expand(gf, out);
                return gf;
            }

            // out: [N, tokens, C_out*pt*ph*pw]. The target is the LAST component,
            // so drop the leading reference tokens before unpatchifying.
            int64_t target_tokens = static_cast<int64_t>(t_len) * h_len * w_len;
            if (out->ne[1] > target_tokens) {
                int64_t ref_tokens = out->ne[1] - target_tokens;
                out                = ggml_cont(compute_ctx, ggml_permute(compute_ctx, out, 0, 2, 1, 3));
                out                = ggml_view_3d(compute_ctx, out, out->ne[0], out->ne[1], target_tokens,
                                                  out->nb[1], out->nb[2], ref_tokens * out->nb[2]);
                out                = ggml_cont(compute_ctx, ggml_permute(compute_ctx, out, 0, 2, 1, 3));
            }
            // proj_out packs each patch as (pt, ph, pw, C) with C fastest, so the
            // channel-last unpatchify layout is the matching one.
            out = DiT::unpatchify_and_crop(compute_ctx, out, static_cast<int>(H), static_cast<int>(W), ph, pw, false);  // [N, C, H, W]

            ggml_build_forward_expand(gf, out);
            return gf;
        }

        sd::Tensor<float> compute(int n_threads,
                                  const sd::Tensor<float>& x,
                                  const sd::Tensor<float>& timesteps,
                                  const sd::Tensor<float>& context,
                                  const std::vector<sd::Tensor<float>>& ref_latents = {}) {
            auto get_graph = [&]() -> ggml_cgraph* {
                return build_graph(x, timesteps, context, ref_latents);
            };
            return restore_trailing_singleton_dims(GGMLRunner::compute<float>(get_graph, n_threads, false, false, false), x.dim());
        }

        sd::Tensor<float> compute(int n_threads,
                                  const DiffusionParams& diffusion_params) override {
            GGML_ASSERT(diffusion_params.x != nullptr);
            GGML_ASSERT(diffusion_params.timesteps != nullptr);
            static const std::vector<sd::Tensor<float>> empty_ref_latents;
            return compute(n_threads,
                           *diffusion_params.x,
                           *diffusion_params.timesteps,
                           tensor_or_empty(diffusion_params.context),
                           diffusion_params.ref_latents && diffusion_params.ref_image_params.pass_to_dit ? *diffusion_params.ref_latents : empty_ref_latents);
        }
    };

}  // namespace JoyImage

#endif  // __SD_MODEL_DIFFUSION_JOYAI_IMAGE_HPP__
