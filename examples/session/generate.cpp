#include "generate.h"
#include "session_params.h"
#include "common/media_io.h"
#include "common/log.h"
#include <chrono>
#include <cstdio>

static int g_out_counter = 0;

static bool save_image(const sd_image_t& img, int idx) {
    char name[64];
    std::snprintf(name, sizeof(name), "session_out_%04d.png", idx);
    return write_image_to_file(name, img.data, img.width, img.height, img.channel);
}

bool run_gen(Session& sess, const std::vector<std::string>& args,
             int& out_index, std::string& err, GenTiming& out_timing) {
    if (!sess.sticky) {
        // explicit mode: reset gen params to defaults before applying this line
        sess.gen = SDGenerationParams{};
    }
    if (!apply_flags(args, sess.cli, sess.ctx, sess.gen, err)) {
        return false;
    }
    if (!sess.ensure_ctx(err)) {
        return false;
    }

    if (sess.gen.sample_params.sample_method == SAMPLE_METHOD_COUNT) {
        sess.gen.sample_params.sample_method = sd_get_default_sample_method(sess.sd_ctx.get());
    }
    if (sess.gen.high_noise_sample_params.sample_method == SAMPLE_METHOD_COUNT) {
        sess.gen.high_noise_sample_params.sample_method = sd_get_default_sample_method(sess.sd_ctx.get());
    }
    if (sess.gen.sample_params.scheduler == SCHEDULER_COUNT) {
        sess.gen.sample_params.scheduler =
            sd_get_default_scheduler(sess.sd_ctx.get(), sess.gen.sample_params.sample_method);
    }

    sd_image_t* out = nullptr;
    int n           = 0;
    bool ok         = false;

    out_timing.before  = sample_mem();
    auto t_start        = std::chrono::steady_clock::now();
    if (sess.cli.mode == VID_GEN) {
        sd_vid_gen_params_t vp = sess.gen.to_sd_vid_gen_params_t();
        sd_audio_t* audio      = nullptr;
        ok = generate_video(sess.sd_ctx.get(), &vp, &out, &n, &audio);
        free_sd_audio(audio);
    } else {
        sd_img_gen_params_t ip = sess.gen.to_sd_img_gen_params_t();
        ok = generate_image(sess.sd_ctx.get(), &ip, &out, &n);
    }
    auto t_end          = std::chrono::steady_clock::now();
    out_timing.after    = sample_mem();
    out_timing.seconds  = std::chrono::duration<double>(t_end - t_start).count();

    if (!ok || out == nullptr || n <= 0) {
        free_sd_images(out, n);
        err = "generate failed";
        return false;
    }
    sess.stats.record(out_timing.seconds, out_timing.before, out_timing.after, get_vram_total_bytes());
    for (int i = 0; i < n; ++i) {
        if (out[i].data) {
            save_image(out[i], g_out_counter);
            out_index = g_out_counter;
            ++g_out_counter;
        }
    }
    free_sd_images(out, n);

    if (sess.sticky) {
        sess.gen.seed += 1;
    }
    return true;
}
