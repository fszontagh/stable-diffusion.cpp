#ifndef EXAMPLES_SERVER_SDWEBUI_CONFIG_STRUCT_H
#define EXAMPLES_SERVER_SDWEBUI_CONFIG_STRUCT_H
#include <string>
#include <vector>
#include "json.hpp"

struct SdWebuiSettings {
    bool samples_save                                  = true;
    std::string samples_format                         = "png";  // used in image generation
    std::string samples_filename_pattern               = "";
    bool save_images_add_number                        = true;
    std::string save_images_replace_action             = "Replace";
    bool grid_save                                     = true;
    std::string grid_format                            = "png";
    bool grid_extended_filename                        = false;
    bool grid_only_if_multiple                         = true;
    bool grid_prevent_empty_spots                      = false;
    std::string grid_zip_filename_pattern              = "";
    int n_rows                                         = -1;
    std::string font                                   = "";
    std::string grid_text_active_color                 = "#000000";
    std::string grid_text_inactive_color               = "#999999";
    std::string grid_background_color                  = "#ffffff";
    bool save_images_before_face_restoration           = false;
    bool save_images_before_highres_fix                = false;
    bool save_images_before_color_correction           = false;
    bool save_mask                                     = false;
    bool save_mask_composite                           = false;
    int jpeg_quality                                   = 80;
    bool webp_lossless                                 = false;
    bool export_for_4chan                              = true;
    int img_downscale_threshold                        = 4;
    int target_side_length                             = 4000;
    int img_max_size_mp                                = 200;
    bool use_original_name_batch                       = true;
    bool use_upscaler_name_as_suffix                   = false;
    bool save_selected_only                            = true;
    bool save_write_log_csv                            = true;
    bool save_init_img                                 = false;
    std::string temp_dir                               = "";
    bool clean_temp_dir_at_start                       = false;
    bool save_incomplete_images                        = false;
    bool notification_audio                            = true;
    int notification_volume                            = 100;
    std::string outdir_samples                         = "";
    std::string outdir_txt2img_samples                 = "outputs/txt2img-images"; // used to store the images from txt2img
    std::string outdir_img2img_samples                 = "outputs/img2img-images";
    std::string outdir_extras_samples                  = "outputs/extras-images";
    std::string outdir_grids                           = "";
    std::string outdir_txt2img_grids                   = "outputs/txt2img-grids";
    std::string outdir_img2img_grids                   = "outputs/img2img-grids";
    std::string outdir_save                            = "log/images";
    std::string outdir_init_images                     = "outputs/init-images";
    bool save_to_dirs                                  = true;
    bool grid_save_to_dirs                             = true;
    bool use_save_to_dirs_for_ui                       = false;
    std::string directories_filename_pattern           = "[date]";
    int directories_max_prompt_words                   = 8;
    int ESRGAN_tile                                    = 192;
    int ESRGAN_tile_overlap                            = 8;
    std::vector<std::string> realesrgan_enabled_models = {"R-ESRGAN 4x+", "R-ESRGAN 4x+ Anime6B"};
    std::vector<std::string> dat_enabled_models        = {"DAT x2", "DAT x3", "DAT x4"};
    int DAT_tile                                       = 192;
    int DAT_tile_overlap                               = 8;
    std::string upscaler_for_img2img                  = "";
    bool set_scale_by_when_changing_upscaler           = false;
    bool face_restoration                              = false;
    std::string face_restoration_model                 = "CodeFormer";
    double code_former_weight                          = 0.5;
    bool face_restoration_unload                       = false;
    std::string auto_launch_browser                    = "Local";
    bool enable_console_prompts                        = false;
    bool show_warnings                                 = false;
    bool show_gradio_deprecation_warnings              = true;
    int memmon_poll_rate                               = 8;
    bool samples_log_stdout                            = false;
    bool multiple_tqdm                                 = true;
    bool enable_upscale_progressbar                    = true;
    bool print_hypernet_extra                          = false;
    bool list_hidden_files                             = true;
    bool disable_mmap_load_safetensors                 = false;
    bool hide_ldm_prints                               = true;
    bool dump_stacks_on_signal                         = false;
    std::string profiling_explanation                  = "Those settings allow you to enable torch profiler when generating pictures.\nProfiling allows you to see which code uses how much of computer's resources during generation.\nEach generation writes its own profile to one file, overwriting previous.\nThe file can be viewed in <a href=\"chrome:tracing\">Chrome</a>, or on a <a href=\"https://ui.perfetto.dev/\">Perfetto</a> web site.\nWarning: writing profile can take a lot of time, up to 30 seconds, and the file itelf can be around 500MB in size.";
    bool profiling_enable                              = false;
    std::vector<std::string> profiling_activities      = {"CPU"};
    bool profiling_record_shapes                       = true;
    bool profiling_profile_memory                      = true;
    bool profiling_with_stack                          = true;
    std::string profiling_filename                     = "trace.json";
    bool api_enable_requests                           = true;
    bool api_forbid_local_requests                     = true;
    std::string api_useragent                          = "";
    bool unload_models_when_training                   = false;
    bool pin_memory                                    = false;
    bool save_optimizer_state                          = false;
    bool save_training_settings_to_txt                 = true;
    std::string dataset_filename_word_regex            = "";
    std::string dataset_filename_join_string           = " ";
    int training_image_repeats_per_epoch               = 1;
    int training_write_csv_every                       = 500;
    bool training_xattention_optimizations             = false;
    bool training_enable_tensorboard                   = false;
    bool training_tensorboard_save_images              = false;
    int training_tensorboard_flush_every               = 120;
    std::string sd_model_checkpoint                    = ""; // used to store last checkpoint -> here only store the name of the checkpoint
    int sd_checkpoints_limit                           = 1;
    bool sd_checkpoints_keep_in_cpu                    = true;
};
inline void to_json(nlohmann::json& j, const SdWebuiSettings& s) {
    j["samples_save"] = s.samples_save;
    j["samples_format"] = s.samples_format;
    j["samples_filename_pattern"] = s.samples_filename_pattern;
    j["save_images_add_number"] = s.save_images_add_number;
    j["save_images_replace_action"] = s.save_images_replace_action;
    j["grid_save"] = s.grid_save;
    j["grid_format"] = s.grid_format;
    j["grid_extended_filename"] = s.grid_extended_filename;
    j["grid_only_if_multiple"] = s.grid_only_if_multiple;
    j["grid_prevent_empty_spots"] = s.grid_prevent_empty_spots;
    j["grid_zip_filename_pattern"] = s.grid_zip_filename_pattern;
    j["n_rows"] = s.n_rows;
    j["font"] = s.font;
    j["grid_text_active_color"] = s.grid_text_active_color;
    j["grid_text_inactive_color"] = s.grid_text_inactive_color;
    j["grid_background_color"] = s.grid_background_color;
    j["save_images_before_face_restoration"] = s.save_images_before_face_restoration;
    j["save_images_before_highres_fix"] = s.save_images_before_highres_fix;
    j["save_images_before_color_correction"] = s.save_images_before_color_correction;
    j["save_mask"] = s.save_mask;
    j["save_mask_composite"] = s.save_mask_composite;
    j["jpeg_quality"] = s.jpeg_quality;
    j["webp_lossless"] = s.webp_lossless;
    j["export_for_4chan"] = s.export_for_4chan;
    j["img_downscale_threshold"] = s.img_downscale_threshold;
    j["target_side_length"] = s.target_side_length;
    j["img_max_size_mp"] = s.img_max_size_mp;
    j["use_original_name_batch"] = s.use_original_name_batch;
    j["use_upscaler_name_as_suffix"] = s.use_upscaler_name_as_suffix;
    j["save_selected_only"] = s.save_selected_only;
    j["save_write_log_csv"] = s.save_write_log_csv;
    j["save_init_img"] = s.save_init_img;
    j["temp_dir"] = s.temp_dir;
    j["clean_temp_dir_at_start"] = s.clean_temp_dir_at_start;
    j["save_incomplete_images"] = s.save_incomplete_images;
    j["notification_audio"] = s.notification_audio;
    j["notification_volume"] = s.notification_volume;
    j["outdir_samples"] = s.outdir_samples;
    j["outdir_txt2img_samples"] = s.outdir_txt2img_samples;
    j["outdir_img2img_samples"] = s.outdir_img2img_samples;
    j["outdir_extras_samples"] = s.outdir_extras_samples;
    j["outdir_grids"] = s.outdir_grids;
    j["outdir_txt2img_grids"] = s.outdir_txt2img_grids;
    j["outdir_img2img_grids"] = s.outdir_img2img_grids;
    j["outdir_save"] = s.outdir_save;
    j["outdir_init_images"] = s.outdir_init_images;
    j["save_to_dirs"] = s.save_to_dirs;
    j["grid_save_to_dirs"] = s.grid_save_to_dirs;
    j["use_save_to_dirs_for_ui"] = s.use_save_to_dirs_for_ui;
    j["directories_filename_pattern"] = s.directories_filename_pattern;
    j["directories_max_prompt_words"] = s.directories_max_prompt_words;
    j["ESRGAN_tile"] = s.ESRGAN_tile;
    j["ESRGAN_tile_overlap"] = s.ESRGAN_tile_overlap;
    j["realesrgan_enabled_models"] = s.realesrgan_enabled_models;
    j["dat_enabled_models"] = s.dat_enabled_models;
    j["DAT_tile"] = s.DAT_tile;
    j["DAT_tile_overlap"] = s.DAT_tile_overlap;
    j["upscaler_for_img2img"] = s.upscaler_for_img2img;
    j["set_scale_by_when_changing_upscaler"] = s.set_scale_by_when_changing_upscaler;
    j["face_restoration"] = s.face_restoration;
    j["face_restoration_model"] = s.face_restoration_model;
    j["code_former_weight"] = s.code_former_weight;
    j["face_restoration_unload"] = s.face_restoration_unload;
    j["auto_launch_browser"] = s.auto_launch_browser;
    j["enable_console_prompts"] = s.enable_console_prompts;
    j["show_warnings"] = s.show_warnings;
    j["show_gradio_deprecation_warnings"] = s.show_gradio_deprecation_warnings;
    j["memmon_poll_rate"] = s.memmon_poll_rate;
    j["samples_log_stdout"] = s.samples_log_stdout;
    j["multiple_tqdm"] = s.multiple_tqdm;
    j["enable_upscale_progressbar"] = s.enable_upscale_progressbar;
    j["print_hypernet_extra"] = s.print_hypernet_extra;
    j["list_hidden_files"] = s.list_hidden_files;
    j["disable_mmap_load_safetensors"] = s.disable_mmap_load_safetensors;
    j["hide_ldm_prints"] = s.hide_ldm_prints;
    j["dump_stacks_on_signal"] = s.dump_stacks_on_signal;
    j["profiling_explanation"] = s.profiling_explanation;
    j["profiling_enable"] = s.profiling_enable;
    j["profiling_activities"] = s.profiling_activities;
    j["profiling_record_shapes"] = s.profiling_record_shapes;
    j["profiling_profile_memory"] = s.profiling_profile_memory;
    j["profiling_with_stack"] = s.profiling_with_stack;
    j["profiling_filename"] = s.profiling_filename;
    j["api_enable_requests"] = s.api_enable_requests;
    j["api_forbid_local_requests"] = s.api_forbid_local_requests;
    j["api_useragent"] = s.api_useragent;
    j["unload_models_when_training"] = s.unload_models_when_training;
    j["pin_memory"] = s.pin_memory;
    j["save_optimizer_state"] = s.save_optimizer_state;
    j["save_training_settings_to_txt"] = s.save_training_settings_to_txt;
    j["dataset_filename_word_regex"] = s.dataset_filename_word_regex;
    j["dataset_filename_join_string"] = s.dataset_filename_join_string;
    j["training_image_repeats_per_epoch"] = s.training_image_repeats_per_epoch;
    j["training_write_csv_every"] = s.training_write_csv_every;
    j["training_xattention_optimizations"] = s.training_xattention_optimizations;
    j["training_enable_tensorboard"] = s.training_enable_tensorboard;
    j["training_tensorboard_save_images"] = s.training_tensorboard_save_images;
    j["training_tensorboard_flush_every"] = s.training_tensorboard_flush_every;
    j["sd_model_checkpoint"] = s.sd_model_checkpoint;
    j["sd_checkpoints_limit"] = s.sd_checkpoints_limit;
    j["sd_checkpoints_keep_in_cpu"] = s.sd_checkpoints_keep_in_cpu;
};


inline void from_json(const nlohmann::json& j, SdWebuiSettings& s) {
    j.at("samples_save").get_to(s.samples_save);
    j.at("samples_format").get_to(s.samples_format);
    j.at("samples_filename_pattern").get_to(s.samples_filename_pattern);
    j.at("save_images_add_number").get_to(s.save_images_add_number);
    j.at("save_images_replace_action").get_to(s.save_images_replace_action);
    j.at("grid_save").get_to(s.grid_save);
    j.at("grid_format").get_to(s.grid_format);
    j.at("grid_extended_filename").get_to(s.grid_extended_filename);
    j.at("grid_only_if_multiple").get_to(s.grid_only_if_multiple);
    j.at("grid_prevent_empty_spots").get_to(s.grid_prevent_empty_spots);
    j.at("grid_zip_filename_pattern").get_to(s.grid_zip_filename_pattern);
    j.at("n_rows").get_to(s.n_rows);
    j.at("font").get_to(s.font);
    j.at("grid_text_active_color").get_to(s.grid_text_active_color);
    j.at("grid_text_inactive_color").get_to(s.grid_text_inactive_color);
    j.at("grid_background_color").get_to(s.grid_background_color);
    j.at("save_images_before_face_restoration").get_to(s.save_images_before_face_restoration);
    j.at("save_images_before_highres_fix").get_to(s.save_images_before_highres_fix);
    j.at("save_images_before_color_correction").get_to(s.save_images_before_color_correction);
    j.at("save_mask").get_to(s.save_mask);
    j.at("save_mask_composite").get_to(s.save_mask_composite);
    j.at("jpeg_quality").get_to(s.jpeg_quality);
    j.at("webp_lossless").get_to(s.webp_lossless);
    j.at("export_for_4chan").get_to(s.export_for_4chan);
    j.at("img_downscale_threshold").get_to(s.img_downscale_threshold);
    j.at("target_side_length").get_to(s.target_side_length);
    j.at("img_max_size_mp").get_to(s.img_max_size_mp);
    j.at("use_original_name_batch").get_to(s.use_original_name_batch);
    j.at("use_upscaler_name_as_suffix").get_to(s.use_upscaler_name_as_suffix);
    j.at("save_selected_only").get_to(s.save_selected_only);
    j.at("save_write_log_csv").get_to(s.save_write_log_csv);
    j.at("save_init_img").get_to(s.save_init_img);
    j.at("temp_dir").get_to(s.temp_dir);
    j.at("clean_temp_dir_at_start").get_to(s.clean_temp_dir_at_start);
    j.at("save_incomplete_images").get_to(s.save_incomplete_images);
    j.at("notification_audio").get_to(s.notification_audio);
    j.at("notification_volume").get_to(s.notification_volume);
    j.at("outdir_samples").get_to(s.outdir_samples);
    j.at("outdir_txt2img_samples").get_to(s.outdir_txt2img_samples);
    j.at("outdir_img2img_samples").get_to(s.outdir_img2img_samples);
    j.at("outdir_extras_samples").get_to(s.outdir_extras_samples);
    j.at("outdir_grids").get_to(s.outdir_grids);
    j.at("outdir_txt2img_grids").get_to(s.outdir_txt2img_grids);
    j.at("outdir_img2img_grids").get_to(s.outdir_img2img_grids);
    j.at("outdir_save").get_to(s.outdir_save);
    j.at("outdir_init_images").get_to(s.outdir_init_images);
    j.at("save_to_dirs").get_to(s.save_to_dirs);
    j.at("grid_save_to_dirs").get_to(s.grid_save_to_dirs);
    j.at("use_save_to_dirs_for_ui").get_to(s.use_save_to_dirs_for_ui);
    j.at("directories_filename_pattern").get_to(s.directories_filename_pattern);
    j.at("directories_max_prompt_words").get_to(s.directories_max_prompt_words);
    j.at("ESRGAN_tile").get_to(s.ESRGAN_tile);
    j.at("ESRGAN_tile_overlap").get_to(s.ESRGAN_tile_overlap);
    j.at("realesrgan_enabled_models").get_to(s.realesrgan_enabled_models);
    j.at("dat_enabled_models").get_to(s.dat_enabled_models);
    j.at("DAT_tile").get_to(s.DAT_tile);
    j.at("DAT_tile_overlap").get_to(s.DAT_tile_overlap);
    j.at("upscaler_for_img2img").get_to(s.upscaler_for_img2img);
    j.at("set_scale_by_when_changing_upscaler").get_to(s.set_scale_by_when_changing_upscaler);
    j.at("face_restoration").get_to(s.face_restoration);
    j.at("face_restoration_model").get_to(s.face_restoration_model);
    j.at("code_former_weight").get_to(s.code_former_weight);
    j.at("face_restoration_unload").get_to(s.face_restoration_unload);
    j.at("auto_launch_browser").get_to(s.auto_launch_browser);
    j.at("enable_console_prompts").get_to(s.enable_console_prompts);
    j.at("show_warnings").get_to(s.show_warnings);
    j.at("show_gradio_deprecation_warnings").get_to(s.show_gradio_deprecation_warnings);
    j.at("memmon_poll_rate").get_to(s.memmon_poll_rate);
    j.at("samples_log_stdout").get_to(s.samples_log_stdout);
    j.at("multiple_tqdm").get_to(s.multiple_tqdm);
    j.at("enable_upscale_progressbar").get_to(s.enable_upscale_progressbar);
    j.at("print_hypernet_extra").get_to(s.print_hypernet_extra);
    j.at("list_hidden_files").get_to(s.list_hidden_files);
    j.at("disable_mmap_load_safetensors").get_to(s.disable_mmap_load_safetensors);
    j.at("hide_ldm_prints").get_to(s.hide_ldm_prints);
    j.at("dump_stacks_on_signal").get_to(s.dump_stacks_on_signal);
    j.at("profiling_explanation").get_to(s.profiling_explanation);
    j.at("profiling_enable").get_to(s.profiling_enable);
    j.at("profiling_activities").get_to(s.profiling_activities);
    j.at("profiling_record_shapes").get_to(s.profiling_record_shapes);
    j.at("profiling_profile_memory").get_to(s.profiling_profile_memory);
    j.at("profiling_with_stack").get_to(s.profiling_with_stack);
    j.at("profiling_filename").get_to(s.profiling_filename);
    j.at("api_enable_requests").get_to(s.api_enable_requests);
    j.at("api_forbid_local_requests").get_to(s.api_forbid_local_requests);
    j.at("api_useragent").get_to(s.api_useragent);
    j.at("unload_models_when_training").get_to(s.unload_models_when_training);
    j.at("pin_memory").get_to(s.pin_memory);
    j.at("save_optimizer_state").get_to(s.save_optimizer_state);
    j.at("save_training_settings_to_txt").get_to(s.save_training_settings_to_txt);
    j.at("dataset_filename_word_regex").get_to(s.dataset_filename_word_regex);
    j.at("dataset_filename_join_string").get_to(s.dataset_filename_join_string);
    j.at("training_image_repeats_per_epoch").get_to(s.training_image_repeats_per_epoch);
    j.at("training_write_csv_every").get_to(s.training_write_csv_every);
    j.at("training_xattention_optimizations").get_to(s.training_xattention_optimizations);
    j.at("training_enable_tensorboard").get_to(s.training_enable_tensorboard);
    j.at("training_tensorboard_save_images").get_to(s.training_tensorboard_save_images);
    j.at("training_tensorboard_flush_every").get_to(s.training_tensorboard_flush_every);
    j.at("sd_model_checkpoint").get_to(s.sd_model_checkpoint);
    j.at("sd_checkpoints_limit").get_to(s.sd_checkpoints_limit);
    j.at("sd_checkpoints_keep_in_cpu").get_to(s.sd_checkpoints_keep_in_cpu);
}

#endif