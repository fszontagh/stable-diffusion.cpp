#ifndef EXAMPLES_SERVER_SDMODEL_SCANNER_H
#define EXAMPLES_SERVER_SDMODEL_SCANNER_H

#include <filesystem>
#include <string>
#include <unordered_set>
#include <vector>

#include "http_json_responses.h"
class SDModelScanner {
public:
    SDModelScanner(
        std::string sdmodels_path,
        std::string vaes_path,
        std::string embeddings_path,
        std::string loras_path,
        std::string controlnet_path,
        std::string esrgan_path)
        : sdmodels_path(sdmodels_path), vaes_path(vaes_path), loras_path(loras_path), embeddings_path(embeddings_path), controlnet_path(controlnet_path), esrgan_path(esrgan_path) {
        // pre scan folders
        this->scan(sdmodels_path, sdmodels);
        this->scan(vaes_path, vaes);
        this->scan(loras_path, loras);
        this->scan(embeddings_path, embeddings);
        this->scan(controlnet_path, controlnets);
    }

    const std::vector<http_jsonresponse::sd_models_info_t>& getSdModels() const {
        return this->sdmodels;
    }
    const std::vector<http_jsonresponse::sd_models_info_t>& getVaes() const {
        return this->vaes;
    }
    const std::vector<http_jsonresponse::sd_models_info_t>& getLoras() const {
        return this->loras;
    }
    const std::vector<http_jsonresponse::sd_models_info_t>& getEmbeddings() const {
        return this->embeddings;
    }
    const std::vector<http_jsonresponse::sd_models_info_t>& getControlnets() const {
        return this->controlnets;
    }

    const std::vector<http_jsonresponse::sd_models_info_t>& getESRGANs() const {
        return this->esrgan;
    }
    const std::string findModelPath(const std::string& name) {
        const auto info = this->findModel(name);
        if (info.filename.empty() == false) {
            return info.filename;
        }
        return std::string();
    };
    const http_jsonresponse::sd_models_info_t findModel(const std::string& name) {
        for (const auto model : this->sdmodels) {
            if (model.model_name == name) {
                return model;
            }
            if (model.filename == name) {
                return model;
            }
            if (model.title == name) {
                return model;
            }
        }

        for (const auto vae : this->vaes) {
            if (vae.model_name == name) {
                return vae;
            }
            if (vae.filename == name) {
                return vae;
            }
            if (vae.title == name) {
                return vae;
            }
        }

        for (const auto lora : this->loras) {
            if (lora.model_name == name) {
                return lora;
            }
            if (lora.filename == name) {
                return lora;
            }
            if (lora.title == name) {
                return lora;
            }
        }

        for (const auto controlnet : this->controlnets) {
            if (controlnet.model_name == name) {
                return controlnet;
            }
            if (controlnet.filename == name) {
                return controlnet;
            }
            if (controlnet.title == name) {
                return controlnet;
            }
        }

        return http_jsonresponse::sd_models_info_t{};
    }

    void
    reScanSdModels() {
        this->scan(sdmodels_path, sdmodels);
    }

    void reScanVaes() {
        this->scan(vaes_path, vaes);
    }

    void reScanLoras() {
        this->scan(loras_path, loras);
    }

    void reScanEmbeddings() {
        this->scan(embeddings_path, embeddings);
    }

    void reScanControlnet() {
        this->scan(controlnet_path, controlnets);
    }

    void reScaneUpscalers() {
        this->scan(esrgan_path, esrgan);
    }

    void setExtensions(const std::unordered_set<std::string>& extensions) {
        allowedExtensions = extensions;
    }

private:
    std::vector<http_jsonresponse::sd_models_info_t> sdmodels;
    std::vector<http_jsonresponse::sd_models_info_t> vaes;
    std::vector<http_jsonresponse::sd_models_info_t> loras;
    std::vector<http_jsonresponse::sd_models_info_t> embeddings;
    std::vector<http_jsonresponse::sd_models_info_t> controlnets;
    std::vector<http_jsonresponse::sd_models_info_t> esrgan;

    std::unordered_set<std::string> allowedExtensions = {
        ".safetensors",
        ".pt",
        ".ckpt",
        "gguf"};

    std::string sdmodels_path, vaes_path, loras_path, embeddings_path, controlnet_path, esrgan_path;

    /**
     * @brief Scans a directory and its subdirectories for files with allowed extensions, and generates a list of sd_models_info_t.
     *
     * @param directory The directory to scan
     * @param files A reference to a vector to store the results in
     */
    void scan(const std::string& directory, std::vector<http_jsonresponse::sd_models_info_t>& files) {
        files.clear();
        for (const auto& entry : std::filesystem::recursive_directory_iterator(directory)) {
            if (entry.is_regular_file() && isAllowedExtension(entry.path().extension().string())) {
                http_jsonresponse::sd_models_info_t fileInfo = createFileInfo(entry.path());
                files.push_back(fileInfo);
            }
        }
    }

    http_jsonresponse::sd_models_info_t createFileInfo(const std::filesystem::path& filePath) {
        http_jsonresponse::sd_models_info_t info;
        info.filename   = filePath.string();
        info.title      = filePath.parent_path().filename().string() + "/" + filePath.filename().string();
        info.model_name = generateModelName(filePath.filename().string());
        return info;
    }

    // Modelnév generálása (pl. fájlnév kiterjesztés nélkül)
    std::string generateModelName(const std::string& filename) {
        auto pos = filename.find_last_of('.');
        return pos == std::string::npos ? filename : filename.substr(0, pos);
    }
    bool isAllowedExtension(const std::string& extension) const {
        if (allowedExtensions.empty()) {
            return true;
        }
        std::string lowerExt = toLower(extension);
        return allowedExtensions.find(lowerExt) != allowedExtensions.end();
    }
    std::string toLower(const std::string& str) const {
        std::string lowerStr;
        lowerStr.reserve(str.size());
        for (char c : str) {
            lowerStr.push_back(std::tolower(static_cast<unsigned char>(c)));
        }
        return lowerStr;
    }
};

#endif