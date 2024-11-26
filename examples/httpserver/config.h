#ifndef EXAMPLES_SERVER_CONFIG_H
#define EXAMPLES_SERVER_CONFIG_H

#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <ostream>

#include "json.hpp"
#include "sdWebuiConfigStruct.h"

class Config {
public:
    Config(const std::string& filename)
        : config_filename(filename) {
            
        if (std::filesystem::exists(filename) == false) {
            SdWebuiSettings initConfig;
            this->settings = initConfig;
            this->save();
        }
    }

    bool load() {
        std::lock_guard<std::mutex> lock(config_mutex);
        try {
            nlohmann::json tmpj;
            std::ifstream input_file(config_filename);
            if (!input_file.is_open()) {
                std::cerr << "Error opening config file for reading.\n";
                return false;
            }
            input_file >> tmpj;
            auto test = tmpj.get<SdWebuiSettings>();
            this->settings = tmpj;
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Failed to load config: " << e.what() << std::endl;
            return false;
        }
    }

    bool save() {
        std::lock_guard<std::mutex> lock(config_mutex);
        try {
            std::ofstream output_file(config_filename);
            if (!output_file.is_open()) {
                std::cerr << "Error opening config file for writing.\n";
                return false;
            }

            output_file << this->settings.dump(4);  // Pretty print with indentation of 4 spaces
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Failed to save config: " << e.what() << std::endl;
            return false;
        }
    }

    /// Get a setting value.
    ///
    /// @param key The setting name.
    ///
    /// @return The value of the setting. If the setting does not exist, a
    /// default-constructed `T` is returned.
    template <typename T>
    T get(const std::string& key) const {
        if (settings.contains(key)) {
            return settings[key].get<T>();
        }
        return T();
    }

    std::string getString(const std::string& key) const {
        return settings[key].get<std::string>();
    }

    bool getBool(const std::string& key) const {
        return settings[key].get<bool>();
    }

    /// Check if a setting exists.
    ///
    /// @param key The setting name.
    ///
    /// @return true if the setting exists, false otherwise.
    bool exists(const std::string& key) const {
        return settings.contains(key);
    }

    /// Check if a setting exists.
    /// Alias of exists
    ///
    /// @param key The setting name.
    ///
    /// @return true if the setting exists, false otherwise.
    bool contains(const std::string& key) const {
        return this->exists(key);
    }

    /// Set a setting to a value.
    ///
    /// @param key The setting name.
    /// @param value The new value for the setting.
    ///
    /// @note Calling this function does not immediately save the config file.
    /// To save the config file, call save() after calling this function.
    template <typename T>
    void set(const std::string& key, const T& value) {
        settings[key] = value;
    }

    const nlohmann::json& getSettings() const {
        return settings;
    }
    /// Set the settings to a given JSON object.
    ///
    /// @param newSettings The new settings as a JSON object.
    ///
    /// @note Calling this function does not immediately save the config file.
    /// To save the config file, call save() after calling this function.
    void setSettings(const nlohmann::json& newSettings) {
        settings = newSettings;
    }

    /// Get the settings as a `Settings` struct.
    ///
    /// @return The settings as a `Settings` struct. If the settings do not form
    /// a valid `Settings` struct, a default-constructed `Settings` is returned.
    const SdWebuiSettings getSettingsStruct() const {
        return settings.get<SdWebuiSettings>();
    }

    const std::string getConfigFile() const {
        return config_filename;
    }

private:
    std::string config_filename;
    std::mutex config_mutex;
    nlohmann::json settings;
};

#endif