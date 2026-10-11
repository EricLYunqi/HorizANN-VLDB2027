#include "config_loader.h"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <yaml-cpp/yaml.h>

namespace ann {

static std::string readFileToString(const std::string& path) {
    std::ifstream ifs(path);
    if (!ifs) throw std::runtime_error("Cannot open config file: " + path);
    std::stringstream buffer; buffer << ifs.rdbuf();
    return buffer.str();
}

ClusterConfig loadClusterConfig(const std::string& path) {
    const YAML::Node root = YAML::Load(readFileToString(path));

    ClusterConfig cfg{};
    // master
    if (root["master"]) {
        const auto& mj = root["master"];
        cfg.master.host = mj["host"] ? mj["host"].as<std::string>() : std::string("0.0.0.0");
        cfg.master.port = mj["port"] ? mj["port"].as<int>() : 50051;
        cfg.master.dimension = mj["dimension"] ? mj["dimension"].as<int>() : 128;
    } else {
        cfg.master.host = "0.0.0.0";
        cfg.master.port = 50051;
        cfg.master.dimension = 128;
    }

    // workers
    if (root["workers"]) {
        for (const auto& wj : root["workers"]) {
            WorkerCfg w{};
            w.id = wj["id"].as<int>();
            w.host = wj["host"].as<std::string>();
            w.port = wj["port"].as<int>();
            w.dimension = wj["dimension"] ? wj["dimension"].as<int>() : 128;
            cfg.workers.push_back(std::move(w));
        }
    }
    return cfg;
}

} // namespace ann


