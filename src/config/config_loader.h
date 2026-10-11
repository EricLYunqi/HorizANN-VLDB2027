#pragma once

#include <string>
#include <vector>

namespace ann {

struct WorkerCfg {
    int id;
    std::string host;
    int port;
    int dimension;
    std::string address() const { return host + ":" + std::to_string(port); }
};

struct MasterCfg {
    std::string host;
    int port;
    int dimension;
    std::string address() const { return host + ":" + std::to_string(port); }
};

struct ClusterConfig {
    MasterCfg master;
    std::vector<WorkerCfg> workers;
};

// Load and parse YAML configuration from file path
ClusterConfig loadClusterConfig(const std::string& path);

} // namespace ann


