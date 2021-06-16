// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cldnn/runtime/debug_configuration.hpp"
#include <iostream>
#include <memory>

namespace cldnn {

const char *debug_configuration::prefix = "GPU_Debug: ";

// Default policy is that dump_configuration will override other configuration from IE.

debug_configuration::debug_configuration()
        : verbose(0)
        , dump_graphs(std::string()) {
#ifdef GPU_DEBUG_CONFIG
    const std::string OV_GPU_VERBOSE("OV_GPU_Verbose");
    const std::string OV_GPU_DUMP_GRAPHS("OV_GPU_DumpGraphs");
    std::string prefix_config = std::string(prefix) + "Config ";
    if (const auto env_var = std::getenv(OV_GPU_VERBOSE.c_str())) {
        verbose = std::stoi(env_var);
        std::cout << prefix_config << OV_GPU_VERBOSE << "=" << verbose << std::endl;
    }

    if (const auto env_var = std::getenv(OV_GPU_DUMP_GRAPHS.c_str())) {
        dump_graphs = env_var;
        std::cout << prefix_config << OV_GPU_DUMP_GRAPHS << "=" << dump_graphs << std::endl;
    }
#endif
}

const debug_configuration *debug_configuration::get_instance() {
    static std::unique_ptr<debug_configuration> instance(nullptr);
#ifdef GPU_DEBUG_CONFIG
    static std::mutex _m;
    std::lock_guard<std::mutex> lock(_m);
    if (nullptr == instance)
        instance.reset(new debug_configuration());
    return instance.get();
#else
    return nullptr;
#endif
}
} // namespace cldnn
