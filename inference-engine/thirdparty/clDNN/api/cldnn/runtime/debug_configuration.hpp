// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <string>
#include <mutex>

#ifdef GPU_DEBUG_CONFIG
#define GPU_DEBUG_IF(cond) if (cond)
#else
#define GPU_DEBUG_IF(cond) if (0)
#endif

namespace cldnn {

class debug_configuration {
private:
    debug_configuration();
public:
    static const char *prefix;
    int verbose;
    std::string dump_graphs;
    static const debug_configuration *get_instance();
};

}  // namespace cldnn
