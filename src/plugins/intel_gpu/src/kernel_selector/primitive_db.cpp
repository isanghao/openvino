// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "primitive_db.h"
#include <assert.h>
#include <algorithm>
#include <vector>
#include <utility>
#include <stdexcept>

#if 1
#include <fstream>
#include <iostream>
#endif

namespace kernel_selector {
namespace gpu {
namespace cache {

primitive_db::primitive_db()
    : primitives({
#include "ks_primitive_db.inc"
      }),
      batch_header_str({
#include "ks_primitive_db_batch_headers.inc"
      }) {
}

std::vector<code> primitive_db::get(const primitive_id& id) const {
#if 1
    {
        std::string pattern("#include \"include/fetch_utils.cl\"");
        std::ifstream kernel_file{id + ".cl", std::ios::in | std::ios::binary};
        if (!kernel_file.is_open())
            kernel_file.open(std::string("/mnt/local/mingyuki/openvino/src/plugins/intel_gpu/src/kernel_selector/cl_kernels/") + id + ".cl", std::ios::in | std::ios::binary);
        if (kernel_file.is_open() && id == "mha_opt") {
            // std::cout << __FILE__ << ":" << __LINE__ << "  " << id << std::endl;
            code ret;
            auto beg = kernel_file.tellg();
            kernel_file.seekg(0, std::ios::end);
            auto end = kernel_file.tellg();
            kernel_file.seekg(0, std::ios::beg);

            ret.resize((size_t)(end - beg));
            kernel_file.read(&ret[0], (size_t)(end - beg));
            ret.replace(ret.find(pattern), pattern.length(), "");
            return {std::move(ret)};
        }
    }
#endif
    try {
        const auto codes = primitives.equal_range(id);
        std::vector<code> temp;
        std::for_each(codes.first, codes.second, [&](const std::pair<const std::string, std::string>& c) {
            temp.push_back(c.second);
        });

        if (temp.size() != 1) {
            throw std::runtime_error("cannot find the kernel " + id + " in primitive database.");
        }

        return temp;
    } catch (...) {
        throw std::runtime_error("cannot find the kernel " + id + " in primitive database.");
    }
}
}  // namespace cache
}  // namespace gpu
}  // namespace kernel_selector
