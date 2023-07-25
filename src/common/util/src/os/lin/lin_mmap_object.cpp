// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <dlfcn.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstring>
#include <iostream>
#include <sstream>

#include "openvino/util/file_util.hpp"
#include "openvino/util/mmap_object.hpp"

namespace ov {

class HandleHolder {
    int m_handle = -1;
    void reset() noexcept {
        if (m_handle != -1) {
            close(m_handle);
            m_handle = -1;
        }
    }

public:
    explicit HandleHolder(int handle = -1) : m_handle(handle) {}

    HandleHolder(const HandleHolder&) = delete;
    HandleHolder& operator=(const HandleHolder&) = delete;

    HandleHolder(HandleHolder&& other) noexcept : m_handle(other.m_handle) {
        other.m_handle = -1;
    }

    HandleHolder& operator=(HandleHolder&& other) noexcept {
        if (this == &other) {
            return *this;
        }
        reset();
        m_handle = other.m_handle;
        other.m_handle = -1;
        return *this;
    }

    ~HandleHolder() {
        reset();
    }

    int get() const noexcept {
        return m_handle;
    }
};

class MapHolder : public MappedMemory {
    void* m_data = MAP_FAILED;
    size_t m_size = 0;
    HandleHolder m_handle;

public:
    MapHolder() = default;

    void set(const std::string& path, const size_t data_size = 0, const size_t offset = 0) {
        int prot = PROT_READ;
        int mode = O_RDONLY;
        struct stat sb = {};
        m_handle = HandleHolder(open(path.c_str(), mode));
        if (m_handle.get() == -1) {
            throw std::runtime_error("Can not open file " + path +
                                     " for mapping. Ensure that file exists and has appropriate permissions");
        }
        if (fstat(m_handle.get(), &sb) == -1) {
            throw std::runtime_error("Can not get file size for " + path);
        }
        if (data_size + offset > static_cast<size_t>(sb.st_size)) {
            throw std::runtime_error("Can not map out of scope memory for " + path);
        }
        m_size = data_size > 0 ? data_size : sb.st_size - offset;
        const size_t page_size = sysconf(_SC_PAGE_SIZE);
        // align offset with page size
        const size_t aligned_offset = (offset / page_size) * page_size;
        const size_t aligned_size = (offset % page_size) + m_size;
        // move pointer to the expected data (after alignment with page size)
        const size_t offset_delta = offset - aligned_offset;
        if (aligned_size > 0) {
            m_data = mmap(nullptr, aligned_size, prot, MAP_PRIVATE, m_handle.get(), aligned_offset);
            if (m_data == MAP_FAILED) {
                throw std::runtime_error("Can not create file mapping for " + path + ", err=" + std::strerror(errno));
            }
            m_data = reinterpret_cast<char*>(m_data) + offset_delta;
        } else {
            m_size = 0;
            m_data = MAP_FAILED;
        }
    }

    ~MapHolder() {
        if (m_data != MAP_FAILED) {
            munmap(m_data, m_size);
        }
    }

    char* data() noexcept override {
        return static_cast<char*>(m_data);
    }

    size_t size() const noexcept override {
        return m_size;
    }
};

std::shared_ptr<ov::MappedMemory> load_mmap_object(const std::string& path,
                                                   const size_t data_size,
                                                   const size_t offset) {
    auto holder = std::make_shared<MapHolder>();
    holder->set(path, data_size, offset);
    return holder;
}

}  // namespace ov
