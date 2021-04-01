// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <iostream>
#include <string>
#include <sstream>
#include <utility>
#include <vector>
#include <fstream>
#include <memory>

#include <inference_engine.hpp>
#include <ie_compound_blob.h>

#include <samples/common.hpp>
#include <samples/classification_results.h>

#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120

#include <CL/cl2.hpp>
//#include <CL/cl_intel_planar_yuv.h>

#include <gpu/gpu_context_api_ocl.hpp>
#include <cldnn/cldnn_config.hpp>

using namespace InferenceEngine;
using namespace InferenceEngine::gpu;

enum format : int32_t {
    bfyx,
    byxf,
};

void checkStatus(int status, const char *message) {
    if (status != 0) {
        std::string str_message(message + std::string(": "));
        std::string str_number(std::to_string(status));

        throw std::runtime_error(str_message + str_number);
    }
}

struct OpenCL {
    cl::Context _context;
    cl::Device _device;
    cl::CommandQueue _queue;

    explicit OpenCL(std::shared_ptr<std::vector<cl_context_properties>> media_api_context_properties = nullptr) {
        // get Intel iGPU OCL device, create context and queue
        {
            const unsigned int refVendorID = 0x8086;
            cl_uint n = 0;
            clGetPlatformIDs(0, NULL, &n);

            // Get platform list
            std::vector<cl_platform_id> platform_ids(n);
            clGetPlatformIDs(n, platform_ids.data(), NULL);

            for (auto& id : platform_ids) {
                cl::Platform platform = cl::Platform(id);
                std::vector<cl::Device> devices;
                platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
                for (auto& d : devices) {
                    if (refVendorID == d.getInfo<CL_DEVICE_VENDOR_ID>()) {
                        _device = d;
                        _context = cl::Context(_device);
                        break;
                    }
                }
            }
            cl_command_queue_properties props = CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;
            _queue = cl::CommandQueue(_context, _device, props);
        }
    }

    explicit OpenCL(cl_context context) {
        // user-supplied context handle
        _context = cl::Context(context);
        _device = cl::Device(_context.getInfo<CL_CONTEXT_DEVICES>()[0]);

        cl_command_queue_properties props = CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;
        _queue = cl::CommandQueue(_context, _device, props);
    }
};

/**
 * \brief Parse image size provided as string in format WIDTHxHEIGHT
 * @return parsed width and height
 */
std::pair<size_t, size_t> parseImageSize(const std::string& size_string) {
    auto delimiter_pos = size_string.find("x");
    if (delimiter_pos == std::string::npos
     || delimiter_pos >= size_string.size() - 1
     || delimiter_pos == 0) {
        std::stringstream err;
        err << "Incorrect format of image size parameter, expected WIDTHxHEIGHT, "
               "actual: " << size_string;
        throw std::runtime_error(err.str());
    }

    size_t width = static_cast<size_t>(
        std::stoull(size_string.substr(0, delimiter_pos)));
    size_t height = static_cast<size_t>(
        std::stoull(size_string.substr(delimiter_pos + 1, size_string.size())));

    if (width == 0 || height == 0) {
        throw std::runtime_error(
            "Incorrect format of image size parameter, width and height must not be equal to 0");
    }

    if (width % 2 != 0 || height % 2 != 0) {
        throw std::runtime_error("Unsupported image size, width and height must be even numbers");
    }

    return {width, height};
}

/**
 * \brief Read image data from file
 * @return buffer containing the image data
 */
std::unique_ptr<unsigned char[]> readImageDataFromFile(const std::string& image_path, size_t size) {
    std::ifstream file(image_path, std::ios_base::ate | std::ios_base::binary);
    if (!file.good() || !file.is_open()) {
        std::stringstream err;
        err << "Cannot access input image file. File path: " << image_path;
        throw std::runtime_error(err.str());
    }

    const size_t file_size = file.tellg();
    if (file_size < size) {
        std::stringstream err;
        err << "Invalid read size provided. File size: " << file_size << ", to read: " << size;
        // throw std::runtime_error(err.str());
    }
    file.seekg(0);

    std::unique_ptr<unsigned char[]> data(new unsigned char[size]);
    file.read(reinterpret_cast<char*>(data.get()), size);
    return data;
}

/**
 * \brief Sets batch size of the network to the specified value
 */
void setBatchSize(CNNNetwork& network, size_t batch) {
    ICNNNetwork::InputShapes inputShapes = network.getInputShapes();
    for (auto& shape : inputShapes) {
        auto& dims = shape.second;
        if (dims.empty()) {
            throw std::runtime_error("Network's input shapes have empty dimensions");
        }
        dims[0] = batch;
    }
    network.reshape(inputShapes);
}

/**
* @brief The entry point of the Inference Engine sample application
*/
int main(int argc, char *argv[]) {
    try {
        // ------------------------------ Parsing and validatiing input arguments------------------------------
        if (argc != 4) {
            std::cout << "Usage : ./hello_nv12_input_classification <path_to_model> <path_to_image> <image_size>"
                      << std::endl;
            return EXIT_FAILURE;
        }

        const std::string input_model{argv[1]};
        const std::string input_image_path{argv[2]};
        size_t input_width = 0, input_height = 0;
        std::tie(input_width, input_height) = parseImageSize(argv[3]);
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 1. Load inference engine ------------------------------------------------
        Core ie;
        // -----------------------------------------------------------------------------------------------------

        // -------------------------- 2. Read the IR generated by the Model Optimizer (.xml and .bin files) ----
        CNNNetwork network = ie.ReadNetwork(input_model);
        setBatchSize(network, 1);
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 3. Configure input and output -------------------------------------------
        // --------------------------- Prepare input blobs -----------------------------------------------------
        if (network.getInputsInfo().empty()) {
            std::cerr << "Network inputs info is empty" << std::endl;
            return EXIT_FAILURE;
        }
        InputInfo::Ptr input_info = network.getInputsInfo().begin()->second;
        std::string input_name = network.getInputsInfo().begin()->first;

        input_info->setLayout(Layout::NCHW);
        input_info->setPrecision(Precision::U8);
        // set input resize algorithm to enable input autoresize
        input_info->getPreProcess().setResizeAlgorithm(ResizeAlgorithm::RESIZE_BILINEAR);
        // set input color format to ColorFormat::NV12 to enable automatic input color format
        // pre-processing
        input_info->getPreProcess().setColorFormat(ColorFormat::NV12);

        // --------------------------- Prepare output blobs ----------------------------------------------------
        if (network.getOutputsInfo().empty()) {
            std::cerr << "Network outputs info is empty" << std::endl;
            return EXIT_FAILURE;
        }
        DataPtr output_info = network.getOutputsInfo().begin()->second;
        std::string output_name = network.getOutputsInfo().begin()->first;

        output_info->setPrecision(Precision::FP32);
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 4. Loading a model to the device ----------------------------------------
        // ie.SetConfig({{ CLDNNConfigParams::KEY_CLDNN_GRAPH_DUMPS_DIR, "optimizer_dump" }}, "GPU");
        std::cout << "Load network..." << std::endl;
        ExecutableNetwork executable_network = ie.LoadNetwork(network,
            "GPU",
            { { CLDNNConfigParams::KEY_CLDNN_NV12_TWO_INPUTS, PluginConfigParams::YES } });
        // -----------------------------------------------------------------------------------------------------
        std::cout << " Done" << std::endl;;

        // --------------------------- 5. Create an infer request ----------------------------------------------
        InferRequest infer_request = executable_network.CreateInferRequest();
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 6. Prepare input --------------------------------------------------------
        // read image with size converted to NV12 data size: height(NV12) = 3 / 2 * logical height
        auto image_buf = readImageDataFromFile(input_image_path, input_width * (input_height * 3 / 2));
        cl::Image2D img_y, img_uv;
        cl_mem nv12_image_plane_y, nv12_image_plane_uv;

        // Get OpenVINO OpenCL context and pack input
        // into OpenCL images
        auto cldnn_context = executable_network.GetContext();
        {
            cl_context ctx = std::dynamic_pointer_cast<ClContext>(cldnn_context)->get();
            auto ocl_instance = std::make_shared<OpenCL>(ctx);
            cl_int err;

            cl_image_format image_format;
            image_format.image_channel_order = CL_R;
            image_format.image_channel_data_type = CL_UNORM_INT8;
            cl_image_desc image_desc = { 0 };
            image_desc.image_type = CL_MEM_OBJECT_IMAGE2D;
            image_desc.image_width = input_width;
            image_desc.image_height = input_height;

            nv12_image_plane_y = clCreateImage(ocl_instance->_context.get(), CL_MEM_READ_WRITE, &image_format, &image_desc, NULL, &err);
            checkStatus(err, "Creating nv12 image plane_y failed");

            image_format.image_channel_order = CL_RG;
            image_desc.image_width = input_width / 2;
            image_desc.image_height = input_height / 2;

            nv12_image_plane_uv = clCreateImage(ocl_instance->_context.get(), CL_MEM_READ_WRITE, &image_format, &image_desc, NULL, &err);
            checkStatus(err, "Creating nv12 image plane_uv failed");

            size_t origin[3] = { 0, 0, 0 };
            size_t y_region[3] = { (size_t)input_width, (size_t)input_height, 1 };
            size_t uv_region[3] = { (size_t)input_width / 2, (size_t)input_height / 2, 1 };

            err = clEnqueueWriteImage(ocl_instance->_queue.get(), nv12_image_plane_y,
                true, origin, y_region, 0, 0, &image_buf[0], 0, NULL, NULL);
            checkStatus(err, "Writing nv12 image plane_y failed");

            err = clEnqueueWriteImage(ocl_instance->_queue.get(), nv12_image_plane_uv,
                true, origin, uv_region, 0, 0, &image_buf[input_width * input_height], 0, NULL, NULL);
            checkStatus(err, "Writing nv12 image plane_uv failed");

            img_y = cl::Image2D(nv12_image_plane_y);
            img_uv = cl::Image2D(nv12_image_plane_uv);
        }

        // --------------------------- Create a blob to hold the NV12 input data -------------------------------
        // Create tensor descriptors for Y and UV blobs
        Blob::Ptr input = make_shared_blob_nv12(cldnn_context, img_y, img_uv);

        // --------------------------- Set the input blob to the InferRequest ----------------------------------
        infer_request.SetBlob(input_name, input);
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 7. Do inference ---------------------------------------------------------
        /* Running the request synchronously */
        infer_request.Infer();
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 8. Process output -------------------------------------------------------
        Blob::Ptr output = infer_request.GetBlob(output_name);

        // Print classification results
        ClassificationResult classificationResult(output, {input_image_path});
        classificationResult.print();
        // -----------------------------------------------------------------------------------------------------
    } catch (const std::exception & ex) {
        std::cerr << ex.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
