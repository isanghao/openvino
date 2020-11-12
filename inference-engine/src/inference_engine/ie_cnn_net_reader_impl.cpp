﻿// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <file_utils.h>
#include <ie_cnn_net_reader_impl.h>

#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "cnn_network_ngraph_impl.hpp"
#include "details/os/os_filesystem.hpp"
#include "ie_format_parser.h"
#include "ie_ir_reader.hpp"
#include "ie_profiling.hpp"
#include "parsers.h"
#include "blob_factory.hpp"
#include "debug.h"
#include "xml_parse_utils.h"

using namespace std;
using namespace InferenceEngine;
using namespace InferenceEngine::details;
using namespace XMLParseUtils;

IE_SUPPRESS_DEPRECATED_START
CNNNetReaderImpl::CNNNetReaderImpl(const FormatParserCreator::Ptr& _creator)
    : parseSuccess(false), _version(0), parserCreator(_creator) {}

#if defined(ENABLE_IR_READER)
static void parsePreProcess(std::shared_ptr<ICNNNetwork>& network, const pugi::xml_node& root, const TBlob<uint8_t>::Ptr& weights) {
    /*
        <pre-process mean-precision="FP32">
        <channel id = ”0”>
        <mean offset = "121930449" size = "51529" / >  // in case of array – ref to the .bin file
        </channel>
        </pre-process>
    */

    auto ppNode = root.child("pre-process");
    if (ppNode.empty()) {
        return;
    }
    // find out to what input this belongs to
    std::string inputName;
    InputInfo::Ptr preProcessInput;

    inputName = GetStrAttr(ppNode, "reference-layer-name", "");
    inputName = details::trim(inputName);
    InputsDataMap inputs;
    network->getInputsInfo(inputs);
    if (inputName.empty()) {
        // fallback (old format), look for the picture in the inputs
        if (inputs.empty()) THROW_IE_EXCEPTION << "network has no input";

        for (auto i : inputs) {
            if (i.second->getTensorDesc().getDims().size() == 4) {
                preProcessInput = i.second;
                break;
            }
        }
        if (!preProcessInput) {
            preProcessInput = inputs.begin()->second;
        }

        inputName = preProcessInput->name();
    } else {
        preProcessInput = inputs.at(inputName);
        if (!preProcessInput)
            THROW_IE_EXCEPTION << "pre-process name ref '" << inputName << "' refers to un-existing input";
    }

    // dims vector without batch size
    SizeVector inputDims = preProcessInput->getTensorDesc().getDims();
    size_t noOfChannels = 0, width = 0, height = 0;

    if (inputDims.size() < 2) {
        THROW_IE_EXCEPTION << "network did not define input dimensions properly";
    } else if (inputDims.size() == 2) {  // NC
        noOfChannels = inputDims[1];
        width = inputDims[1];
        height = inputDims[0];
    } else if (inputDims.size() == 3) {
        width = inputDims[2];
        height = inputDims[1];
        noOfChannels = inputDims[0];
    } else if (inputDims.size() == 4) {
        width = inputDims[3];
        height = inputDims[2];
        noOfChannels = inputDims[1];
    } else if (inputDims.size() == 5) {
        width = inputDims[4];
        height = inputDims[3];
        noOfChannels = inputDims[2];
    }

    PreProcessInfo& pp = preProcessInput->getPreProcess();
    pp.init(noOfChannels);

    auto meanSegmentPrecision = GetPrecisionAttr(ppNode, "mean-precision", Precision::UNSPECIFIED);
    if (!meanSegmentPrecision || meanSegmentPrecision == Precision::MIXED)
        THROW_IE_EXCEPTION << "mean blob defined without specifying precision.";

    ResponseDesc resp;
    InferenceEngine::PreProcessChannel::Ptr preProcessChannel;

    int lastChanNo = -1;
    std::unordered_set<int> idsForMeanImage;

    FOREACH_CHILD(chan, ppNode, "channel") {
        int chanNo = GetIntAttr(chan, "id", lastChanNo + 1);
        if (chanNo >= static_cast<int>(noOfChannels) || chanNo < 0) {
            THROW_IE_EXCEPTION << "Pre-process channel id invalid: " << chanNo;
        }
        lastChanNo = chanNo;
        preProcessChannel = pp[chanNo];

        auto meanNode = chan.child("mean");
        if (!meanNode.empty()) {
            if (!meanNode.attribute("size")) {
                THROW_IE_EXCEPTION << "mean should have the attribute: size";
            }
            if (meanNode.attribute("size")) {
                idsForMeanImage.insert(chanNo);
                size_t size = static_cast<size_t>(GetIntAttr(meanNode, "size"));
                size_t offset = static_cast<size_t>(GetIntAttr(meanNode, "offset"));
                if (width * height * meanSegmentPrecision.size() != size) {
                    THROW_IE_EXCEPTION << "mean blob size mismatch expected input, got: " << size
                                       << " extpecting " << width << " x " << height << " x "
                                       << meanSegmentPrecision.size();
                }
                preProcessChannel->meanData = make_blob_with_precision(TensorDesc(meanSegmentPrecision, {height, width}, Layout::HW));
                preProcessChannel->meanData->allocate();
                auto lockedMem = preProcessChannel->meanData->buffer();
                char* data = lockedMem.as<char *>();
                auto weightsLocked = weights->cbuffer();
                const char* origData = weightsLocked.as<const char*>();
                memcpy(data, origData + offset, size);
            }
        }
    }

    if (idsForMeanImage.size() == noOfChannels) {
        pp.setVariant(MEAN_IMAGE);
    } else if (idsForMeanImage.size() == 0) {
        pp.setVariant(NONE);
    } else {
        std::string validMeanImageIds = "";
        for (auto id : idsForMeanImage) {
            validMeanImageIds += std::to_string(id) + " ";
        }
        THROW_IE_EXCEPTION << "mean is not provided for all channels\n"
                              "Provided mean image for: "
                           << validMeanImageIds;
    }
 }
#endif

StatusCode CNNNetReaderImpl::SetWeights(const TBlob<uint8_t>::Ptr& weights, ResponseDesc* desc) noexcept {
    if (!_parser && _version < 10) {
        return DescriptionBuffer(desc) << "network must be read first";
    }
    try {
        if (_version == 10) {
#if defined(ENABLE_IR_READER)
            // It's time to perform actual reading of V10 network and instantiate CNNNetworkNGraphImpl
            IRReader v10Reader(extensions);
            std::stringstream model;
            xmlDoc->save(model);
            network = std::make_shared<CNNNetworkNGraphImpl>(v10Reader.read(model.str(), weights));
            pugi::xml_node root = xmlDoc->document_element();

            parsePreProcess(network, root, weights);
#else
            return DescriptionBuffer(desc) << "Please, recompile Inference Engine with the ENABLE_IR_READER=ON Cmake option";
#endif
        } else {
            _parser->SetWeights(weights);
        }
    } catch (const InferenceEngineException& iee) {
        xmlDoc.reset();
        return DescriptionBuffer(desc) << iee.what();
    }

    xmlDoc.reset();
    return OK;
}

size_t CNNNetReaderImpl::GetFileVersion(pugi::xml_node& root) {
    return XMLParseUtils::GetUIntAttr(root, "version", 0);
}

StatusCode CNNNetReaderImpl::ReadNetwork(const void* model, size_t size, ResponseDesc* resp) noexcept {
    if (network) {
        return DescriptionBuffer(NETWORK_NOT_READ, resp)
               << "Network has been read already, use new reader instance to read new network.";
    }

    xmlDoc = std::make_shared<pugi::xml_document>();
    pugi::xml_parse_result res = xmlDoc->load_buffer(model, size);
    if (res.status != pugi::status_ok) {
        return DescriptionBuffer(resp) << res.description() << "at offset " << res.offset;
    }
    StatusCode ret = ReadNetwork();
    if (ret != OK) {
        return DescriptionBuffer(resp) << "Error reading network: " << description;
    }
    return OK;
}

StatusCode CNNNetReaderImpl::ReadWeights(const char* filepath, ResponseDesc* resp) noexcept {
    IE_PROFILING_AUTO_SCOPE(CNNNetReaderImpl::ReadWeights)
    int64_t fileSize = FileUtils::fileSize(filepath);

    if (fileSize < 0)
        return DescriptionBuffer(resp) << "filesize for: " << filepath << " - " << fileSize
                                       << "<0. Please, check weights file existence.";

    // If IR V10 then there hasn't been loaded network yet
    if (network.get() == nullptr && _version < 10) {
        return DescriptionBuffer(resp) << "network is empty";
    }

    auto ulFileSize = static_cast<size_t>(fileSize);

    try {
        TBlob<uint8_t>::Ptr weightsPtr(new TBlob<uint8_t>(TensorDesc(Precision::U8, {ulFileSize}, Layout::C)));
        weightsPtr->allocate();
        FileUtils::readAllFile(filepath, weightsPtr->buffer(), ulFileSize);
        return SetWeights(weightsPtr, resp);
    } catch (const InferenceEngineException& ex) {
        return DescriptionBuffer(resp) << ex.what();
    }
}

StatusCode CNNNetReaderImpl::ReadNetwork(const char* filepath, ResponseDesc* resp) noexcept {
    IE_PROFILING_AUTO_SCOPE(CNNNetReaderImpl::ReadNetwork)
    if (network) {
        return DescriptionBuffer(NETWORK_NOT_READ, resp)
               << "Network has been read already, use new reader instance to read new network.";
    }

    auto parse_result = ParseXml(filepath);
    if (!parse_result.error_msg.empty()) {
        return DescriptionBuffer(resp) << parse_result.error_msg;
    }
    xmlDoc = std::move(parse_result.xml);

    StatusCode ret = ReadNetwork();
    if (ret != OK) {
        return DescriptionBuffer(resp) << "Error reading network: " << description;
    }
    return OK;
}

StatusCode CNNNetReaderImpl::ReadNetwork() {
    description.clear();

    try {
        // check which version it is...
        pugi::xml_node root = xmlDoc->document_element();

        _version = GetFileVersion(root);
        if (_version < 2) THROW_IE_EXCEPTION << "deprecated IR version: " << _version;
        if (_version == 10) {
            // Activate an alternative code path for V10 that should be read into ngraph::Function
            // We cannot proceed with reading right now, because there is not binary file loaded.
            // So we are postponing real read until weights are specified.
            parseSuccess = true;
        } else if (_version < 10) {
            _parser = parserCreator->create(_version);
            InferenceEngine::details::CNNNetworkImplPtr local_network = _parser->Parse(root);
            name = local_network->getName();
            local_network->validate(_version);
            network = local_network;
            parseSuccess = true;
        } else {
            THROW_IE_EXCEPTION << "cannot parse future versions: " << _version;
        }
    } catch (const std::string& err) {
        description = err;
        parseSuccess = false;
        return GENERAL_ERROR;
    } catch (const InferenceEngineException& e) {
        description = e.what();
        parseSuccess = false;
        return GENERAL_ERROR;
    } catch (const std::exception& e) {
        description = e.what();
        parseSuccess = false;
        return GENERAL_ERROR;
    } catch (...) {
        description = "Unknown exception thrown";
        parseSuccess = false;
        return UNEXPECTED;
    }

    return OK;
}

void CNNNetReaderImpl::addExtensions(const std::vector<InferenceEngine::IExtensionPtr>& ext) {
    extensions = ext;
}

std::shared_ptr<IFormatParser> V2FormatParserCreator::create(size_t version) {
#ifdef ENABLE_IR_READER
    return std::make_shared<FormatParser>(version);
#else
    THROW_IE_EXCEPTION << "Please, recompile Inference Engine library with the ENABLE_IR_READER=ON Cmake option";
    return nullptr;
#endif
}

InferenceEngine::ICNNNetReader* InferenceEngine::CreateCNNNetReader() noexcept {
    return new CNNNetReaderImpl(std::make_shared<V2FormatParserCreator>());
}
IE_SUPPRESS_DEPRECATED_END
