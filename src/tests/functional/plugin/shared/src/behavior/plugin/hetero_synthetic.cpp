// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/plugin/hetero_synthetic.hpp"
#include <ngraph/op/util/op_types.hpp>
#include "ov_models/builders.hpp"
#include "ov_models/subgraph_builders.hpp"
#include "common_test_utils/file_utils.hpp"
#include "openvino/util/file_util.hpp"
#include <random>
#include "ie_algorithm.hpp"

namespace HeteroTests {

static std::vector<std::function<std::shared_ptr<ngraph::Function>()>> builders = {
    [] {return ngraph::builder::subgraph::makeSplitMultiConvConcat();},
    [] {return ngraph::builder::subgraph::makeNestedSplitConvConcat();},
    [] {return ngraph::builder::subgraph::makeSplitConvConcatNestedInBranch();},
    [] {return ngraph::builder::subgraph::makeSplitConvConcatNestedInBranchNestedOut();},
    [] {return ngraph::builder::subgraph::makeNestedBranchConvConcat();},
};

std::vector<FunctionParameter> HeteroSyntheticTest::withMajorNodesFunctions(
    const std::function<std::shared_ptr<ngraph::Function>()>& builder,
    const std::unordered_set<std::string>& majorNodes,
    bool dynamic_batch) {
    auto function = builder();
    std::vector<FunctionParameter> result;
    result.push_back(FunctionParameter{majorNodes, function, dynamic_batch, 0});
    return result;
}

std::vector<FunctionParameter> HeteroSyntheticTest::singleMajorNodeFunctions(
    const std::vector<std::function<std::shared_ptr<ngraph::Function>()>>& builders,
    bool dynamic_batch) {
    std::vector<FunctionParameter> result;
    for (auto&& builder : builders) {
        auto function = builder();
        for (auto&& node : function->get_ordered_ops()) {
            if (!ngraph::op::is_constant(node) &&
                    !(ngraph::op::is_parameter(node)) &&
                    !(ngraph::op::is_output(node))) {
                result.push_back(FunctionParameter{{node->get_friendly_name()}, function, dynamic_batch, 0});
            }
        }
    }
    return result;
}

std::vector<FunctionParameter> HeteroSyntheticTest::randomMajorNodeFunctions(
    const std::vector<std::function<std::shared_ptr<ngraph::Function>()>>& builders,
    bool dynamic_batch,
    uint32_t seed) {
    std::vector<FunctionParameter> results;
    for (auto p = 0.2; p < 1.; p+=0.2) {
        while (seed == 0) {
            seed = std::random_device {}();
        }
        std::mt19937 e{seed};
        std::bernoulli_distribution d{p};
        for (auto&& builder : builders) {
            auto function = builder();
            auto ordered_ops = function->get_ordered_ops();
            for (std::size_t i = 0; i < ordered_ops.size(); ++i) {
                std::unordered_set<std::string> majorPluginNodeIds;
                for (auto&& node : ordered_ops) {
                    if (!(ngraph::op::is_constant(node)) &&
                            !(ngraph::op::is_parameter(node)) &&
                            !(ngraph::op::is_output(node)) && d(e)) {
                        majorPluginNodeIds.emplace(node->get_friendly_name());
                    }
                }
                if (std::any_of(std::begin(results), std::end(results), [&] (const FunctionParameter& param) {
                    return majorPluginNodeIds == param._majorPluginNodeIds;
                })) {
                    continue;
                }
                results.push_back(FunctionParameter{majorPluginNodeIds, function, dynamic_batch, seed});
            }
        }
    }
    return results;
}

std::vector<FunctionParameter> HeteroSyntheticTest::_singleMajorNodeFunctions
    = HeteroSyntheticTest::singleMajorNodeFunctions(builders);

std::vector<FunctionParameter> HeteroSyntheticTest::_randomMajorNodeFunctions
    = HeteroSyntheticTest::randomMajorNodeFunctions(builders);

std::string HeteroSyntheticTest::getTestCaseName(const ::testing::TestParamInfo<HeteroSyntheticTestParameters>& obj) {
    std::vector<PluginParameter> pluginParameters;
    FunctionParameter functionParamter;
    std::tie(pluginParameters, functionParamter) = obj.param;
    std::string name = "function=" + functionParamter._function->get_friendly_name();
    name += "_layers=";
    std::size_t num = functionParamter._majorPluginNodeIds.size() - 1;
    for (auto&& id : functionParamter._majorPluginNodeIds) {
        name += id + ((num !=0) ? "," : "");
        num--;
    }
    name += "_targetDevice=HETERO:";
    num = pluginParameters.size() - 1;
    for (auto&& pluginParameter : pluginParameters) {
        name += pluginParameter._name + ((num !=0) ? "," : "");
        num--;
    }
    return name;
}

void HeteroSyntheticTest::SetUp() {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    auto& param = GetParam();
    targetDevice = "HETERO:";
    int num = std::get<Plugin>(param).size() - 1;
    for (auto&& pluginParameter : std::get<Plugin>(param)) {
        bool registred = true;
        try {
            if (pluginParameter._location == "openvino_template_plugin") {
                PluginCache::get().ie()->RegisterPlugin(ov::util::make_plugin_library_name(
                    ov::test::utils::getExecutableDirectory(), pluginParameter._location + OV_BUILD_POSTFIX),
                    pluginParameter._name);
            } else {
                PluginCache::get().ie()->RegisterPlugin(pluginParameter._location
                    + OV_BUILD_POSTFIX, pluginParameter._name);
            }
        } catch (InferenceEngine::Exception& ex) {
            if (std::string{ex.what()}.find("Device with \"" + pluginParameter._name
                                             + "\"  is already registered in the OpenVINO Runtime")
                == std::string::npos) {
                throw ex;
            } else {
                registred = false;
            }
        }
        if (registred) {
            _registredPlugins.push_back(pluginParameter._name);
        }
        targetDevice += pluginParameter._name;
        targetDevice += ((num !=0) ? "," : "");
        --num;
    }
    function = std::get<Function>(param)._function;
    if (std::get<Function>(param)._dynamic_batch) {
        for (auto&& input : function->inputs()) {
            auto shape = input.get_partial_shape();
            shape[0] = ov::Dimension(1, 16);
        }
    }
}

void HeteroSyntheticTest::TearDown() {
    if (!ov::test::utils::current_test_is_disabled()) {
        for (auto&& pluginName : _registredPlugins) {
            PluginCache::get().ie()->UnregisterPlugin(pluginName);
        }
    }
}

std::string HeteroSyntheticTest::SetUpAffinity() {
    auto& param = GetParam();
    std::string affinities;
    auto& pluginParameters = std::get<Plugin>(param);
    affinities += "\n{\n";
    for (auto&& node : std::get<Function>(param)._function->get_ordered_ops()) {
        std::string affinity;
        auto get_affinity = [&](const std::string& name) {
            if (std::get<Function>(param)._majorPluginNodeIds.end() !=
                std::get<Function>(param)._majorPluginNodeIds.find(name)) {
                return pluginParameters.at(0)._name;
            } else {
                return pluginParameters.at(1)._name;
            }
        };
        if (ngraph::op::is_constant(node) || ngraph::op::is_output(node) || ngraph::op::is_parameter(node)) {
            auto& node_with_affinity_name =
                ngraph::op::is_output(node)
                    ? node->input_value(0).get_node()->get_friendly_name()
                    : node->output(0).get_target_inputs().begin()->get_node()->get_friendly_name();
            affinity = get_affinity(node_with_affinity_name);
        } else {
            affinity = get_affinity(node->get_friendly_name());
        }
        node->get_rt_info()["affinity"] = affinity;
        affinities += "\t{\"" + node->get_friendly_name() + "\",\t\t\"" + affinity + "\"}\n";
    }
    affinities += "}";
    affinities += "\nseed = " + std::to_string(std::get<Function>(param)._seed);
    return affinities;
}

TEST_P(HeteroSyntheticTest, someLayersToMajorPluginOthersToFallback) {
    auto affinities = SetUpAffinity();
    SCOPED_TRACE(affinities);
    Run();
    if (!ov::test::utils::current_test_is_disabled()) {
        ASSERT_NE(nullptr, cnnNetwork.getFunction());
    }
}

}  //  namespace HeteroTests
