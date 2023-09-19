// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/gru_sequence.hpp"

#include "transformations/op_conversions/bidirectional_sequences_decomposition.hpp"
#include "transformations/op_conversions/convert_sequences_to_tensor_iterator.hpp"
#include "openvino/pass/manager.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"

namespace ov {
namespace test {
using ov::test::utils::InputLayerType;
using ov::test::utils::SequenceTestsMode;
using ngraph::helpers::is_tensor_iterator_exist;

std::string GRUSequenceTest::getTestCaseName(const testing::TestParamInfo<GRUSequenceParams> &obj) {
    std::vector<InputShape> shapes;
    SequenceTestsMode mode;
    std::vector<std::string> activations;
    std::vector<float> activations_alpha;
    std::vector<float> activations_beta;
    float clip;
    bool linear_before_reset;
    ov::op::RecurrentSequenceDirection direction;
    InputLayerType WRBType;
    ov::element::Type type;
    std::string targetDevice;
    std::tie(mode, shapes, activations, clip, linear_before_reset, direction, WRBType,
                type, targetDevice) = obj.param;
    std::ostringstream result;
    result << "mode=" << mode << "_";
    result << "IS=(";
    for (const auto& shape : shapes) {
        result << ov::test::utils::partialShape2str({shape.first}) << "_";
    }
    result << ")_TS=(";
    for (const auto& shape : shapes) {
        for (const auto& item : shape.second) {
            result << ov::test::utils::vec2str(item) << "_";
        }
    }
    result << "linear_before_reset=" << linear_before_reset << "_";
    result << "activations=" << ov::test::utils::vec2str(activations) << "_";
    result << "direction=" << direction << "_";
    result << "WRBType=" << WRBType << "_";
    result << "clip=" << clip << "_";
    result << "IT=" << type.get_type_name() << "_";
    result << "targetDevice=" << targetDevice << "_";
    return result.str();
}

void GRUSequenceTest::SetUp() {
    std::vector<InputShape> shapes;
    std::vector<std::string> activations;
    std::vector<float> activations_alpha;
    std::vector<float> activations_beta;
    float clip;
    bool linear_before_reset;
    ov::op::RecurrentSequenceDirection direction;
    InputLayerType wbr_type;
    std::tie(m_mode, shapes, activations, clip, linear_before_reset, direction, wbr_type,
            inType, targetDevice) = this->GetParam();
    outType = inType;
    init_input_shapes(shapes);
    if (inType == ElementType::bf16 || inType == ElementType::f16) {
        rel_threshold = 1e-2;
    }

    const size_t seq_lengths = targetStaticShapes.front()[0][1];
    const size_t hidden_size = targetStaticShapes.front()[1][2];
    const size_t input_size = targetStaticShapes.front()[0][2];
    const size_t num_directions = direction == ov::op::RecurrentSequenceDirection::BIDIRECTIONAL ? 2 : 1;
    const size_t batch = inputDynamicShapes[0][0].is_static() ? inputDynamicShapes[0][0].get_length() :
        inputDynamicShapes[1][0].is_static() ? inputDynamicShapes[1][0].get_length() :
        inputDynamicShapes.size() > 2 && inputDynamicShapes[2][0].is_static() ? inputDynamicShapes[2][0].get_length() :
        1lu;


    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(inType, inputDynamicShapes[0]),
                               std::make_shared<ov::op::v0::Parameter>(inType, inputDynamicShapes[1])};

    const auto& w_shape = ov::Shape{num_directions, 3 * hidden_size, input_size};
    const auto& r_shape = ov::Shape{num_directions, 3 * hidden_size, hidden_size};
    const auto& b_shape = ov::Shape{num_directions, (linear_before_reset ? 4 : 3) * hidden_size};

    std::shared_ptr<ov::Node> seq_lengths_node;
    if (m_mode == SequenceTestsMode::CONVERT_TO_TI_MAX_SEQ_LEN_PARAM ||
        m_mode == SequenceTestsMode::CONVERT_TO_TI_RAND_SEQ_LEN_PARAM ||
        m_mode == SequenceTestsMode::PURE_SEQ_RAND_SEQ_LEN_PARAM) {
        auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, inputDynamicShapes[2]);
        param->set_friendly_name("seq_lengths");
        params.push_back(param);
        seq_lengths_node = param;
    } else if (m_mode == SequenceTestsMode::CONVERT_TO_TI_RAND_SEQ_LEN_CONST ||
               m_mode == SequenceTestsMode::PURE_SEQ_RAND_SEQ_LEN_CONST) {
        auto tensor = ov::test::utils::create_and_fill_tensor(ov::element::i64, targetStaticShapes[0][2], seq_lengths, 0);
        seq_lengths_node = std::make_shared<ov::op::v0::Constant>(tensor);
    } else {
        std::vector<int64_t> lengths(batch, seq_lengths);
        seq_lengths_node = std::make_shared<ov::op::v0::Constant>(ov::element::i64, targetStaticShapes[0][2], lengths);
    }

    std::shared_ptr<ov::Node> w, r, b;
    if (wbr_type == InputLayerType::PARAMETER) {
        const auto w_param = std::make_shared<ov::op::v0::Parameter>(inType, w_shape);
        const auto r_param = std::make_shared<ov::op::v0::Parameter>(inType, r_shape);
        const auto b_param = std::make_shared<ov::op::v0::Parameter>(inType, b_shape);
        w = w_param;
        r = r_param;
        b = b_param;
        params.push_back(w_param);
        params.push_back(r_param);
        params.push_back(b_param);
    } else {
        auto tensor_w = ov::test::utils::create_and_fill_tensor(inType, w_shape);
        w = std::make_shared<ov::op::v0::Constant>(tensor_w);

        auto tensor_R = ov::test::utils::create_and_fill_tensor(inType, r_shape);
        r = std::make_shared<ov::op::v0::Constant>(tensor_R);

        auto tensor_B = ov::test::utils::create_and_fill_tensor(inType, b_shape);
        b = std::make_shared<ov::op::v0::Constant>(tensor_B);
    }

    auto gru_sequence = std::make_shared<ov::op::v5::GRUSequence>(params[0], params[1], seq_lengths_node, w, r, b, hidden_size, direction,
                                                            activations, activations_alpha, activations_beta, clip, linear_before_reset);
    ov::OutputVector results{std::make_shared<ov::op::v0::Result>(gru_sequence->output(0)),
                             std::make_shared<ov::op::v0::Result>(gru_sequence->output(1))};
    function = std::make_shared<ov::Model>(results, params, "gru_sequence");

    bool is_pure_sequence = (m_mode == SequenceTestsMode::PURE_SEQ ||
                             m_mode == SequenceTestsMode::PURE_SEQ_RAND_SEQ_LEN_PARAM ||
                             m_mode == SequenceTestsMode::PURE_SEQ_RAND_SEQ_LEN_CONST);
    if (!is_pure_sequence) {
        ov::pass::Manager manager;
        if (direction == ov::op::RecurrentSequenceDirection::BIDIRECTIONAL)
            manager.register_pass<ov::pass::BidirectionalGRUSequenceDecomposition>();
        manager.register_pass<ov::pass::ConvertGRUSequenceToTensorIterator>();
        manager.run_passes(function);
        bool ti_found = is_tensor_iterator_exist(function);
        EXPECT_EQ(ti_found, true);
    } else {
        bool ti_found = is_tensor_iterator_exist(function);
        EXPECT_EQ(ti_found, false);
    }
}

void GRUSequenceTest::generate_inputs(const std::vector<ov::Shape>& target_input_static_shapes) {
    inputs.clear();
    auto params = function->get_parameters();
    OPENVINO_ASSERT(target_input_static_shapes.size() >= params.size());
    for (int i = 0; i < params.size(); i++) {
        auto tensor = ov::test::utils::create_and_fill_tensor(params[i]->get_element_type(), target_input_static_shapes[i], m_max_seq_len, 0);
        inputs.insert({params[i], tensor});
    }
}
} //  namespace test
} //  namespace ov