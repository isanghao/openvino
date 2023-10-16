// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/op/mha.hpp"
#include "matmul_shape_inference.hpp"

namespace ov {
namespace intel_gpu {
namespace op {

MhaFusion::MhaFusion(const ov::Output<Node>& Q,
                     const ov::Output<Node>& K,
                     const ov::Output<Node>& V,
                     const ov::element::Type output_type)
    : Op({Q, K, V}), m_output_type(output_type) {
    validate_and_infer_types();
}

void MhaFusion::validate_and_infer_types() {
    const auto input_size = get_input_size();
    NODE_VALIDATION_CHECK(this,
        input_size == 3,
        "Number of inputs is incorrect. Current value is: ",
        input_size,
        ", expected 3.");

    m_output_type == ov::element::undefined ? get_input_element_type(0) : m_output_type;
}

bool MhaFusion::visit_attributes(ov::AttributeVisitor &visitor) {
    visitor.on_attribute("output_type", m_output_type);
    return true;
}

std::shared_ptr<ov::Node> MhaFusion::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    check_new_args_count(this, new_args);

    return std::make_shared<MhaFusion>(new_args.at(0), new_args.at(1), new_args.at(2), m_output_type);
}

}  // namespace op
}  // namespace intel_gpu
}  // namespace ov
