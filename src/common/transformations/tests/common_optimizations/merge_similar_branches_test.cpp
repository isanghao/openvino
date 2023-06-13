// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/merge_similar_branches.hpp"

#include <gtest/gtest.h>

#include <random>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "openvino/opsets/opset11.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/runtime/core.hpp"

using namespace std;
using namespace testing;
using namespace ov;
using namespace ov::element;
using namespace ov::pass;

using MergeSimilarBranchesTest = TransformationTests;

class MergeSimilarBranchesTestF : public TransformationTestsF {
public:
    MergeSimilarBranchesTestF() {
        comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
        comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
        comparator.enable(FunctionsComparator::CmpValues::CONSUMERS_COUNT);

        manager.register_pass<MergeSimilarBranches>();
    }
};

class DISABLED_MergeSimilarBranchesTestF : public MergeSimilarBranchesTestF {};

TEST_F(MergeSimilarBranchesTestF, unary_nodes) {
    using namespace ov::opset11;
    {
        const auto input = make_shared<Parameter>(f32, PartialShape{1});
        const auto neg = make_shared<Negative>(input);
        const auto abs1 = make_shared<Abs>(neg);
        const auto abs2 = make_shared<Abs>(neg);
        const auto abs3 = make_shared<Abs>(neg);
        const auto relu1 = make_shared<Relu>(abs1);
        const auto relu2 = make_shared<Relu>(abs2);
        const auto relu3 = make_shared<Relu>(abs3);
        const auto concat = make_shared<Concat>(OutputVector{relu1, relu2, relu3}, 0);
        model = make_shared<Model>(NodeVector{concat}, ParameterVector{input});
    }
    {
        const auto input = make_shared<Parameter>(f32, PartialShape{1});
        const auto neg = make_shared<Negative>(input);
        const auto abs = make_shared<Abs>(neg);
        const auto relu = make_shared<Relu>(abs);
        const auto concat = make_shared<Concat>(OutputVector{relu, relu, relu}, 0);
        model_ref = make_shared<Model>(NodeVector{concat}, ParameterVector{input});
    }
}

TEST_F(MergeSimilarBranchesTestF, binary_nodes_equal_constants) {
    using namespace ov::opset11;
    {
        const auto input = make_shared<Parameter>(f32, PartialShape{});
        const auto constant1 = Constant::create(f32, {1}, {1});
        const auto constant2 = Constant::create(f32, {1}, {1});
        const auto add1 = make_shared<Add>(input, constant1);
        const auto add2 = make_shared<Add>(input, constant2);
        const auto mul = make_shared<Multiply>(add1, add2);
        model = make_shared<Model>(NodeVector{mul}, ParameterVector{input});
    }
    {
        const auto input = make_shared<Parameter>(f32, PartialShape{});
        const auto constant = Constant::create(f32, {1}, {1});
        const auto add = make_shared<Add>(input, constant);
        const auto mul = make_shared<Multiply>(add, add);
        model_ref = make_shared<Model>(NodeVector{mul}, ParameterVector{input});
    }
}

TEST_F(MergeSimilarBranchesTestF, binary_nodes_same_op_node) {
    using namespace ov::opset11;
    {
        const auto input1 = make_shared<Parameter>(f32, PartialShape{2, 3});
        const auto input2 = make_shared<Parameter>(f32, PartialShape{3, 2});
        const auto new_shape = Constant::create(i64, {2}, {2, 3});
        const auto reshape = make_shared<Reshape>(input2, new_shape, false);
        const auto sub1 = make_shared<Subtract>(input1, reshape);
        const auto sub2 = make_shared<Subtract>(input1, reshape);
        const auto mul = make_shared<Multiply>(sub1, sub2);
        model = make_shared<Model>(NodeVector{mul}, ParameterVector{input1, input2});
    }
    {
        const auto input1 = make_shared<Parameter>(f32, PartialShape{2, 3});
        const auto input2 = make_shared<Parameter>(f32, PartialShape{3, 2});
        const auto new_shape = Constant::create(i64, {2}, {2, 3});
        const auto reshape = make_shared<Reshape>(input2, new_shape, false);
        const auto sub = make_shared<Subtract>(input1, reshape);
        const auto mul = make_shared<Multiply>(sub, sub);
        model_ref = make_shared<Model>(NodeVector{mul}, ParameterVector{input1, input2});
    }
}

TEST_F(MergeSimilarBranchesTestF, binary_nodes_single_source) {
    using namespace ov::opset11;
    {
        const auto input = make_shared<Parameter>(u32, PartialShape{7, 3});
        const auto sub1 = make_shared<Subtract>(input, input);
        const auto sub2 = make_shared<Subtract>(input, input);
        const auto add = make_shared<Add>(sub1, sub2);
        model = make_shared<Model>(NodeVector{add}, ParameterVector{input});
    }
    {
        const auto input = make_shared<Parameter>(u32, PartialShape{7, 3});
        const auto sub1 = make_shared<Subtract>(input, input);
        const auto add = make_shared<Add>(sub1, sub1);
        model_ref = make_shared<Model>(NodeVector{add}, ParameterVector{input});
    }
}

TEST_F(MergeSimilarBranchesTestF, keep_result_producers) {
    using namespace ov::opset11;
    {
        const auto input = make_shared<Parameter>(f32, PartialShape{});
        const auto abs1 = make_shared<Abs>(input);
        const auto abs2 = make_shared<Abs>(input);
        const auto relu1 = make_shared<Relu>(abs1);
        const auto relu2 = make_shared<Relu>(abs2);
        model = make_shared<Model>(NodeVector{relu1, relu2}, ParameterVector{input});
    }
    {
        const auto input = make_shared<Parameter>(f32, PartialShape{});
        const auto abs = make_shared<Abs>(input);
        const auto relu1 = make_shared<Relu>(abs);
        const auto relu2 = make_shared<Relu>(abs);
        model_ref = make_shared<Model>(NodeVector{relu1, relu2}, ParameterVector{input});
    }
}

TEST_F(MergeSimilarBranchesTestF, multiple_consumers) {
    using namespace ov::opset11;
    {
        const auto input = make_shared<Parameter>(f32, PartialShape{1});
        const auto neg1 = make_shared<Negative>(input);
        const auto neg2 = make_shared<Negative>(input);
        const auto relu1 = make_shared<Relu>(neg1);
        const auto relu2 = make_shared<Relu>(neg2);
        const auto relu3 = make_shared<Relu>(neg2);
        const auto relu4 = make_shared<Relu>(neg2);
        const auto concat = make_shared<Concat>(OutputVector{relu1, relu2, relu3, relu4}, 0);
        model = make_shared<Model>(NodeVector{concat}, ParameterVector{input});
    }
    {
        const auto input = make_shared<Parameter>(f32, PartialShape{1});
        const auto neg = make_shared<Negative>(input);
        const auto relu = make_shared<Relu>(neg);
        const auto concat = make_shared<Concat>(OutputVector{relu, relu, relu, relu}, 0);
        model_ref = make_shared<Model>(NodeVector{concat}, ParameterVector{input});
    }
}

TEST_F(MergeSimilarBranchesTestF, different_branches) {
    using namespace ov::opset11;
    {
        const auto input1 = make_shared<Parameter>(f32, PartialShape{7});
        const auto input2 = make_shared<Parameter>(f32, PartialShape{7});
        const auto constant1 = Constant::create(f32, {1}, {1});
        const auto constant2 = Constant::create(f32, {1}, {11});
        const auto constant3 = Constant::create(f32, {1}, {1});
        const auto add1 = make_shared<Add>(input1, constant1);
        const auto add2 = make_shared<Add>(input1, constant2);
        const auto add3 = make_shared<Add>(input1, input2);
        const auto add4 = make_shared<Add>(input1, constant3);
        const auto add5 = make_shared<Add>(input1, input2);
        const auto concat = make_shared<Concat>(OutputVector{add1, add2, add3, add4, add5}, 0);
        model = make_shared<Model>(NodeVector{concat}, ParameterVector{input1, input2});
    }
    {
        const auto input1 = make_shared<Parameter>(f32, PartialShape{7});
        const auto input2 = make_shared<Parameter>(f32, PartialShape{7});
        const auto constant1 = Constant::create(f32, {1}, {1});
        const auto constant2 = Constant::create(f32, {1}, {11});
        const auto add1 = make_shared<Add>(input1, constant1);
        const auto add2 = make_shared<Add>(input1, constant2);
        const auto add3 = make_shared<Add>(input1, input2);
        const auto concat = make_shared<Concat>(OutputVector{add1, add2, add3, add1, add3}, 0);
        model_ref = make_shared<Model>(NodeVector{concat}, ParameterVector{input1, input2});
    }
}

TEST_F(MergeSimilarBranchesTestF, different_op_versions) {
    using namespace ov::op;
    const auto input = make_shared<v0::Parameter>(f32, PartialShape{8, 128});
    const auto indices = v0::Constant::create(i64, {2}, {4, 4});
    const auto axis = v0::Constant::create(i64, {1}, {1});
    const auto gather_v7 = make_shared<v7::Gather>(input, indices, axis);
    const auto gather_v8 = make_shared<v8::Gather>(input, indices, axis);
    const auto relu1 = make_shared<v0::Relu>(gather_v7);
    const auto relu2 = make_shared<v0::Relu>(gather_v8);
    const auto add = make_shared<v1::Add>(relu1, relu2);
    model = make_shared<Model>(NodeVector{add}, ParameterVector{input});
}

TEST_F(MergeSimilarBranchesTestF, different_input_nodes) {
    using namespace ov::opset11;
    const auto input1 = make_shared<Parameter>(f32, PartialShape{7});
    const auto input2 = make_shared<Parameter>(f32, PartialShape{7});
    const auto abs = make_shared<Abs>(input2);
    const auto neg = make_shared<Negative>(input2);
    const auto prelu1 = make_shared<PRelu>(input1, abs);
    const auto prelu2 = make_shared<PRelu>(input1, neg);
    const auto concat = make_shared<Concat>(OutputVector{prelu1, prelu2}, 0);
    model = make_shared<Model>(NodeVector{concat}, ParameterVector{input1, input2});
}

TEST_F(DISABLED_MergeSimilarBranchesTestF, mixed_input_order) {
    using namespace ov::opset11;
    {
        const auto input = make_shared<Parameter>(i16, PartialShape{13, 27});
        const auto relu = make_shared<Relu>(input);
        const auto equal1 = make_shared<Equal>(input, relu);
        const auto equal2 = make_shared<Equal>(relu, input);
        const auto l_and1 = make_shared<LogicalAnd>(equal1, equal2);
        const auto l_and2 = make_shared<LogicalAnd>(equal2, equal1);
        const auto concat = make_shared<Concat>(OutputVector{l_and1, l_and2}, 0);
        model = make_shared<Model>(NodeVector{concat}, ParameterVector{input});
    }
    {
        const auto input = make_shared<Parameter>(i16, PartialShape{13, 27});
        const auto relu = make_shared<Relu>(input);
        const auto equal = make_shared<Equal>(input, relu);
        const auto l_and = make_shared<LogicalAnd>(equal, equal);
        const auto concat = make_shared<Concat>(OutputVector{l_and, l_and}, 0);
        model_ref = make_shared<Model>(NodeVector{concat}, ParameterVector{input});
    }
}

TEST(MergeSimilarBranchesTest, matmuls_fusion) {
    using namespace ov::opset11;

    const auto matmuls_input0_shape = Shape{9, 7};
    const auto matmuls_input1_shape = Shape{7, 11};
    vector<uint32_t> matmulA_input1_data(shape_size(matmuls_input1_shape));
    vector<uint32_t> matmulB_input1_data(shape_size(matmuls_input1_shape));
    vector<uint32_t> matmulC_input1_data(shape_size(matmuls_input1_shape));
    seed_seq{1, 2, 3}.generate(matmulA_input1_data.begin(), matmulA_input1_data.end());
    seed_seq{4, 5, 6}.generate(matmulB_input1_data.begin(), matmulB_input1_data.end());
    seed_seq{7, 8, 9}.generate(matmulC_input1_data.begin(), matmulC_input1_data.end());

    const auto input = make_shared<Parameter>(u32, matmuls_input0_shape);
    const auto constA = Constant::create(u32, matmuls_input1_shape, matmulA_input1_data);
    const auto constB = Constant::create(u32, matmuls_input1_shape, matmulB_input1_data);
    const auto constC = Constant::create(u32, matmuls_input1_shape, matmulC_input1_data);
    const auto matmulA = make_shared<MatMul>(input, constA);
    const auto matmulB = make_shared<MatMul>(input, constB);
    const auto matmulC = make_shared<MatMul>(input, constC);
    const auto concat = make_shared<Concat>(OutputVector{matmulA, matmulB, matmulC}, 0);

    const auto model = make_shared<Model>(NodeVector{concat}, ParameterVector{input});
    const auto cloned_model = model->clone();

    Manager manager;
    manager.register_pass<MergeSimilarBranches>();
    manager.run_passes(model);
    ASSERT_EQ(count_ops_of_type<op::v0::MatMul>(model), 1);

    auto acc_comparator = FunctionsComparator::no_default();
    acc_comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
    const auto res = acc_comparator.compare(model, cloned_model);
    ASSERT_TRUE(res.valid) << res.message;
}
