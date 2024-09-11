// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"

#if OUTPUT_DIMS != 4
#error "dynamic_quantize_gpu_opt.cl: Unsupported output dimension"
#endif

#define VLOAD_N CAT(vload, VEC_SIZE)
#define VSTORE_N CAT(vstore, VEC_SIZE)
#define CONVERT_CHAR_N CAT(convert_char, VEC_SIZE)
#define AS_TYPE_N_(type, n, x) as_##type##n(x)
#define AS_TYPE_N(type, n, x) AS_TYPE_N_(type, n, x)
#define AS_INPUT_TYPE_N(x) AS_TYPE_N(INPUT0_TYPE, VEC_SIZE, x)

#if QUANTIZE_GROUP_SIZE <= 128
KERNEL(dynamic_quantize_gpu_opt)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* input,
    __global OUTPUT_TYPE* output,
    __global OUTPUT1_TYPE* output_scale) {
    const uint bf = get_global_id(0);
    const uint b = bf / INPUT0_FEATURE_NUM;
    const uint f = bf % INPUT0_FEATURE_NUM;
    const uint y_grp = get_global_id(1);

    const uint input_offset = INPUT0_GET_INDEX(b, f, y_grp * QUANTIZE_GROUP_SIZE, 0);
    const uint output_offset = OUTPUT_GET_INDEX(b, f, y_grp * QUANTIZE_GROUP_SIZE, 0);
    const uint quantize_block = QUANTIZE_GROUP_SIZE / 4;
    half4 input_0[quantize_block];
    char4 quantized_value[quantize_block];
    half  max[quantize_block];

    unroll_for (uint i = 0 ; i < quantize_block; ++i) {
        input_0[i] = vload4(0, &input[input_offset + i * 4]);
        max[i] = fmax(fmax(fabs(input_0[i][0]), fabs(input_0[i][1])), fmax(fabs(input_0[i][2]), fabs(input_0[i][3])));
    }

    half max_value = 0.001;
    for (uint i = 0 ; i < quantize_block; i+=8) {
        half temp = fmax(fmax(fmax(max[i], max[i+1]), fmax(max[i+2], max[i+3])),
                                fmax(fmax(max[i+4], max[i+5]), fmax(max[i+6], max[i+7])));
        max_value = fmax(max_value, temp);
    }

    half quan_scale = max_value / 128;

    unroll_for (uint i = 0 ; i < quantize_block; ++i) {
        quantized_value[i] = convert_char4(input_0[i] / (half4)quan_scale);
        vstore4(quantized_value[i], 0, &output[output_offset + i * 4]);
    }

    output_scale[OUTPUT1_GET_INDEX(b, f, y_grp, 0)] = quan_scale;
}

#else

REQD_SUB_GROUP_SIZE(SIMD)
KERNEL(dynamic_quantize_gpu_opt)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* input,
    __global OUTPUT_TYPE* output,
    __global OUTPUT1_TYPE* output_scale)
{
    const uint bf = (uint)get_global_id(2);
    const uint sglid = get_sub_group_local_id();
    const uint local_id = (uint)get_local_id(1);

    const uint block_size = SIMD * VEC_SIZE;
    const uint b_offset = bf * INPUT0_FEATURE_PITCH;

    const uint offset = b_offset + VEC_SIZE * sglid;

    const uint iteration = ALIGNED_BLOCK_NUM / BLOCK_NUM;

    __local half local_mem[BLOCK_NUM];

    MAKE_VECTOR_TYPE(INPUT0_TYPE, VEC_SIZE) val[iteration];
    MAKE_VECTOR_TYPE(INPUT0_TYPE, VEC_SIZE) abs_val;
    half max = 0.0h;
    half grp_max = 0.001h;
    half max_value;

    unroll_for(int i = 0; i < iteration; ++i) {
        if ((local_id * iteration + i) >= TOTAL_BLOCK_NUM)
            continue;

        val[i] = AS_INPUT_TYPE_N(VLOAD_N(0, input + offset + ((local_id * iteration + i) * block_size)));
        abs_val = fabs(val[i]);

        unroll_for (int j = 0; j < VEC_SIZE; j++) {
            max = fmax(max, abs_val[j]);
        }

        grp_max = fmax(grp_max, max);
    }

    max_value = sub_group_reduce_max(grp_max);
    if (sglid == 0)
        local_mem[local_id] = max_value;

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int j = 0; j < BLOCK_NUM; j++) {
        max_value = fmax(max_value, local_mem[j]);
    }

    half scale = 127.0h / max_value;

    unroll_for(int i = 0; i < iteration; ++i) {
        if ((local_id * iteration + i) >= TOTAL_BLOCK_NUM)
            continue;

        val[i] *= scale;
        VSTORE_N(CONVERT_CHAR_N(val[i]), 0, output + offset + ((local_id * iteration + i) * block_size));
    }

    if (sglid == 0 && local_id == 0)
        output_scale[bf] = 1.0h / scale;
}
#endif