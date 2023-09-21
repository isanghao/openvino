// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"
#include "include/batch_headers/fetch_weights.cl"
#include "include/batch_headers/imad.cl"

// ======================================================================================
// Host side jit-constants:
// ======================================================================================
// SIMD            { 16 } - number of work-items in sub-group/simd size;
//                          currently limited to only 16
// FEATURES_PER_WI { 16, 32 } - number of output features calculated by one
//                              work-item; must be multiple of SIMD
// LWG_DEPTH       { 1..16 } - number of sub-groups per work-group that will
//                             calculate the same output features, but accumulating
//                             different input features;
//                             helps in low EU utilization, but requires additional
//                             barrier and local memory reads/writes
// FORCE_PREFETCH { 0, 1 }   - flag to force the compiler to generate explicit
//                             data prefetching; requires additional global barrier
// ======================================================================================

#define FSV 4
#define WEIGHTS_OSV 16

#define DEQUANTIZED_TYPE float

#define INPUT_TYPE4       MAKE_VECTOR_TYPE(INPUT0_TYPE, 4)
#define FILTER_TYPE4      MAKE_VECTOR_TYPE(FILTER_TYPE, 4)
#define OUTPUT_TYPE4      MAKE_VECTOR_TYPE(OUTPUT_TYPE, 4)
#define BIAS_TYPE4        MAKE_VECTOR_TYPE(BIAS_TYPE, 4)
#define DEQUANTIZED_TYPE4 MAKE_VECTOR_TYPE(DEQUANTIZED_TYPE, 4)

#define AS_INPUT_TYPE4(val)       CAT(as_, INPUT_TYPE4)(val)
#define AS_FILTER_TYPE4(val)      CAT(as_, FILTER_TYPE4)(val)
#define TO_DEQUANTIZED_TYPE(val)  CAT(convert_, DEQUANTIZED_TYPE)(val)
#define TO_DEQUANTIZED_TYPE4(val) CAT(convert_, DEQUANTIZED_TYPE4)(val)
#define TO_OUTPUT_TYPE4(val)      CAT(convert_, OUTPUT_TYPE4)(val)

#define GET_INPUT_INDEX(b, f, y, x)   GET_DATA_B_FS_YX_FSV4_INDEX(INPUT0, b, f, y, x)
#define GET_WEIGHTS_INDEX(b, f, y, x) GET_FILTER_OS_IS_YX_OSV16_ISV4_INDEX(FILTER, b, f, y, x)
#define GET_OUTPUT_INDEX(b, f, y, x)  GET_DATA_B_FS_YX_FSV4_INDEX(OUTPUT, b, f, y, x)

#define OUTPUT_FS_PITCH (OUTPUT_FEATURE_PITCH * FSV)
#define INPUT_FS_PITCH (INPUT0_FEATURE_PITCH * FSV)

#define WEIGHTS_IS_PITCH (WEIGHTS_OSV * FSV)
#define WEIGHTS_OS_PITCH ((FILTER_IFM_NUM + FSV - 1) / FSV * FSV * WEIGHTS_OSV)

#define MAX_SPATIAL_SIZE (OUTPUT_SIZE_X * OUTPUT_SIZE_Y)
#define SAFE_SPATIAL (MAX_SPATIAL_SIZE % SIMD == 0)
#define SAFE_FEATURES (OUTPUT_FEATURE_NUM % FEATURES_PER_WI == 0)

// Dispatch dimensions:
//     b x f               x spatial (y * x)
// WI: 1 x FEATURES_PER_WI x 1
// SG: 1 x FEATURES_PER_WI x SIMD

REQD_SUB_GROUP_SIZE(SIMD)
__attribute__((reqd_work_group_size(SIMD, 1, LWG_DEPTH)))
KERNEL(convolution)(
    const __global uint          *input,
    __global OUTPUT_TYPE4        *output,
    const __global int           *weights
#if BIAS_TERM
    , const __global BIAS_TYPE   *biases
#endif
#if HAS_FUSED_OPS_DECLS
    , FUSED_OPS_DECLS
#endif
) {
    const uint f = (uint)get_global_id(1) * FEATURES_PER_WI;
#if LWG_DEPTH == 1
    const uint yx = (uint)get_global_id(0);
#else
    const uint yx = (uint)get_group_id(0) * SIMD + get_sub_group_local_id();
#endif
    const uint x = yx % OUTPUT_SIZE_X;
    const uint y = yx / OUTPUT_SIZE_X;
#if LWG_DEPTH == 1
    const uint b = (uint)get_global_id(2);
    const uint lwg_d = 0;
#else
    const uint b = get_group_id(2);
    const uint lwg_d = get_sub_group_id();
#endif

    int dotProd[FEATURES_PER_WI] = { 0 };

    int wei_sg[FEATURES_PER_WI / SIMD];
    int wei_sg_pre[FEATURES_PER_WI / SIMD];

    uint input_offset = GET_INPUT_INDEX(b, lwg_d * FSV, y * STRIDE_SIZE_Y, x * STRIDE_SIZE_X) / FSV;
    uint weights_offset = GET_WEIGHTS_INDEX(f + get_sub_group_local_id(), lwg_d * FSV, 0, 0) / FSV;

    // Prefetch input and weights
    uint in_u = input[input_offset];
    input_offset += INPUT_FS_PITCH / FSV * LWG_DEPTH;

    unroll_for (uint wi = 0; wi < (FEATURES_PER_WI / SIMD); ++wi) {
        const uint weights_os_offset = (wi * SIMD / WEIGHTS_OSV) * (WEIGHTS_OS_PITCH / FSV);
        const uint weights_osv_offset = (wi * SIMD % WEIGHTS_OSV) * (FSV / FSV);
        wei_sg_pre[wi] = weights[weights_offset + (weights_os_offset + weights_osv_offset)];
    }
    weights_offset += WEIGHTS_IS_PITCH / FSV * LWG_DEPTH;

#if FORCE_PREFETCH
    // Forces the compiler to emit prefetching send's before main loop.
    barrier(CLK_GLOBAL_MEM_FENCE);
#endif

    // Process four input features in one iteration - IMAD.
    for (uint k = 0; k < (FILTER_IFM_NUM + FSV - 1) / FSV / LWG_DEPTH; ++k) {
        INPUT_TYPE4 in_val = AS_INPUT_TYPE4(in_u);

        unroll_for (uint wi = 0; wi < (FEATURES_PER_WI / SIMD); ++wi) {
            wei_sg[wi] = wei_sg_pre[wi];
        }

        in_u = input[input_offset];
        input_offset += INPUT_FS_PITCH / FSV * LWG_DEPTH;

        unroll_for (uint wi = 0; wi < (FEATURES_PER_WI / SIMD); ++wi) {
            const uint weights_os_offset = (wi * SIMD / WEIGHTS_OSV) * (WEIGHTS_OS_PITCH / FSV);
            const uint weights_osv_offset = (wi * SIMD % WEIGHTS_OSV) * (FSV / FSV);
            wei_sg_pre[wi] = weights[weights_offset + (weights_os_offset + weights_osv_offset)];
        }
        weights_offset += WEIGHTS_IS_PITCH / FSV * LWG_DEPTH;

        FILTER_TYPE4 wei_val;
        unroll_for (uint out_fi = 0; out_fi < FEATURES_PER_WI; ++out_fi) {
            int wei_i = _sub_group_shuffle(wei_sg[out_fi / SIMD], out_fi % SIMD);
            FILTER_TYPE4 wei_val = AS_FILTER_TYPE4(wei_i);
#if 0
            dotProd[out_fi] = IMAD(dotProd[out_fi], in_val, wei_val);
#else
            dotProd[out_fi] = IMAD(0, in_val, wei_val);
            int tmp = 0;
            tmp += in_val[0] * wei_val[0];
            tmp += in_val[1] * wei_val[1];
            tmp += in_val[2] * wei_val[2];
            tmp += in_val[3] * wei_val[3];
            printf("%d %d %d - %u * %d => %d, %d\n", (int)get_global_id(0),(int)get_global_id(1), (int)get_global_id(2), in_val, wei_val, dotProd[out_fi], tmp);
#endif
        }
    }

}

#undef FSV
#undef WEIGHTS_OSV

#undef DEQUANTIZED_TYPE

#undef INPUT_TYPE4
#undef FILTER_TYPE4
#undef OUTPUT_TYPE4
#undef BIAS_TYPE4
#undef DEQUANTIZED_TYPE4

#undef AS_INPUT_TYPE4
#undef AS_FILTER_TYPE4
#undef TO_DEQUANTIZED_TYPE
#undef TO_DEQUANTIZED_TYPE4
#undef TO_OUTPUT_TYPE4

#undef GET_INPUT_INDEX
#undef GET_WEIGHTS_INDEX
#undef GET_OUTPUT_INDEX

#undef OUTPUT_FS_PITCH
#undef INPUT_FS_PITCH

#undef WEIGHTS_IS_PITCH
#undef WEIGHTS_OS_PITCH

#undef MAX_SPATIAL_SIZE
#undef SAFE_SPATIAL
#undef SAFE_FEATURES
