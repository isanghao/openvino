// Copyright (C) 2023-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/fetch_utils.cl"

// inline uint FUNC(print_matrix_half)(__constant char *title, half* buf, int rows, int cols) {
//     printf("%s\n", title);
//     for (int i = 0; i < rows; i++) {
//         for (int j = 0; j < cols; j++) {
//             printf("%.3f  ", (float)buf[cols * i + j]);
//         }
//         printf("\n");
//     }
// }

// inline uint FUNC(print_matrix_float)(__constant char *title, float* buf, int rows, int cols) {
//     printf("%s\n", title);
//     for (int i = 0; i < rows; i++) {
//         for (int j = 0; j < cols; j++) {
//             printf("%.3f  ", (float)buf[cols * i + j]);
//         }
//         printf("\n");
//     }
// }

/*
 * DEFINES
 * d:  DEPTH_SIZE
 * Br: BLK_ROW_SIZE
 * Bc: BLK_COL_SIZE
 * Tr: NUM_BLK_ROW
 * Tc: NUM_BLK_COL
 * Br * d: Q_BLK_SIZE
 * Bc * d: K_BLK_SIZE
 * Bc * d: V_BLK_SIZE
 * Br * Bc: SCORE_MAT_SIZE
 * Br * d: OUT_BLK_SIZE
 * NUM_COL_THREAD
 * COL_THREAD_SIZE
 */
// #adefine SUB_GROUP_SIZE 8
#define SUB_GROUP_VEC_TYPE half8

#ifdef SUB_GROUP_SIZE
REQD_SUB_GROUP_SIZE(SUB_GROUP_SIZE)
#endif
// __attribute__((reqd_work_group_size(1, 1, 16)))
KERNEL(mha_opt)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* inputq,
    const __global INPUT1_TYPE* inputk,
    const __global INPUT2_TYPE* inputv,
    __global OUTPUT_TYPE* output)
{
    const uint b = (uint)get_global_id(0) / OUTPUT_FEATURE_NUM; // batch index
    const uint f = (uint)get_global_id(0) % OUTPUT_FEATURE_NUM; // head index
    const uint block_id = (uint)get_global_id(1);
    // const uint row_id = (uint)get_global_id(2) / NUM_COL_THREAD;
    // const uint col_tid = (uint)get_global_id(2) % NUM_COL_THREAD;
    const uint row_id = (uint)get_global_id(2) % BLK_ROW_SIZE;
    const uint col_tid = (uint)get_global_id(2) / BLK_ROW_SIZE;

    const int col_t_start = col_tid * 8;
    const int col_t_end = col_t_start + 8;
    const int col_t_start2 = col_tid * 16;
    const int col_t_end2 = col_t_start2 + 16;
    half p_l = 0;
    half l = 0;
    __local half p_m[BLK_ROW_SIZE];
    __local half m[BLK_ROW_SIZE];
    __local half P[SCORE_MAT_SIZE];   // SCORE_MAT_SIZE = Br * Bc
    __local half O[OUT_BLK_SIZE];     // OUT_BLK_SIZE = Br * d
    __local half k_block[K_BLK_SIZE];
    __local half v_block[V_BLK_SIZE];
    __local half q_block[Q_BLK_SIZE];

    if (col_tid == 0) {
        p_m[row_id] = -HALF_MAX;
        m[row_id] = -HALF_MAX;
    }
#define MEASURE_BLOCK_1
#define MEASURE_BLOCK_2
#define MEASURE_BLOCK_3
#define MEASURE_BLOCK_4
#define AMEASURE_BLOCK_5
#define RETURN_BLOCK_4
#define MEASURE
    // Read i-th row of Q block
    half accum = 0.0;
    const int q_row_idx = BLK_ROW_SIZE * block_id + row_id;
    for (int c = col_t_start2; c < col_t_end2; c++) {
        // Replace GET_INDEX_SAFE, it's slow.
        q_block[DEPTH_SIZE * row_id + c] = inputq[0];
        O[DEPTH_SIZE * row_id + c] = 0.0f;
#ifdef MEASURE_BLOCK_1
        accum += q_block[DEPTH_SIZE * row_id + c];
#endif
    }

    barrier(CLK_LOCAL_MEM_FENCE);
#ifdef RETURN_BLOCK_1
    output[0] = accum;
    return;
#endif
    __local half row_max[BLK_ROW_SIZE][NUM_COL_THREAD];
    row_max[row_id][col_tid] = -HALF_MAX;
    half row_sum = 0.0f;

// note: DEPTH_SIZE is supposed to be multiple of 4 at this moment
#define QK_VEC_TYPE half4
#define QK_VEC_SIZE 4

#define VEC_TYPE half4
#define VEC_SIZE 4

    for (int j = 0; j < NUM_BLK_COL; j++) {
        // Fill Key block
        const int x_offset = BLK_COL_SIZE * j; // X-axis
        unroll_for (int y = row_id; y < DEPTH_SIZE; y += BLK_ROW_SIZE) {
            int kidx = INPUT1_GET_INDEX(b, f, y, x_offset);
            unroll_for (int x = col_t_start; x < col_t_end; x++) {
                k_block[(DEPTH_SIZE * x) + y] = inputk[0];
#ifdef MEASURE_BLOCK_2
                accum += k_block[(DEPTH_SIZE * x) + y];
#endif
            }
        }
#ifdef RETURN_BLOCK_2
        continue;
#endif

        // Fill Value block
        const int y_offset = BLK_COL_SIZE * j; // Y-axis
        unroll_for (int y = row_id; y < BLK_COL_SIZE; y += BLK_ROW_SIZE) {
            int vidx = INPUT2_GET_INDEX(b, f, y_offset + y, 0);
            unroll_for (int x = col_t_start2; x < col_t_end2; x++) {
                v_block[(BLK_COL_SIZE * x) + y] = inputv[0];
#ifdef MEASURE_BLOCK_3
                accum += v_block[(BLK_COL_SIZE * x) + y];
#endif
            }
        }


        barrier(CLK_LOCAL_MEM_FENCE);
#ifdef RETURN_BLOCK_3
        continue;
#endif

        // S = matmul(Q, K) and get max value.
#ifndef SUB_GROUP_SIZE
        for (int c = col_t_start; c < col_t_end; c++) {
            QK_VEC_TYPE acc4 = 0.f;
            unroll_for (int d = 0; d < DEPTH_SIZE; d += QK_VEC_SIZE) {
                acc4 = mad(*(QK_VEC_TYPE*)(q_block + DEPTH_SIZE * row_id + d), *(QK_VEC_TYPE*)(k_block + DEPTH_SIZE * c + d), acc4);
            }
            // unroll_for (int d = 0; d < DEPTH_SIZE; d += 16) {
            //     half16 q = vload16(0, (QK_VEC_TYPE*)(q_block + DEPTH_SIZE * row_id + d));
            //     half16 k = vload16(0, (QK_VEC_TYPE*)(k_block + DEPTH_SIZE * c + d));
            //     unroll_for (int i = 0; i < 16; i++)
            //         acc4 = mad(q[i], k[i], acc4);
            // }
#if QK_VEC_SIZE > 1
            half acc = 0.f;
            unroll_for (int i = 0; i < QK_VEC_SIZE; i++) {
                acc += acc4[i];
            }
#else
            half acc = acc4;
#endif
            P[BLK_COL_SIZE * row_id + c] = acc;
            row_max[row_id][col_tid] = max(row_max[row_id][col_tid], acc);
#ifdef MEASURE_BLOCK_4
            accum += P[BLK_COL_SIZE * row_id + c];
            // accum += row_max;
#    endif
        }

#else /* SUB_GROUP_SIZE */
        int offset = INPUT1_GET_INDEX(b, f, 0, 0);
        for (int c = 0; c < BLK_COL_SIZE; c++) {
            SUB_GROUP_VEC_TYPE acc16 = 0.f;
            ushort *k_ptr = k_block + offset + DEPTH_SIZE * c;

            for (int d = 0; d < DEPTH_SIZE; d += SUB_GROUP_SIZE) {
                SUB_GROUP_VEC_TYPE a = *(SUB_GROUP_VEC_TYPE*)(q_block + DEPTH_SIZE * row_id + d);
                SUB_GROUP_VEC_TYPE b;
                // half __b = as_half(_sub_group_block_read_us((const __global ushort*)(k_ptr) + (0)));
                half __b = as_half(_sub_group_block_read_us((const __local ushort*)(k_block + DEPTH_SIZE * c + d) + (0)));
                // half __b = as_half(_sub_group_block_read_us((const __global ushort*)(inputk + offset + DEPTH_SIZE * c + d) + (0)));
                k_ptr += SUB_GROUP_SIZE;

                unroll_for(int s = 0; s < SUB_GROUP_SIZE; s++) {
                    b[s] = sub_group_broadcast(__b, s);
                }
                acc16 = mad(a, b, acc16);
            }

            half acc = 0.f;

            unroll_for(int i = 0; i < SUB_GROUP_SIZE; i++) {
                acc += acc16[i];
            }

            P[BLK_COL_SIZE * row_id + c] = acc;

            row_max = max(row_max, acc);
#    ifdef MEASURE_BLOCK_4
            accum += P[BLK_COL_SIZE * row_id + c];
#    endif
        }
#endif
#ifdef RETURN_BLOCK_4
        continue;
#endif
        barrier(CLK_LOCAL_MEM_FENCE);

        if (col_tid == 0) {
            half rm = -HALF_MAX;
            unroll_for (int c = 0; c < NUM_COL_THREAD; c++) {
                rm = max(rm , row_max[row_id][c]);
            }
            m[row_id] = max(p_m[row_id], rm);
        }

        barrier(CLK_LOCAL_MEM_FENCE);
        // Calculate P
        row_sum = 0.0f;
        VEC_TYPE e = 0.f;
        unroll_for (int x = col_t_start; x < col_t_end; x += VEC_SIZE) {
            e = exp((*(VEC_TYPE*)(P + BLK_COL_SIZE * row_id + x) - (VEC_TYPE)m[row_id]));
            *(VEC_TYPE*)(P + BLK_COL_SIZE * row_id + x) = e;
#if VEC_SIZE > 1
            // unroll_for (int i = 0; i < VEC_SIZE; i++) {
            //     row_sum += e[i];
            // }
#else
            row_sum += e;
#endif
        }
#ifdef MEASURE_BLOCK_5
        accum += row_sum;
#endif
        barrier(CLK_LOCAL_MEM_FENCE);

        unroll_for (int x = 0; x < BLK_COL_SIZE; x++) {
            row_sum += P[BLK_COL_SIZE * row_id + x];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
#ifdef RETURN_BLOCK_5
        continue;
#endif
        // Calculate l value.
        half exp_m = exp(p_m[row_id] - m[row_id]);
        l = exp_m * p_l + row_sum;

        // Calculate O + PV block.
        for (int d = col_t_start2; d < col_t_end2; d++) {
            half acc = 0.f;
            VEC_TYPE acc4 = 0.f;
            unroll_for (int c = 0; c < BLK_COL_SIZE; c += VEC_SIZE) {
                acc4 = c;
            }
#if VEC_SIZE > 1
            unroll_for (int i = 0; i < VEC_SIZE; i++) {
                acc += acc4[i];
            }
#else
            acc = acc4;
#endif
            O[DEPTH_SIZE * row_id + d] = acc;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // Set m(j-1) and l(j-1)
        p_m[row_id] = m[row_id];
        p_l = l;
    }

    const int out_row_idx = BLK_ROW_SIZE * block_id + row_id;
    int oidx = OUTPUT_GET_INDEX(b, f, out_row_idx, 0);
    unroll_for (int c = col_t_start2; c < col_t_end2; c++) {
        output[oidx + c] = O[DEPTH_SIZE * row_id + c]/l;
    }
#ifdef MEASURE
    output[0] = accum;
#endif
}
