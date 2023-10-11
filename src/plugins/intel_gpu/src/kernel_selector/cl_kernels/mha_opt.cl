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
#define TILE_SIZE 4
    const uint b = (uint)get_global_id(0) / OUTPUT_FEATURE_NUM; // batch index
    const uint f = (uint)get_global_id(0) % OUTPUT_FEATURE_NUM; // head index
    const uint block_id = (uint)get_global_id(1);
    // const uint row_id = (uint)get_global_id(2) / 8;
    // const uint col_id = (uint)get_global_id(2) % 8;
    const uint row_id = (uint)get_global_id(2) % 16;
    const uint col_id = (uint)get_global_id(2) / 16;
    half p_m = -HALF_MAX;
    half m = -HALF_MAX;
    half p_l = 0;
    half l = 0;
    __local half P[SCORE_MAT_SIZE];   // SCORE_MAT_SIZE = Br * Bc
    __local half O[OUT_BLK_SIZE];     // OUT_BLK_SIZE = Br * d
    __local half k_block[K_BLK_SIZE];
    __local half v_block[V_BLK_SIZE];
    __local half q_block[Q_BLK_SIZE];
#define AMEASURE_BLOCK_1
#define AMEASURE_BLOCK_2
#define AMEASURE_BLOCK_3
#define AMEASURE_BLOCK_4
#define AMEASURE_BLOCK_5
#define ARETURN_BLOCK_5
#define AMEASURE
    // Read i-th row of Q block
    half accum = 0.0;
    const q_h_tile_size = DEPTH_SIZE/8;
    for (int r = 0; r < TILE_SIZE; r++) {
        const int y = TILE_SIZE * row_id + r;
        const int q_offset = INPUT0_GET_INDEX(b, f, BLK_ROW_SIZE * block_id + y, 0);
        for (int x = col_id * q_h_tile_size; x < (col_id + 1) * q_h_tile_size; x++) {
            // printf("q>> %d %d %d %d\n", row_id, col_id, TILE_SIZE * row_id + r, q_offset + c);
            q_block[DEPTH_SIZE * y + x] = inputq[q_offset + x];
            O[DEPTH_SIZE * y + x] = 0.0f;
    #ifdef MEASURE_BLOCK_1
            accum += q_block[DEPTH_SIZE * y + x];
    #endif
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
#ifdef RETURN_BLOCK_1
    output[0] = accum;
    return;
#endif
    half row_max = -HALF_MAX;
    half row_sum = 0.0f;
// note: DEPTH_SIZE is supposed to be multiple of 4 at this moment
#define VEC_TYPE half4
#define VEC_SIZE 4
    // const int qk_t_m = TILE_SIZE * row_id;
    // const int qk_t_n = TILE_SIZE * col_id;
    for (int j = 0; j < NUM_BLK_COL; j++) {
        // Fill Key block
        unroll_for (int r = 0; r < TILE_SIZE; r ++) {
            const int y = TILE_SIZE * row_id + r;
            int k_offset = INPUT1_GET_INDEX(b, f, y, BLK_COL_SIZE * j + TILE_SIZE * col_id + 0);
            unroll_for (int c = 0; c < TILE_SIZE; c++) {
                const int x = TILE_SIZE * col_id + c;
                k_block[DEPTH_SIZE * x + y] = inputk[k_offset + c];
#ifdef MEASURE_BLOCK_2
                accum += k_block[DEPTH_SIZE * x + y];
#endif
            }
        }
#ifdef RETURN_BLOCK_2
        continue;
#endif
        // Fill Value block
        unroll_for (int r = 0; r < TILE_SIZE; r ++) {
            const int y = TILE_SIZE * col_id + r;
            int v_offset = INPUT2_GET_INDEX(b, f, BLK_COL_SIZE * j + y, TILE_SIZE * row_id + 0);
            unroll_for (int c = 0; c < TILE_SIZE; c++) {
                const int x = TILE_SIZE * row_id + c;
                v_block[(BLK_COL_SIZE * x) + y] = inputv[v_offset + c];
                // printf("%d %d %d %d %d %d %f %f\n", b, f, y, x, (BLK_COL_SIZE * x) + y, v_offset + c, v_block[(BLK_COL_SIZE * x) + y], inputv[v_offset + c]);
#ifdef MEASURE_BLOCK_3
                accum += v_block[(BLK_COL_SIZE * x) + y];
#endif
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
#ifdef RETURN_BLOCK_3
        continue;
#endif

#define QK_VEC_TYPE half4
#define QK_VEC_SIZE 4
        // Outer loop
        QK_VEC_TYPE a_read = 0.f;
        QK_VEC_TYPE b_read = 0.f;
        half acc = 0.f;
        QK_VEC_TYPE acc4_arr[TILE_SIZE];
        for (int n = 0; n < TILE_SIZE; n++) {
            // initialize to 0.
            unroll_for (int i = 0; i < TILE_SIZE; i++)
                acc4_arr[i] = 0.f;
            const int x = TILE_SIZE * col_id + n;
            for (int k = 0; k < DEPTH_SIZE; k += QK_VEC_SIZE) {
                b_read = *(QK_VEC_TYPE*)(k_block + DEPTH_SIZE * x + k);
#if QK_VEC_SIZE == 1
                unroll_for (int m = 0; m < TILE_SIZE; m+=4) {
                    half4 __a_read = *(half4*)(q_block + DEPTH_SIZE * (TILE_SIZE * row_id + m)  + k);
                    acc4_arr[0] = mad(__a_read[0], b_read, acc4_arr[0]);
                    acc4_arr[1] = mad(__a_read[1], b_read, acc4_arr[1]);
                    acc4_arr[2] = mad(__a_read[2], b_read, acc4_arr[2]);
                    acc4_arr[3] = mad(__a_read[3], b_read, acc4_arr[3]);
                }
#else
                unroll_for (int m = 0; m < TILE_SIZE; m++) {
                    a_read = *(QK_VEC_TYPE*)(q_block + DEPTH_SIZE * (TILE_SIZE * row_id + m)  + k);
                    acc4_arr[m] = mad(a_read, b_read, acc4_arr[m]);
                }
#endif
            }
            unroll_for(int r = 0; r < TILE_SIZE; r++) {
                acc = 0.f;
                const int y = TILE_SIZE * row_id + r;
#if QK_VEC_SIZE > 1
                unroll_for(int c = 0; c < QK_VEC_SIZE; c++) {
                    acc += acc4_arr[r][c];
                }
#else
                acc += acc4_arr[r];
#endif
                P[BLK_COL_SIZE * y + x] = acc;
#ifdef MEASURE_BLOCK_4
                accum += acc;
                accum += P[BLK_COL_SIZE * y + x];
#endif
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
//         // S = matmul(Q, K) and get max value.
//         row_max = -HALF_MAX;
//         for (int c = 0; c < BLK_COL_SIZE; c++) {
//             VEC_TYPE acc4 = 0.f;
//             unroll_for (int d = 0; d < DEPTH_SIZE; d += VEC_SIZE) {
//                 acc4 = mad(*(VEC_TYPE*)(q_block + DEPTH_SIZE * row_id + d), *(VEC_TYPE*)(k_block + DEPTH_SIZE * c + d), acc4);
//             }
//             half acc = 0.f;
//             unroll_for (int i = 0; i < VEC_SIZE; i++) {
//                 acc += acc4[i];
//             }
//             P[BLK_COL_SIZE * row_id + c] = acc;
//             row_max = max(row_max , acc);
// #ifdef MEASURE_BLOCK_4
//             accum += P[BLK_COL_SIZE * row_id + c];
//             accum += row_max;
// #endif
//         }
//         m = max(p_m, row_max);
#ifdef RETURN_BLOCK_4
        continue;
#endif
        // Calculate P
        row_sum = 0.0f;
        half4 e = 0.f;
        unroll_for (int x = 0; x < BLK_COL_SIZE; x += VEC_SIZE) {
            e = exp((*(VEC_TYPE*)(P + BLK_COL_SIZE * row_id + x) - (VEC_TYPE)m));
            *(VEC_TYPE*)(P + BLK_COL_SIZE * row_id + x) = e;
            unroll_for (int i = 0; i < VEC_SIZE; i++) {
                row_sum += e[i];
            }
        }
#ifdef MEASURE_BLOCK_5
        accum += row_sum;
#endif
        barrier(CLK_LOCAL_MEM_FENCE);
#ifdef RETURN_BLOCK_5
        continue;
#endif
        // Calculate l value.
        half exp_m = exp(p_m - m);
        l = exp_m * p_l + row_sum;
        // Calculate O + PV block.
        // half acc = 0.f;
        VEC_TYPE acc4 = 0.f;
        for (int d = 0; d < DEPTH_SIZE; d++) {
            acc4 = 0.f;
            unroll_for (int c = 0; c < BLK_COL_SIZE; c += VEC_SIZE) {
                acc4 = mad(*(VEC_TYPE*)(P + BLK_COL_SIZE * row_id + c), *(VEC_TYPE*)(v_block + BLK_COL_SIZE * d + c), acc4);
            }
            acc = 0.f;
            unroll_for (int i = 0; i < VEC_SIZE; i++) {
                acc += acc4[i];
            }
            O[DEPTH_SIZE * row_id + d] = exp_m * O[DEPTH_SIZE * row_id + d] + acc;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // Set m(j-1) and l(j-1)
        p_m = m;
        p_l = l;
    }
    const int out_row_idx = BLK_ROW_SIZE * block_id + row_id;
    int oidx = OUTPUT_GET_INDEX(b, f, out_row_idx, 0);
    unroll_for (int c = 0; c < DEPTH_SIZE; c++) {
        output[oidx + c] = O[DEPTH_SIZE * row_id + c]/l;
    }
#ifdef MEASURE
    output[0] = accum;
#endif
}
