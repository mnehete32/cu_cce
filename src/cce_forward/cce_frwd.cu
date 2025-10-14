#include "cute/arch/copy_sm75.hpp"
#include "cute/layout.hpp"
#include "cute/numeric/numeric_types.hpp"
#include "cute/pointer.hpp"
#include "cute/stride.hpp"
#include "cute/tensor_impl.hpp"
#include "cute/underscore.hpp"
#include "cute/util/debug.hpp"
#include "cutlass/fast_math.h"
#include "cutlass/functional.h"
#include "cutlass/util/device_utils.h"
#include "cutlass/util/host_tensor.h"

#include <complex.h>
#include <cutlass/cutlass.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/util/host_tensor.h>
#include <thrust/host_vector.h>

#include <cmath>
#include <cstddef>
#include <cute/algorithm/gemm.hpp>
#include <cute/layout.hpp>
#include <cute/tensor.hpp>

__device__ void atomicLogAddExp(float * address, float val) {
    unsigned int * addr_u = (unsigned int *) address;
    unsigned int   old_u  = *addr_u;
    unsigned int   assumed_u;

    do {
        assumed_u       = old_u;
        float assumed_f = __uint_as_float(assumed_u);

        float new_f;
        if (assumed_f == -INFINITY) {
            new_f = val;
        } else {
            float m = fmaxf(assumed_f, val);
            // log(e^a + e^b) = m + log( e^(a-m) + e^(b-m) )
            // stable compute of log(1 + exp(-|diff|))
            new_f   = m + log1pf(expf(-fabsf(val - assumed_f)));
        }

        unsigned int new_u = __float_as_uint(new_f);
        old_u              = atomicCAS(addr_u, assumed_u, new_u);
    } while (old_u != assumed_u);
}

template <class Element,
          class ProblemShape,
          class CtaTiler,
          class TA,
          class AStride,
          class ASmemLayout,
          class AThreadLayout,
          class GmemTiledCopyA,
          class TB,
          class BStride,
          class BSmemLayout,
          class BThreadLayout,
          class GmemTiledCopyB,
          class T_LSE,
          class CStride,
          class CSmemLayout,
          class CThreadLayout,
          class TiledMMA>
__global__ void cce_frwd_kernel(ProblemShape shape_BTVC,
                                CtaTiler     cta_tiler,
                                TA const *   embd,
                                AStride      stride_A,
                                ASmemLayout  smem_layout_A,
                                AThreadLayout,
                                GmemTiledCopyA gmem_tiled_copy_A,
                                TB const *     classifier,
                                BStride        stride_B,
                                BSmemLayout    smem_layout_B,
                                BThreadLayout,
                                GmemTiledCopyB gmem_tiled_copy_B,
                                T_LSE *        lse,
                                CStride        stride_C,
                                CSmemLayout    smem_layout_C,
                                CThreadLayout,
                                TiledMMA tiled_mma) {
    CUTE_STATIC_ASSERT_V(cute::rank(shape_BTVC) == cute::Int<3>{});  // (BT, V,C)
    CUTE_STATIC_ASSERT_V(cute::rank(cta_tiler) == cute::Int<3>{});   // (BLK_BT,BLK_V, BLK_C)

    // CTA tiler has to be static.
    CUTE_STATIC_ASSERT_V(cute::is_static<CtaTiler>{});

    // Shared memory layouts have to be static.
    CUTE_STATIC_ASSERT_V(cute::is_static<ASmemLayout>{});
    CUTE_STATIC_ASSERT_V(cute::is_static<BSmemLayout>{});
    CUTE_STATIC_ASSERT_V(cute::is_static<CSmemLayout>{});

    // Shared memory layouts have to match CTA tiler.
    CUTE_STATIC_ASSERT_V(cute::size<0>(smem_layout_A) == cute::size<0>(cta_tiler));  // BLK_BT
    CUTE_STATIC_ASSERT_V(cute::size<1>(smem_layout_A) == cute::size<2>(cta_tiler));  // BLK_C
    CUTE_STATIC_ASSERT_V(cute::size<0>(smem_layout_B) == cute::size<1>(cta_tiler));  // BLK_BT
    CUTE_STATIC_ASSERT_V(cute::size<1>(smem_layout_B) == cute::size<2>(cta_tiler));  // BLK_C

    auto mA =
        cute::make_tensor(cute::make_gmem_ptr(embd), cute::select<0, 2>(shape_BTVC), cute::LayoutRight{});  // (BT, C)

    auto mB = cute::make_tensor(
        cute::make_gmem_ptr(classifier), cute::select<1, 2>(shape_BTVC), cute::LayoutRight{});  // (V, C)
    auto mLSE = cute::make_tensor(
        cute::make_gmem_ptr(lse), cute::make_shape(cute::select<0>(shape_BTVC), cute::_1{}), cute::LayoutRight{});

    auto cta_coord = cute::make_coord(blockIdx.x, blockIdx.y, cute::_);  // (bt, :)

    auto gA =
        cute::local_tile(mA, cta_tiler, cta_coord, cute::Step<cute::_1, cute::X, cute::_1>{});  // (BLK_BT, BLK_K, k)

    auto gB =
        cute::local_tile(mB, cta_tiler, cta_coord, cute::Step<cute::X, cute::_1, cute::_1>{});  // (BLK_V, BLK_K, k)

    auto gLSE = cute::local_tile(mLSE, cta_tiler, cta_coord, cute::Step<cute::_1, cute::X, cute::X>{});
    // auto gC =
    //     cute::local_tile(mC, cta_tiler, cta_coord, cute::Step<cute::_1, cute::_1, cute::X>{});  // (BLK_V, BLK_K, k)
    // auto gC = cute::make_tensor<float>(cute::shape(cute::select<0, 1>(cta_tiler)), cute::LayoutRight{});

    // Shared memory buffers.
    __shared__ TA smemA[cute::cosize_v<ASmemLayout>];
    __shared__ TB smemB[cute::cosize_v<BSmemLayout>];

    auto sA = cute::make_tensor(cute::make_smem_ptr(smemA), smem_layout_A);  // (BLK_BT, BLK_C)
    auto sB = cute::make_tensor(cute::make_smem_ptr(smemB), smem_layout_B);  // (BLK_BT, BLK_C)

    // Partition via tiled copy.
    auto thread_gmem_copy_A = gmem_tiled_copy_A.get_slice(threadIdx.x);
    auto tAgA               = thread_gmem_copy_A.partition_S(gA);  // (CPY, CPY_BT, CPY_C, c)
    auto tAsA               = thread_gmem_copy_A.partition_D(sA);  // (CPY, CPY_BT, CPY_C)
    auto tArA               = cute::make_fragment_like(tAsA);      // (CPY, CPY_BT, CPY_C)

    auto thread_gmem_copy_B = gmem_tiled_copy_B.get_slice(threadIdx.x);
    auto tBgB               = thread_gmem_copy_B.partition_S(gB);  // (CPY, CPY_BT, CPY_C, c)
    auto tBsB               = thread_gmem_copy_B.partition_D(sB);  // (CPY, CPY_BT, CPY_C)
    auto tBrB               = cute::make_fragment_like(tBsB);      // (CPY, CPY_BT, CPY_C)

    // Partition via MMA.
    auto thread_mma = tiled_mma.get_slice(threadIdx.x);

    auto tCsA     = thread_mma.partition_A(sA);  // (MMA, MMA_BT, MMA_C)
    auto tCsB     = thread_mma.partition_B(sB);  // (MMA, MMA_BT, MMA_C)
    auto tLSErLSE = thread_mma.partition_C(gLSE);
    auto tCrC     = cute::partition_fragment_C(tiled_mma, cute::select<0, 1>(cta_tiler));

    auto tCrA = cute::make_fragment_like(tCsA);  // (MMA, MMA_BT, MMA_C)
    auto tCrB = cute::make_fragment_like(tCsB);  // (MMA, MMA_BT, MMA_C)
    cute::clear(tCrC);

    // BOUND CHECKING
    auto iA = cute::make_identity_tensor(cute::make_shape(cute::size<0>(sA), cute::size<1>(sA)));
    auto iB = cute::make_identity_tensor(cute::make_shape(cute::size<0>(sB), cute::size<1>(sB)));
    auto iC = cute::make_identity_tensor(cute::select<0, 1>(cta_tiler));

    auto tAiA = thread_gmem_copy_A.partition_S(iA);
    auto tBiB = thread_gmem_copy_B.partition_S(iB);
    auto tCiC = thread_mma.partition_C(iC);

    auto tApA = cute::make_tensor<bool>(cute::make_shape(cute::size<1>(tAiA), cute::size<2>(tAiA)),
                                        cute::make_stride(cute::Int<1>{}, cute::Int<0>{}));
    auto tBpB = cute::make_tensor<bool>(cute::make_shape(cute::size<1>(tBiB), cute::size<2>(tBiB)),
                                        cute::make_stride(cute::Int<1>{}, cute::Int<0>{}));

    auto tCpC = cute::make_tensor<bool>(cute::shape(tCiC));
    CUTE_UNROLL
    for (auto bt = 0; bt < cute::size<0>(tApA); ++bt) {
        tApA(bt, 0) = cute::get<0>(tAiA(0, bt, 0)) + blockIdx.x * cute::size<0>(sA) < cute::size<0>(shape_BTVC);
    }
    CUTE_UNROLL
    for (auto v = 0; v < cute::size<0>(tBpB); ++v) {
        tBpB(v, 0) = cute::get<0>(tBiB(0, v, 0)) + blockIdx.y * cute::size<0>(sB) < cute::size<1>(shape_BTVC);
    }

    CUTE_UNROLL
    for (auto btv = 0; btv < cute::size(tCpC); ++btv) {
        tCpC(btv) = cute::get<0>(tCiC(btv)) + blockIdx.x * cute::get<0>(cta_tiler) < cute::size<0>(shape_BTVC) &&
                    cute::get<1>(tCiC(btv)) + blockIdx.y * cute::get<1>(cta_tiler) < cute::size<1>(shape_BTVC);
    }

    cute::copy(gmem_tiled_copy_A, tAgA(cute::_, cute::_, cute::_, 0), tArA);
    cute::copy(gmem_tiled_copy_B, tBgB(cute::_, cute::_, cute::_, 0), tBrB);

    // cute::clear(tArA);
    // cute::clear(tBrB);

    // CUTE_UNROLL
    // for (auto copy_k_idx = 0; copy_k_idx < cute::size<2>(tAiA); ++copy_k_idx) {
    //     if (cute::get<1>(tAiA(0, 0, copy_k_idx)) + 0 * cute::size<1>(sA) < cute::size<2>(shape_BTVC)) {
    //         cute::copy_if(tApA, tAgA(cute::_, cute::_, copy_k_idx, 0), tArA(cute::_, cute::_, copy_k_idx));
    //     }
    // }

    // CUTE_UNROLL
    // for (auto copy_k_idx = 0; copy_k_idx < cute::size<2>(tAiA); ++copy_k_idx) {
    //     if (cute::get<1>(tBiB(0, 0, copy_k_idx)) + 0 * cute::size<1>(sB) < cute::size<2>(shape_BTVC)) {
    //         cute::copy_if(tBpB, tBgB(cute::_, cute::_, copy_k_idx, 0), tBrB(cute::_, cute::_, copy_k_idx));
    //     }
    // }
    cute::copy(tArA, tAsA);
    cute::copy(tBrB, tBsB);
    // Synchronize to ensure the data on shared memory is ready for mma.
    __syncthreads();

    // Perform the gemm computation loop.
    const auto     num_tiles_k         = cute::size<2>(gA);    // k
    constexpr auto num_mmas_per_tile_k = cute::size<2>(tCrA);  // MMA_K

    // Prepare the registers for the first mma iteration.
    cute::copy(tCsA(cute::_, cute::_, 0), tCrA(cute::_, cute::_, 0));
    cute::copy(tCsB(cute::_, cute::_, 0), tCrB(cute::_, cute::_, 0));

    for (auto tile_idx_k = 0; tile_idx_k < num_tiles_k; ++tile_idx_k) {
        CUTE_UNROLL
        for (auto mma_idx_k = 0; mma_idx_k < num_mmas_per_tile_k; ++mma_idx_k) {
            // Before the last mma iteration of each tile iteration, copy data
            // from register to shared memory for the next tile iteration.
            if (mma_idx_k == num_mmas_per_tile_k - 1) {
                // Ensure the data copy from global memory to register is
                // completed.
                __syncthreads();
                // Copy data from register to shared memory for the next tile
                // iteration.
                cute::copy(tArA, tAsA);
                cute::copy(tBrB, tBsB);
                // Ensure the data for the mma of the new tile iteration is
                // ready.
                __syncthreads();
            }

            // Copy data from shared memory to register for the next mma
            // iteration.
            const auto mma_idx_k_next = (mma_idx_k + 1) % num_mmas_per_tile_k;
            cute::copy(tCsA(cute::_, cute::_, mma_idx_k_next), tCrA(cute::_, cute::_, mma_idx_k_next));
            cute::copy(tCsB(cute::_, cute::_, mma_idx_k_next), tCrB(cute::_, cute::_, mma_idx_k_next));
            // Before the first mma iteration of each tile iteration, copy data
            // from global memory to register for the next tile iteration.
            if (mma_idx_k == 0) {
                const auto tile_idx_k_next = (tile_idx_k + 1) % num_tiles_k;

                cute::copy(gmem_tiled_copy_A, tAgA(cute::_, cute::_, cute::_, tile_idx_k_next), tArA);
                cute::copy(gmem_tiled_copy_B, tBgB(cute::_, cute::_, cute::_, tile_idx_k_next), tBrB);
                // cute::clear(tArA);
                // cute::clear(tBrB);
                // // Need predicates for bounds checking.
                // CUTE_UNROLL
                // for (auto copy_k_idx = 0; copy_k_idx < cute::size<2>(tAiA); ++copy_k_idx) {
                //     // Check the K dimension.
                //     if (cute::get<1>(tAiA(0, 0, copy_k_idx)) + tile_idx_k_next * cute::size<1>(sA) <
                //         cute::size<2>(shape_BTVC)) {
                //         cute::copy_if(gmem_tiled_copy_A,
                //                       tApA,
                //                       tAgA(cute::_, cute::_, copy_k_idx, tile_idx_k_next),
                //                       tArA(cute::_, cute::_, copy_k_idx));
                //     }
                // }
                // CUTE_UNROLL
                // for (auto copy_k_idx = 0; copy_k_idx < cute::size<2>(tBiB); ++copy_k_idx) {
                //     if (cute::get<1>(tBiB(0, 0, copy_k_idx)) + tile_idx_k_next * cute::size<1>(sB) <
                //         cute::size<2>(shape_BTVC)) {
                //         cute::copy_if(gmem_tiled_copy_B,
                //                       tBpB,
                //                       tBgB(cute::_, cute::_, copy_k_idx, tile_idx_k_next),
                //                       tBrB(cute::_, cute::_, copy_k_idx));
                //     }
                // }
            }
            // if (cute::thread0()) {
            //     cute::print("tCrB\n");
            //     cute::print_tensor(tCrB);
            //     cute::print("\n");
            // }
            cute::gemm(tiled_mma, tCrA(cute::_, cute::_, mma_idx_k_next), tCrB(cute::_, cute::_, mma_idx_k_next), tCrC);
        }
    }

    // auto max_sum_layout = cute::make_layout(cute::shape(cute::size<0>(cta_tiler)), cute::LayoutRight{});

    __shared__ float smem_sum[cute::size<0>(cta_tiler)];
    __shared__ float smem_max[cute::size<0>(cta_tiler)];

    auto sSum = cute::make_tensor(
        cute::make_smem_ptr(smem_sum), cute::make_shape(cute::size<0>(cta_tiler), cute::_1{}), cute::LayoutRight{});

    auto sMax = cute::make_tensor(
        cute::make_smem_ptr(smem_max), cute::make_shape(cute::size<0>(cta_tiler), cute::_1{}), cute::LayoutRight{});

    auto tSum = thread_mma.partition_C(sSum);
    auto tMax = thread_mma.partition_C(sMax);

    auto localSum = cute::make_fragment_like(tSum);
    auto localMax = cute::make_fragment_like(tMax);

    for (int bt = 0; bt < cute::size<1>(tCiC); bt++) {
        localMax(bt) = -INFINITY;
        localSum(bt) = 0.0f;

        if (cute::get<1>(tCiC(0, bt, 0)) == 0) {
            cute::fill(tSum, 0.0f);
            cute::fill(tMax, -INFINITY);
            // tSum(i) = 0.0f;
            // tMax(i) = -INFINITY;
        }

        __syncthreads();
        auto bt_local = tCrC(cute::_, bt, cute::_);

        for (int j = 0; j < cute::size(bt_local); j++) {
            localMax(bt) = fmaxf(bt_local(j), localMax(bt));
        }

        cutlass::atomic_maximum<float>{}(&tMax(bt), localMax(bt));
        __syncthreads();

        for (int j = 0; j < cute::size(bt_local); j++) {
            localSum(bt) += expf(bt_local(j) - tMax(bt));
        }
        cutlass::atomic_add<float>{}(&tSum(bt), localSum(bt));
        __syncthreads();
        auto partial_lse = tMax(bt) + logf(tSum(bt));
        if (cute::get<1>(tCiC(0, bt, 0)) == 0) {
            atomicLogAddExp(&tLSErLSE(0, bt, 0), partial_lse);
        }
    }

    // cute::copy(tCrC, tCgC);
    // cute::copy_if(tCpC, tCrC, tCgC);
    // if (cute::thread0()) {
    //     cute::print("tArA\n");
    //     cute::print(tArA);
    //     cute::print("\n");

    //     cute::print("tBrB\n");
    //     cute::print(tBrB);
    //     cute::print("\n");

    //     cute::print("tSum\n");
    //     cute::print_tensor(tSum);
    //     cute::print("\n");

    //     cute::print("tMax\n");
    //     cute::print_tensor(tMax);
    //     cute::print("\n");

    //     cute::print("tCiC\n");
    //     cute::print_tensor(tCiC);
    //     cute::print("\n");

    //     // cute::print("tCgC_created\n");
    //     // cute::print_tensor(tCgC_created);
    //     // cute::print("\n");

    //     // cute::print("tCgC\n");
    //     // cute::print_tensor(tCgC);
    //     // cute::print("\n");

    //     cute::print("tCrC\n");
    //     cute::print_tensor(tCrC);
    //     cute::print("\n");

    //     cute::print("localSum\n");
    //     cute::print_tensor(localSum);
    //     cute::print("\n");

    //     cute::print("localMax\n");
    //     cute::print_tensor(localMax);
    //     cute::print("\n");

    //     cute::print("tLSErLSE\n");
    //     cute::print_tensor(tLSErLSE);
    //     cute::print("\n");
    // }
}

template <typename Element, class LSE, class T_EMBD, class T_CLASS, class T_SHAPE>
void cce_fwd(LSE & lse, T_EMBD & embd, T_CLASS & classifier, T_SHAPE & BT_V_C_SHAPE) {
    // Define CTA size.
    const auto bBT       = cute::Int<128>{};
    const auto bV        = cute::Int<128>{};
    const auto bC        = cute::Int<32>{};
    //   const cute::Int<128 * 2 / sizeof(Element)> bBT;
    //   const cute::Int<32> bC;
    const auto cta_tiler = cute::make_shape(bBT, bV, bC);  // (BLK_BT, BLK_C)

    // Define smem layouts.
    // smem_layout_A is (BLK_BT, BLK_C) row-major.
    // smem_layout_B is (BLK_BT, BLK_C) row-major.
    // smem_layout_C is (BLK_BT) row-major.
    const auto smem_shape_A  = cute::make_shape(bBT, bC);                             // (BLK_BT, BLK_C)
    const auto smem_layout_A = cute::make_layout(smem_shape_A, cute::LayoutRight{});  // (BLK_BT, BLK_C)
    const auto smem_shape_B  = cute::make_shape(bV, bC);                              // (BLK_BT, BLK_C)
    const auto smem_layout_B = cute::make_layout(smem_shape_B, cute::LayoutRight{});  // (BLK_V, BLK_C)

    const auto smem_shape_C  = cute::make_shape(bBT, bBT);                            // (BLK_BT,1)
    const auto smem_layout_C = cute::make_layout(smem_shape_C, cute::LayoutRight{});  // (BLK_BT, 1)

    // Define thread layouts.
    const auto THR_BT = cute::Int<16>{};

    const auto thread_shape_A = cute::make_shape(THR_BT, cute::Int<8>{});  // (THR_BT, THR_C)
    const auto thread_shape_B = cute::make_shape(THR_BT, cute::Int<8>{});  // (THR_BT, THR_C)
    const auto thread_shape_C = cute::make_shape(THR_BT, cute::Int<8>{});

    const auto thread_layout_A = cute::make_layout(thread_shape_A, cute::LayoutRight{});  // (THR_BT, THR_C)
    const auto thread_layout_B = cute::make_layout(thread_shape_B, cute::LayoutRight{});  // (THR_BT, THR_C)
    const auto thread_layout_C = cute::make_layout(thread_shape_C, cute::LayoutRight{});  // (THR_BT, THR_C)

    CUTE_STATIC_ASSERT_V(cute::size(thread_layout_A) == cute::size(thread_layout_B));

    // CTA tiler has to be divisible by the thread layouts.
    CUTE_STATIC_ASSERT_V(cute::size<0>(cta_tiler) % cute::size<0>(thread_layout_A) ==
                         cute::Int<0>{});  // BLK_BT % THR_BT == 0
    CUTE_STATIC_ASSERT_V(cute::size<1>(cta_tiler) % cute::size<1>(thread_layout_A) ==
                         cute::Int<0>{});  // BLK_C % THR_C == 0
    CUTE_STATIC_ASSERT_V(cute::size<0>(cta_tiler) % cute::size<0>(thread_layout_B) ==
                         cute::Int<0>{});  // BLK_BT % THR_BT == 0
    CUTE_STATIC_ASSERT_V(cute::size<1>(cta_tiler) % cute::size<1>(thread_layout_B) ==
                         cute::Int<0>{});  // BLK_C % THR_C == 0
    CUTE_STATIC_ASSERT_V(cute::size<1>(cta_tiler) % cute::size<0>(thread_layout_C) ==
                         cute::Int<0>{});  // BLK_BT % THR_BT == 0

    // Shared memory layouts have to be divisible by the thread layouts.
    CUTE_STATIC_ASSERT_V(cute::size<0>(smem_layout_A) % cute::size<0>(thread_layout_A) ==
                         cute::Int<0>{});  // BLK_BT % THR_BT == 0
    CUTE_STATIC_ASSERT_V(cute::size<1>(smem_layout_A) % cute::size<1>(thread_layout_A) ==
                         cute::Int<0>{});  // BLK_C % THR_C == 0
    CUTE_STATIC_ASSERT_V(cute::size<0>(smem_layout_B) % cute::size<0>(thread_layout_B) ==
                         cute::Int<0>{});  // BLK_BT % THR_BT == 0
    CUTE_STATIC_ASSERT_V(cute::size<1>(smem_layout_B) % cute::size<1>(thread_layout_B) ==
                         cute::Int<0>{});  // BLK_C % THR_C == 0

    constexpr int num_vec_elems = sizeof(cute::uint128_t) / sizeof(Element);

    auto gmem_tiled_copy_A = cute::make_tiled_copy(
        cute::Copy_Atom<cute::UniversalCopy<cute::uint128_t>, Element>{},
        thread_layout_A,
        cute::make_layout(cute::make_shape(cute::_1{}, cute::Int<num_vec_elems>{}), cute::LayoutRight{}));

    auto gmem_tiled_copy_B = cute::make_tiled_copy(
        cute::Copy_Atom<cute::UniversalCopy<cute::uint128_t>, Element>{},
        thread_layout_B,
        cute::make_layout(cute::make_shape(cute::_1{}, cute::Int<num_vec_elems>{}), cute::LayoutRight{}));
    // cute::print(gmem_tiled_copy_A);

    auto       tiled_mma  = cute::make_tiled_mma(cute::UniversalFMA<Element, Element, float>{}, thread_layout_C);
    // cute::print(tiled_mma);
    const dim3 block_dims = static_cast<unsigned int>(cute::size(thread_layout_C));
    const dim3 grid_dims  = { static_cast<unsigned int>(cute::size(cute::ceil_div(cute::get<0>(BT_V_C_SHAPE), bBT))),
                              static_cast<unsigned int>(cute::size(cute::ceil_div(cute::get<1>(BT_V_C_SHAPE), bV))) };

    cce_frwd_kernel<Element><<<grid_dims, block_dims>>>(BT_V_C_SHAPE,
                                                        cta_tiler,
                                                        embd.device_data(),
                                                        embd.stride(0),
                                                        smem_layout_A,
                                                        thread_layout_A,
                                                        gmem_tiled_copy_A,
                                                        classifier.device_data(),
                                                        classifier.stride(0),
                                                        smem_layout_B,
                                                        thread_layout_B,
                                                        gmem_tiled_copy_B,
                                                        lse.device_data(),
                                                        lse.stride(0),
                                                        smem_layout_C,
                                                        thread_layout_C,
                                                        tiled_mma);
    CUTE_CHECK_LAST();
}

int main() {
    cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 1024 * 1024 * 1000);
    using Element = float;
    int BT        = 2048;
    int C         = 2048;
    int V         = 2048;

    auto BT_V_C_SHAPE = cute::make_shape(BT, V, C);

    cutlass::HostTensor<Element, cutlass::layout::RowMajor> embd(
        { cute::size<0>(BT_V_C_SHAPE), cute::size<2>(BT_V_C_SHAPE) });
    cutlass::HostTensor<Element, cutlass::layout::RowMajor> classifier(
        { cute::size<1>(BT_V_C_SHAPE), cute::size<2>(BT_V_C_SHAPE) });
    cutlass::HostTensor<Element, cutlass::layout::RowMajor> lse({ cute::size<0>(BT_V_C_SHAPE), cute::_1{} });

    for (int i = 0; i < cute::size<0>(BT_V_C_SHAPE); i++) {
        for (int j = 0; j < cute::size<2>(BT_V_C_SHAPE); j++) {
            embd.at({ i, j }) = 0.00001 * ((i * C) + j);
        }

        lse.at({ i, 0 }) = -INFINITY;
    }

    for (int i = 0; i < cute::size<1>(BT_V_C_SHAPE); i++) {
        for (int j = 0; j < cute::size<2>(BT_V_C_SHAPE); j++) {
            classifier.at({ i, j }) = 0.00001 * ((i * C) + j);
        }
    }

    embd.sync_device();
    classifier.sync_device();
    lse.sync_device();
    // for (int i = 0; i < 1024; i++) {
    cce_fwd<Element>(lse, embd, classifier, BT_V_C_SHAPE);
    // }

    embd.sync_host();
    classifier.sync_host();
    lse.sync_host();

    for (int i = 0; i < cute::size<0>(BT_V_C_SHAPE); i++) {
        std::cout << "Index i:" << i << "->: ";
        std::cout << std::setprecision(10) << lse.at({ i, 0 }) << std::endl;
    }
}
