#include "cute/arch/copy_sm75.hpp"
#include "cute/layout.hpp"
#include "cute/numeric/numeric_types.hpp"
#include "cute/pointer.hpp"
#include "cute/stride.hpp"
#include "cute/tensor_impl.hpp"
#include "cute/underscore.hpp"
#include "cute/util/debug.hpp"
#include "cutlass/util/host_tensor.h"
#include "utils.cuh"

#include <complex.h>
#include <cutlass/cutlass.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/util/host_tensor.h>
#include <thrust/host_vector.h>

#include <cmath>
#include <cute/algorithm/gemm.hpp>
#include <cute/layout.hpp>
#include <cute/tensor.hpp>

__device__ void atomicCasUpdate(float * address, float val) {
    int * addr_i = (int *) address;
    int   old_i  = *addr_i;
    int   assumed_i;

    do {
        assumed_i       = old_i;
        float assumed_f = __int_as_float(assumed_i);

        float new_f = val + assumed_f;

        int new_i = __float_as_int(new_f);
        old_i     = atomicCAS(addr_i, assumed_i, new_i);
    } while (old_i != assumed_i);
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
          class T_DLSE,
          class CStride,
          class CSmemLayout,
          class CThreadLayout,
          class TiledMMA,
          class T_DA,
          class T_DB,
          class T_INDS>
__global__ void cce_backward_kernel(ProblemShape shape_BTVC,
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
                                    T_LSE const *  lse,
                                    T_DLSE const   dlse,
                                    CStride        stride_C,
                                    CSmemLayout    smem_layout_C,
                                    CThreadLayout,
                                    TiledMMA tiled_mma,
                                    T_DA *   d_embd,
                                    T_DB *   d_classifier,
                                    T_INDS * Inds) {
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

    auto mdA =
        cute::make_tensor(cute::make_gmem_ptr(d_embd), cute::select<0, 2>(shape_BTVC), cute::LayoutRight{});  // (BT, C)

    auto mB = cute::make_tensor(
        cute::make_gmem_ptr(classifier), cute::select<1, 2>(shape_BTVC), cute::LayoutRight{});  // (V, C)

    auto mdB = cute::make_tensor(
        cute::make_gmem_ptr(d_classifier), cute::select<1, 2>(shape_BTVC), cute::LayoutRight{});  // (V, C)

    auto mLSE = cute::make_tensor(
        cute::make_gmem_ptr(lse), cute::make_shape(cute::select<0>(shape_BTVC), cute::_1{}), cute::LayoutRight{});

    auto mInds = cute::make_tensor(
        cute::make_gmem_ptr(Inds), cute::make_shape(cute::select<0>(shape_BTVC), cute::_1{}), cute::LayoutRight{});

    auto cta_coord = cute::make_coord(blockIdx.x, blockIdx.y, cute::_);  // (bt, :)

    auto gA =
        cute::local_tile(mA, cta_tiler, cta_coord, cute::Step<cute::_1, cute::X, cute::_1>{});  // (BLK_BT, BLK_K, k)

    auto gdA =
        cute::local_tile(mdA, cta_tiler, cta_coord, cute::Step<cute::_1, cute::X, cute::_1>{});  // (BLK_BT, BLK_K, k)

    auto gB =
        cute::local_tile(mB, cta_tiler, cta_coord, cute::Step<cute::X, cute::_1, cute::_1>{});  // (BLK_V, BLK_K, k)

    auto gdB =
        cute::local_tile(mdB, cta_tiler, cta_coord, cute::Step<cute::X, cute::_1, cute::_1>{});  // (BLK_V, BLK_K, k)

    auto gLSE  = cute::local_tile(mLSE, cta_tiler, cta_coord, cute::Step<cute::_1, cute::X, cute::X>{});
    auto gInds = cute::local_tile(mInds, cta_tiler, cta_coord, cute::Step<cute::_1, cute::X, cute::X>{});

    // Shared memory buffers.
    __shared__ TA smemA[cute::cosize_v<ASmemLayout>];
    __shared__ TB smemB[cute::cosize_v<BSmemLayout>];

    auto sA = cute::make_tensor(cute::make_smem_ptr(smemA), smem_layout_A);  // (BLK_BT, BLK_C)
    auto sB = cute::make_tensor(cute::make_smem_ptr(smemB), smem_layout_B);  // (BLK_BT, BLK_C)

    // Partition via tiled copy.
    auto thread_gmem_copy_A = gmem_tiled_copy_A.get_slice(threadIdx.x);
    auto tAgA               = thread_gmem_copy_A.partition_S(gA);  // (CPY, CPY_BT, CPY_C, c)
    // auto tdAgdA             = thread_gmem_copy_A.partition_S(gdA);
    auto tAsA               = thread_gmem_copy_A.partition_D(sA);  // (CPY, CPY_BT, CPY_C)
    auto tArA               = cute::make_fragment_like(tAsA);      // (CPY, CPY_BT, CPY_C)

    auto thread_gmem_copy_B = gmem_tiled_copy_B.get_slice(threadIdx.x);
    auto tBgB               = thread_gmem_copy_B.partition_S(gB);  // (CPY, CPY_BT, CPY_C, c)
    // auto tdBgdB             = thread_gmem_copy_B.partition_S(gdB);  // (CPY, CPY_BT, CPY_C, c)
    auto tBsB               = thread_gmem_copy_B.partition_D(sB);  // (CPY, CPY_BT, CPY_C)
    auto tBrB               = cute::make_fragment_like(tBsB);      // (CPY, CPY_BT, CPY_C)

    // Partition via MMA.
    auto thread_mma = tiled_mma.get_slice(threadIdx.x);

    auto tCsA           = thread_mma.partition_A(sA);   // (MMA, MMA_BT, MMA_C)
    auto tCsB           = thread_mma.partition_B(sB);   // (MMA, MMA_BT, MMA_C)
    auto tdAgdA         = thread_mma.partition_A(gdA);
    auto tdBgdB         = thread_mma.partition_B(gdB);  // (CPY, CPY_BT, CPY_C, c)
    auto tLSErLSE       = thread_mma.partition_C(gLSE);
    auto tIndsrInds     = thread_mma.partition_C(gInds);
    auto tCrC_col_major = cute::partition_fragment_C(tiled_mma, cute::select<0, 1>(cta_tiler));
    auto tCrC           = cute::make_tensor(tCrC_col_major.data(), tCrC_col_major.shape(), cute::LayoutRight{});

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

            cute::gemm(tiled_mma, tCrA(cute::_, cute::_, mma_idx_k_next), tCrB(cute::_, cute::_, mma_idx_k_next), tCrC);
        }
    }

    // for (auto btv = 0; btv < cute::size(tCpC); ++btv) {
    for (int cpy = 0; cpy < cute::size<0>(tCpC); cpy++) {
        for (int i = 0; i < cute::size<1>(tCpC); i++) {
            for (int j = 0; j < cute::size<2>(tCpC); j++) {
                // auto logit_Idx = tCiC(cpy, i, j);
                // auto bt        = cute::get<0>(logit_Idx);
                // auto gBT       = bt + (blockIdx.x * cute::get<0>(cta_tiler));
                // auto v         = cute::get<1>(logit_Idx);
                auto gV = cute::get<1>(tCiC(cpy, i, j)) + (blockIdx.y * cute::get<1>(cta_tiler));

                // Compute expf(output[bt][v] - lse[bt])
                tCrC(cpy, i, j) = expf(tCrC(cpy, i, j) - tLSErLSE(cpy, i, 0));

                if (gV == tIndsrInds(cpy, i, 0)) {
                    tCrC(cpy, i, j) -= 1.0f;
                }

                // Multiply by dlse
                tCrC(cpy, i, j) *= dlse;
            }
        }
    }

    cute::copy(gmem_tiled_copy_A, tAgA(cute::_, cute::_, cute::_, 0), tArA);
    cute::copy(gmem_tiled_copy_B, tBgB(cute::_, cute::_, cute::_, 0), tBrB);

    cute::copy(tBrB, tBsB);
    // Synchronize to ensure the data on shared memory is ready for mma.
    __syncthreads();

    cute::clear(tCrA);

    auto tCrB_shape = tCrB.shape();
    auto tCrB_new   = cute::make_tensor(
        tCrB.data(),
        cute::make_shape(cute::get<0>(tCrB_shape), cute::get<2>(tCrB_shape), cute::get<1>(tCrB_shape)),
        cute::LayoutLeft{});

    // Prepare the registers for the first mma iteration.
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
                cute::copy(tBrB, tBsB);
                // Ensure the data for the mma of the new tile iteration is
                // ready.
                __syncthreads();
            }

            // Copy data from shared memory to register for the next mma
            // iteration.
            const auto mma_idx_k_next = (mma_idx_k + 1) % num_mmas_per_tile_k;
            // cute::copy(tCsA(cute::_, cute::_, mma_idx_k_next), tCrA(cute::_, cute::_, mma_idx_k_next));
            cute::copy(tCsB(cute::_, cute::_, mma_idx_k_next), tCrB(cute::_, cute::_, mma_idx_k_next));
            // Before the first mma iteration of each tile iteration, copy data
            // from global memory to register for the next tile iteration.
            if (mma_idx_k == 0) {
                const auto tile_idx_k_next = (tile_idx_k + 1) % num_tiles_k;

                // cute::copy(gmem_tiled_copy_A, tAgA(cute::_, cute::_, cute::_, tile_idx_k_next), tArA);
                cute::copy(gmem_tiled_copy_B, tBgB(cute::_, cute::_, cute::_, tile_idx_k_next), tBrB);
                // cute::clear(tArA);
                // cute::clear(tBrB);
                // // Need predicates for bounds checking.
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

            cute::gemm(tiled_mma, tCrC, tCrB_new, tCrA);

            for (int i = 0; i < cute::size<1>(tCrA); i++) {
                atomicCasUpdate(&tdAgdA(0, i, mma_idx_k, tile_idx_k), tCrA(0, i, mma_idx_k));
            }

            cute::clear(tCrA);
        }
    }
    cute::copy(gmem_tiled_copy_A, tAgA(cute::_, cute::_, cute::_, 0), tArA);

    cute::copy(tArA, tAsA);
    // Synchronize to ensure the data on shared memory is ready for mma.
    __syncthreads();

    cute::clear(tCrA);
    cute::clear(tCrB);

    auto tCrC_shape = tCrC.shape();
    auto tCrC_new   = cute::make_tensor(
        tCrC.data(),
        cute::make_shape(cute::get<0>(tCrC_shape), cute::get<2>(tCrC_shape), cute::get<1>(tCrC_shape)),
        cute::LayoutLeft{});

    auto tCrA_shape = tCrA.shape();
    auto tCrA_new   = cute::make_tensor(
        tCrA.data(),
        cute::make_shape(cute::get<0>(tCrA_shape), cute::get<2>(tCrA_shape), cute::get<1>(tCrA_shape)),
        cute::LayoutLeft{});

    // Prepare the registers for the first mma iteration.
    cute::copy(tCsA(cute::_, cute::_, 0), tCrA(cute::_, cute::_, 0));
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
                // Ensure the data for the mma of the new tile iteration is
                // ready.
                __syncthreads();
            }

            // Copy data from shared memory to register for the next mma
            // iteration.
            const auto mma_idx_k_next = (mma_idx_k + 1) % num_mmas_per_tile_k;
            cute::copy(tCsA(cute::_, cute::_, mma_idx_k_next), tCrA(cute::_, cute::_, mma_idx_k_next));
            // cute::copy(tCsB(cute::_, cute::_, mma_idx_k_next), tCrB(cute::_, cute::_, mma_idx_k_next));
            // Before the first mma iteration of each tile iteration, copy data
            // from global memory to register for the next tile iteration.
            if (mma_idx_k == 0) {
                const auto tile_idx_k_next = (tile_idx_k + 1) % num_tiles_k;

                cute::copy(gmem_tiled_copy_A, tAgA(cute::_, cute::_, cute::_, tile_idx_k_next), tArA);
                // cute::copy(gmem_tiled_copy_B, tBgB(cute::_, cute::_, cute::_, tile_idx_k_next), tBrB);
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
            }

            cute::gemm(tiled_mma, tCrC_new, tCrA_new, tCrB);

            for (int i = 0; i < cute::size<1>(tCrB); i++) {
                atomicCasUpdate(&tdBgdB(0, i, mma_idx_k, tile_idx_k), tCrB(0, i, mma_idx_k));
            }
            cute::clear(tCrB);
        }
    }
    // if (cute::thread0()) {
    //     // cute::print("tArA\n");
    //     // cute::print(tArA);
    //     // cute::print("\n");

    //     // cute::print("tBrB\n");
    //     // cute::print(tBrB);
    //     // cute::print("\n");

    //     // cute::print("tSum\n");
    //     // cute::print_tensor(tSum);
    //     // cute::print("\n");

    //     // cute::print("tMax\n");
    //     // cute::print_tensor(tMax);
    //     // cute::print("\n");

    //     // cute::print("tCgC_created\n");
    //     // cute::print_tensor(tCgC_created);
    //     // cute::print("\n");

    //     // cute::print("tCgC\n");
    //     // cute::print_tensor(tCgC);
    //     // cute::print("\n");

    //     cute::print("tCrA\n");
    //     cute::print_tensor(tCrA);
    //     cute::print("\n");

    //     cute::print("tCrB\n");
    //     cute::print_tensor(tCrB);
    //     cute::print("\n");

    //     cute::print("tCrC\n");
    //     cute::print_tensor(tCrC(cute::_0{}, cute::_, cute::_));
    //     cute::print("\n");

    //     cute::print("tCiC\n");
    //     cute::print_tensor(tCiC(cute::_0{}, cute::_, cute::_));
    //     cute::print("\n");

    //     cute::print("tIndsrInds\n");
    //     cute::print_tensor(tIndsrInds);
    //     cute::print("\n");

    //     cute::print("tCpC\n");
    //     cute::print_tensor(tCpC);
    //     cute::print("\n");
    //     // cute::print("localSum\n");
    //     // cute::print_tensor(localSum);
    //     // cute::print("\n");

    //     // cute::print("localMax\n");
    //     // cute::print_tensor(localMax);
    //     // cute::print("\n");

    //     // cute::print("sB\n");
    //     // cute::print_tensor(sB);
    //     // cute::print("\n");
    // }
}

template <typename Element, class DLSE, class LSE, class T_EMBD, class T_CLASS, class T_INDS, class T_SHAPE>
void cce_backward(DLSE &    dlse,
                  LSE &     lse,
                  T_EMBD &  embd,
                  T_CLASS & classifier,
                  T_EMBD &  d_embd,
                  T_CLASS & d_classifier,
                  T_INDS &  Inds,
                  T_SHAPE & BT_V_C_SHAPE) {
    // Define CTA size.
    const auto bBT = cute::Int<64>{};
    const auto bV  = cute::Int<64>{};
    const auto bC  = cute::Int<32>{};

    const auto cta_tiler = cute::make_shape(bBT, bV, bC);  // (BLK_BT, BLK_C)

    // Define smem layouts.
    // smem_layout_A is (BLK_BT, BLK_C) row-major.
    // smem_layout_B is (BLK_BT, BLK_C) row-major.
    // smem_layout_C is (BLK_BT) row-major.
    const auto smem_shape_A  = cute::make_shape(bBT, bC);                             // (BLK_BT, BLK_C)
    const auto smem_layout_A = cute::make_layout(smem_shape_A, cute::LayoutRight{});  // (BLK_BT, BLK_C)
    const auto smem_shape_B  = cute::make_shape(bV, bC);                              // (BLK_BT, BLK_C)
    const auto smem_layout_B = cute::make_layout(smem_shape_B, cute::LayoutRight{});  // (BLK_V, BLK_C)

    const auto smem_shape_C  = cute::make_shape(bBT, bV);                             // (BLK_BT,1)
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

    cce_backward_kernel<Element><<<grid_dims, block_dims>>>(BT_V_C_SHAPE,
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
                                                            dlse,
                                                            lse.stride(0),
                                                            smem_layout_C,
                                                            thread_layout_C,
                                                            tiled_mma,
                                                            d_embd.device_data(),
                                                            d_classifier.device_data(),
                                                            Inds.device_data());
    CUTE_CHECK_LAST();
}

int main() {
    cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 1024 * 1024 * 1000);
    using Element = float;
    int BT        = 128;
    int C         = 128;
    int V         = 128;

    auto BT_V_C_SHAPE = cute::make_shape(BT, V, C);

    cutlass::HostTensor<Element, cutlass::layout::RowMajor> embd(
        { cute::size<0>(BT_V_C_SHAPE), cute::size<2>(BT_V_C_SHAPE) }),
        d_embd({ cute::size<0>(BT_V_C_SHAPE), cute::size<2>(BT_V_C_SHAPE) });

    cutlass::HostTensor<Element, cutlass::layout::RowMajor> classifier(
        { cute::size<1>(BT_V_C_SHAPE), cute::size<2>(BT_V_C_SHAPE) }),
        d_classifier({ cute::size<1>(BT_V_C_SHAPE), cute::size<2>(BT_V_C_SHAPE) });

    cutlass::HostTensor<Element, cutlass::layout::RowMajor> lse({ cute::size<0>(BT_V_C_SHAPE), cute::_1{} });
    // dlse({ cute::size<0>(BT_V_C_SHAPE), cute::_1{} });

    cutlass::HostTensor<int, cutlass::layout::RowMajor> Inds({ cute::size<0>(BT_V_C_SHAPE), cute::_1{} });

    float dlse = 1.0f / ((float) cute::size<0>(BT_V_C_SHAPE));
    for (int i = 0; i < cute::size<0>(BT_V_C_SHAPE); i++) {
        for (int j = 0; j < cute::size<2>(BT_V_C_SHAPE); j++) {
            embd.at({ i, j })   = 0.00001 * ((i * C) + j);
            d_embd.at({ i, j }) = 0.0f;
        }
        // dlse.at({ i, 0 }) = 1.0f / ((float) cute::size<0>(BT_V_C_SHAPE));
        Inds.at({ i, 0 }) = i;
    }

    for (int i = 0; i < cute::size<1>(BT_V_C_SHAPE); i++) {
        for (int j = 0; j < cute::size<2>(BT_V_C_SHAPE); j++) {
            classifier.at({ i, j })   = 0.00001 * ((i * C) + j);
            d_classifier.at({ i, j }) = 0.0f;
        }
    }
    fill_ptr("lse.txt", lse.host_data(), cute::size<0>(BT_V_C_SHAPE));
    embd.sync_device();
    classifier.sync_device();
    lse.sync_device();
    Inds.sync_device();
    d_embd.sync_device();
    d_classifier.sync_device();

    // for (int i = 0; i < 1024; i++) {
    cce_backward<Element>(dlse, lse, embd, classifier, d_embd, d_classifier, Inds, BT_V_C_SHAPE);
    // }

    embd.sync_host();
    classifier.sync_host();
    lse.sync_host();
    d_embd.sync_host();
    d_classifier.sync_host();

    std::cout << "d_embd: " << std::endl;
    for (int i = 0; i < cute::size<0>(BT_V_C_SHAPE); i++) {
        std::cout << "Index i:" << i << "->: ";
        for (int j = 0; j < cute::size<1>(BT_V_C_SHAPE); j++) {
            std::cout << std::setprecision(10) << d_embd.at({ i, j }) << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "d_classifier: " << std::endl;
    for (int i = 0; i < cute::size<1>(BT_V_C_SHAPE); i++) {
        std::cout << "Index i:" << i << "->: ";
        for (int j = 0; j < cute::size<1>(BT_V_C_SHAPE); j++) {
            std::cout << std::setprecision(10) << d_classifier.at({ i, j }) << " ";
        }
        std::cout << std::endl;
    }
}
