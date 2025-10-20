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

template <class ProblemShape,
          class CtaTiler,
          class TA,
          class ASmemLayout,
          class GmemTiledCopyA,
          class TB,
          class BSmemLayout,
          class GmemTiledCopyB,
          class T_LSE,
          class T_DLSE,
          class TiledMMA,
          class T_DA,
          class T_DB,
          class T_INDS>
__global__ void cce_backward_kernel(const ProblemShape   shape_BTVC,
                                    const CtaTiler       cta_tiler,
                                    TA const *           embd,
                                    const ASmemLayout    smem_layout_A,
                                    const GmemTiledCopyA gmem_tiled_copy_A,
                                    TB const *           classifier,
                                    const BSmemLayout    smem_layout_B,
                                    const GmemTiledCopyB gmem_tiled_copy_B,
                                    T_LSE const *        lse,
                                    T_DLSE const         dlse,
                                    const TiledMMA       tiled_mma,
                                    T_DA *               d_embd,
                                    T_DB *               d_classifier,
                                    const T_INDS *       Inds) {
    CUTE_STATIC_ASSERT_V(cute::rank(shape_BTVC) == cute::Int<3>{});  // (BT, V,C)
    CUTE_STATIC_ASSERT_V(cute::rank(cta_tiler) == cute::Int<3>{});   // (BLK_BT,BLK_V, BLK_C)

    // CTA tiler has to be static.
    CUTE_STATIC_ASSERT_V(cute::is_static<CtaTiler>{});

    // Shared memory layouts have to be static.
    CUTE_STATIC_ASSERT_V(cute::is_static<ASmemLayout>{});
    CUTE_STATIC_ASSERT_V(cute::is_static<BSmemLayout>{});

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
    auto tCrB_T     = cute::make_tensor(
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

            cute::gemm(tiled_mma, tCrC, tCrB_T, tCrA);

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
    auto tCrC_T     = cute::make_tensor(
        tCrC.data(),
        cute::make_shape(cute::get<0>(tCrC_shape), cute::get<2>(tCrC_shape), cute::get<1>(tCrC_shape)),
        cute::LayoutLeft{});

    auto tCrA_shape = tCrA.shape();
    auto tCrA_T     = cute::make_tensor(
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

            cute::gemm(tiled_mma, tCrC_T, tCrA_T, tCrB);

            for (int i = 0; i < cute::size<1>(tCrB); i++) {
                atomicCasUpdate(&tdBgdB(0, i, mma_idx_k, tile_idx_k), tCrB(0, i, mma_idx_k));
            }
            cute::clear(tCrB);
        }
    }
}

template <typename Element>
void cce_backward(const float     dlse,
                  const float *   lse,
                  const Element * embd,
                  const Element * classifier,
                  Element *       d_embd,
                  Element *       d_classifier,
                  const long *    Inds,
                  const int       BT,
                  const int       V,
                  const int       C) {
    auto       BT_V_C_SHAPE = cute::make_shape(BT, V, C);
    // Define CTA size.
    const auto bBT          = cute::Int<64>{};
    const auto bV           = cute::Int<64>{};
    const auto bC           = cute::Int<32>{};

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

    cce_backward_kernel<<<grid_dims, block_dims>>>(BT_V_C_SHAPE,
                                                   cta_tiler,
                                                   embd,
                                                   smem_layout_A,
                                                   gmem_tiled_copy_A,
                                                   classifier,
                                                   smem_layout_B,
                                                   gmem_tiled_copy_B,
                                                   lse,
                                                   dlse,
                                                   tiled_mma,
                                                   d_embd,
                                                   d_classifier,
                                                   Inds);
    CUTE_CHECK_LAST();
}
