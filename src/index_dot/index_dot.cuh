#include "cute/layout.hpp"
#include "cute/pointer.hpp"
#include "cute/stride.hpp"
#include "cute/tensor_impl.hpp"
#include "cute/util/debug.hpp"

#include <cutlass/cutlass.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/util/host_tensor.h>
#include <thrust/host_vector.h>

#include <cstddef>
#include <cute/algorithm/gemm.hpp>
#include <cute/layout.hpp>
#include <cute/tensor.hpp>
using namespace std;

template <typename T> class IndexedMatrix {
    const T *   data;        // contiguous 2D data
    int         rows, cols;  // shape of the matrix
    const int * index;       // pointer to row indices
    int         indexCount;  // number of selected rows

  public:
    // Constructor
    __host__ __device__ IndexedMatrix(const T * d, int r, int c, const int * idx, int n) :
        data(d),
        rows(r),
        cols(c),
        index(idx),
        indexCount(n) {}

    // ===== Random-access Iterator =====
    class Iterator {
        const IndexedMatrix * parent;
        int                   linearPos;  // flattened position in selected rows

      public:
        using iterator_category = std::random_access_iterator_tag;
        using value_type        = T;
        using difference_type   = ptrdiff_t;
        using pointer           = const T *;
        using reference         = const T &;

        __host__ __device__ Iterator(const IndexedMatrix * p, int pos) : parent(p), linearPos(pos) {}

        // Dereference
        __host__ __device__ reference operator*() const {
            int rowIndex = parent->index[linearPos / parent->cols];
            int colIndex = linearPos % parent->cols;
            return parent->data[rowIndex * parent->cols + colIndex];
        }

        __host__ __device__ pointer operator->() const { return &(**this); }

        // Pre/post increment & decrement
        __host__ __device__ Iterator & operator++() {
            ++linearPos;
            return *this;
        }

        __host__ __device__ Iterator operator++(int) {
            Iterator tmp = *this;
            ++(*this);
            return tmp;
        }

        __host__ __device__ Iterator & operator--() {
            --linearPos;
            return *this;
        }

        __host__ __device__ Iterator operator--(int) {
            Iterator tmp = *this;
            --(*this);
            return tmp;
        }

        // Random-access
        __host__ __device__ Iterator operator+(difference_type n) const { return Iterator(parent, linearPos + n); }

        __host__ __device__ Iterator operator-(difference_type n) const { return Iterator(parent, linearPos - n); }

        __host__ __device__ difference_type operator-(const Iterator & other) const {
            return linearPos - other.linearPos;
        }

        __host__ __device__ pointer get() const { return &(**this); }

        __host__ __device__ reference operator[](difference_type n) const { return *(*this + n); }

        // Comparisons
        __host__ __device__ bool operator==(const Iterator & other) const {
            return linearPos == other.linearPos && parent == other.parent;
        }

        __host__ __device__ bool operator!=(const Iterator & other) const { return !(*this == other); }

        __host__ __device__ bool operator<(const Iterator & other) const { return linearPos < other.linearPos; }

        __host__ __device__ bool operator>(const Iterator & other) const { return linearPos > other.linearPos; }

        __host__ __device__ bool operator<=(const Iterator & other) const { return linearPos <= other.linearPos; }

        __host__ __device__ bool operator>=(const Iterator & other) const { return linearPos >= other.linearPos; }
    };

    // ===== Iterator access =====
    __host__ __device__ Iterator begin() const { return Iterator(this, 0); }

    __host__ __device__ Iterator end() const { return Iterator(this, indexCount * cols); }

    // Optional: element access by row/col
    __host__ __device__ T & operator()(int i, int j) {
        int rowIndex = index[i];
        return data[rowIndex * cols + j];
    }

    const __host__ __device__ T & operator()(int i, int j) const {
        int rowIndex = index[i];
        return data[rowIndex * cols + j];
    }

    // Shape getters
    __host__ __device__ int numRows() const { return indexCount; }

    __host__ __device__ int numCols() const { return cols; }
};

template <class ProblemShape,
          class CtaTiler,
          class TA,
          class ASmemLayout,
          class GmemTiledCopyA,
          class TB,
          class BSmemLayout,
          class TInds,
          class TiledMMA>
__global__ void idx_neg_dot_kernel(const ProblemShape   shape_BTVC,
                                   const CtaTiler       cta_tiler,
                                   TA const *           embd,
                                   const ASmemLayout    smem_layout_A,
                                   const GmemTiledCopyA gmem_tiled_copy_A,
                                   TB const *           classifier,
                                   const BSmemLayout    smem_layout_B,
                                   const TInds *        Inds,
                                   float *              indexNegDot,
                                   const TiledMMA       tiled_mma,
                                   const float          alpha = -1.0f,
                                   const float          beta  = 0.0f) {
    CUTE_STATIC_ASSERT_V(cute::rank(shape_BTVC) == cute::Int<3>{});  // (BT, V,C)
    CUTE_STATIC_ASSERT_V(cute::rank(cta_tiler) == cute::Int<2>{});   // (BLK_BBT, BLK_C)

    // CTA tiler has to be static.
    CUTE_STATIC_ASSERT_V(cute::is_static<CtaTiler>{});

    // Shared memory layouts have to be static.
    CUTE_STATIC_ASSERT_V(cute::is_static<ASmemLayout>{});
    CUTE_STATIC_ASSERT_V(cute::is_static<BSmemLayout>{});

    // Shared memory layouts have to match CTA tiler.
    CUTE_STATIC_ASSERT_V(cute::size<0>(smem_layout_A) == cute::size<0>(cta_tiler));  // BLK_BT
    CUTE_STATIC_ASSERT_V(cute::size<1>(smem_layout_A) == cute::size<1>(cta_tiler));  // BLK_C
    CUTE_STATIC_ASSERT_V(cute::size<0>(smem_layout_B) == cute::size<0>(cta_tiler));  // BLK_BT
    CUTE_STATIC_ASSERT_V(cute::size<1>(smem_layout_B) == cute::size<1>(cta_tiler));  // BLK_C

    IndexedMatrix<TB> mat(
        classifier, cute::size<1>(shape_BTVC), cute::size<2>(shape_BTVC), Inds, cute::size<0>(shape_BTVC));

    auto mA =
        cute::make_tensor(cute::make_gmem_ptr(embd), cute::select<0, 2>(shape_BTVC), cute::LayoutRight{});  // (BT, C)

    auto mB = cute::make_tensor(
        cute::make_gmem_ptr(mat.begin()), cute::select<0, 2>(shape_BTVC), cute::LayoutRight{});  // (V, C)

    auto mC = cute::make_tensor(cute::make_gmem_ptr(indexNegDot),
                                cute::make_shape(cute::select<0>(shape_BTVC), cute::_1{}),
                                cute::LayoutRight{});                                        // (BT, 1)

    auto cta_coord = cute::make_coord(blockIdx.x, cute::_);                                  // (bt, :)

    auto gA = cute::local_tile(mA, cta_tiler, cta_coord, cute::Step<cute::_1, cute::_1>{});  // (BLK_BT, BLK_K, k)

    auto gB = cute::local_tile(mB, cta_tiler, cta_coord, cute::Step<cute::_1, cute::_1>{});  // (BLK_BT, BLK_K, k)

    auto gC = cute::local_tile(mC, cta_tiler, cta_coord, cute::Step<cute::_1, cute::X>{});   // (BLK_BT, 1)

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

    auto tBgB = thread_gmem_copy_A.partition_S(gB);                // (CPY, CPY_BT, CPY_C, c)
    auto tBsB = thread_gmem_copy_A.partition_D(sB);                // (CPY, CPY_BT, CPY_C)
    auto tBrB = cute::make_fragment_like(tAsA);                    // (CPY, CPY_BT, CPY_C)

    // Partition via MMA.
    auto thread_mma = tiled_mma.get_slice(threadIdx.x);
    auto tCsA       = thread_mma.partition_A(sA);  // (MMA, MMA_BT, MMA_C)
    auto tCsB       = thread_mma.partition_A(sB);  // (MMA, MMA_BT, MMA_C)
    auto tCgC       = thread_mma.partition_C(gC);  // (MMA, MMA_BT,1)

    auto tCrA = cute::make_fragment_like(tCsA);    // (MMA, MMA_BT, MMA_C)
    auto tCrB = cute::make_fragment_like(tCsB);    // (MMA, MMA_BT, MMA_C)
    auto tCrC = cute::make_fragment_like(tCgC);

    cute::clear(tCrC);

    // BOUND CHECKING
    auto iA = cute::make_identity_tensor(cute::make_shape(cute::size<0>(sA), cute::size<1>(sA)));
    auto iC = cute::make_identity_tensor(cute::make_shape(cute::size<0>(gC), cute::size<1>(gC)));

    auto tAiA = thread_gmem_copy_A.partition_S(iA);  // (CPY, CPY_BT, CPY_C)
    auto tCiC = thread_mma.partition_C(iC);          // (MMA, MMA_M, MMA_N)
    // Create predicate tensors.
    auto tApA = cute::make_tensor<bool>(cute::make_shape(cute::size<1>(tAiA), cute::size<2>(tAiA)),
                                        cute::make_stride(cute::Int<1>{}, cute::Int<0>{}));
    auto tCpC = cute::make_tensor<bool>(cute::shape(tCiC));

    CUTE_UNROLL
    for (auto bt = 0; bt < cute::size<0>(tApA); ++bt) {
        tApA(bt, 0) = cute::get<0>(tAiA(0, bt, 0)) + blockIdx.x * cute::size<0>(sA) < cute::size<0>(shape_BTVC);
        tCpC(bt)    = cute::get<0>(tCiC(bt)) + blockIdx.x * cute::size<0>(gC) < cute::size<0>(shape_BTVC);
    }

    cute::clear(tArA);
    cute::clear(tBrB);
    // Need predicates for bounds checking.
    CUTE_UNROLL
    for (auto copy_k_idx = 0; copy_k_idx < cute::size<2>(tAiA); ++copy_k_idx) {
        if (cute::get<1>(tAiA(0, 0, copy_k_idx)) + 0 * cute::size<1>(sA) < cute::size<2>(shape_BTVC)) {
            cute::copy_if(tApA, tAgA(cute::_, cute::_, copy_k_idx, 0), tArA(cute::_, cute::_, copy_k_idx));
            cute::copy_if(tApA, tBgB(cute::_, cute::_, copy_k_idx, 0), tBrB(cute::_, cute::_, copy_k_idx));
        }
    }

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

                cute::clear(tArA);
                cute::clear(tBrB);
                // Need predicates for bounds checking.
                CUTE_UNROLL
                for (auto copy_k_idx = 0; copy_k_idx < cute::size<2>(tAiA); ++copy_k_idx) {
                    // Check the K dimension.
                    if (cute::get<1>(tAiA(0, 0, copy_k_idx)) + tile_idx_k_next * cute::size<1>(sA) <
                        cute::size<2>(shape_BTVC)) {
                        cute::copy_if(tApA,
                                      tAgA(cute::_, cute::_, copy_k_idx, tile_idx_k_next),
                                      tArA(cute::_, cute::_, copy_k_idx));

                        cute::copy_if(tApA,
                                      tBgB(cute::_, cute::_, copy_k_idx, tile_idx_k_next),
                                      tBrB(cute::_, cute::_, copy_k_idx));
                    }
                }
            }

            for (int v = 0; v < cute::size<0>(tCrA); v++) {
                for (int bt = 0; bt < cute::size<1>(tCrA); bt++) {
                    tCrC(v, bt, mma_idx_k) =
                        fmaf(tCrA(v, bt, mma_idx_k), tCrB(v, bt, mma_idx_k), tCrC(v, bt, mma_idx_k));
                }
            }
        }
    }

    // cute::copy_if(tCpC, tCrC, tCgC);
    cute::axpby(alpha, tCrC, beta, tCgC);
    // cute::axpby(alpha, tCrC, beta, tCgC, tCpC);
}

template <typename Element>
void idx_neg_dot(float *         indexNegDot,
                 const Element * embd,
                 const Element * classifier,
                 const int *     Inds,
                 const int       BT,
                 const int       V,
                 const int       C) {
    auto       BT_V_C_SHAPE = cute::make_shape(BT, V, C);
    // Define CTA size.
    const auto bBT          = cute::Int<128>{};
    const auto bC           = cute::Int<32>{};
    //   const cute::Int<128 * 2 / sizeof(Element)> bBT;
    //   const cute::Int<32> bC;
    const auto cta_tiler    = cute::make_shape(bBT, bC);  // (BLK_BT, BLK_C)

    // Define smem layouts.
    // smem_layout_A is (BLK_BT, BLK_C) row-major.
    // smem_layout_B is (BLK_BT, BLK_C) row-major.
    // smem_layout_C is (BLK_BT) row-major.
    const auto smem_shape_A  = cute::make_shape(bBT, bC);                             // (BLK_BT, BLK_C)
    const auto smem_layout_A = cute::make_layout(smem_shape_A, cute::LayoutRight{});  // (BLK_BT, BLK_C)
    const auto smem_shape_B  = cute::make_shape(bBT, bC);                             // (BLK_BT, BLK_C)
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

    cute::print(gmem_tiled_copy_A);

    auto tiled_mma = cute::make_tiled_mma(cute::UniversalFMA<Element, Element, float>{}, thread_layout_C);
    cute::print(tiled_mma);
    const dim3 block_dims = static_cast<unsigned int>(cute::size(thread_layout_C));
    const dim3 grid_dims  = static_cast<unsigned int>(cute::size(cute::ceil_div(BT, bBT)));

    idx_neg_dot_kernel<<<grid_dims, block_dims>>>(BT_V_C_SHAPE,
                                                  cta_tiler,
                                                  embd,
                                                  smem_layout_A,
                                                  gmem_tiled_copy_A,
                                                  classifier,
                                                  smem_layout_B,
                                                  Inds,
                                                  indexNegDot,
                                                  tiled_mma);
    CUTE_CHECK_LAST();
}
