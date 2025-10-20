#include "index_dot.cuh"

#include <cutlass/layout/matrix.h>
#include <cutlass/util/host_tensor.h>

#include <cute/tensor.hpp>

int main() {
    cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 1024 * 1024 * 1000);
    using Element = float;
    int BT        = 128;
    int C         = 128;
    int V         = 128;

    cutlass::HostTensor<Element, cutlass::layout::RowMajor> embd({ BT, C });
    cutlass::HostTensor<Element, cutlass::layout::RowMajor> classifier({ V, C });

    cutlass::HostTensor<Element, cutlass::layout::RowMajor> indexNegDot({ BT, cute::_1{} });
    cutlass::HostTensor<long, cutlass::layout::RowMajor>    Inds({ BT, cute::_1{} });

    for (int i = 0; i < BT; i++) {
        for (int j = 0; j < C; j++) {
            embd.at({ i, j }) = 0.00001 * ((i * C) + j);
        }
        Inds.at({ i, cute::_0{} }) = i;  //(BT - 1) - i;
    }

    for (int i = 0; i < V; i++) {
        for (int j = 0; j < C; j++) {
            classifier.at({ i, j }) = 0.00001 * ((i * C) + j);
        }
    }

    embd.sync_device();
    classifier.sync_device();
    Inds.sync_device();
    indexNegDot.sync_device();
    idx_neg_dot(indexNegDot.device_data(), embd.device_data(), classifier.device_data(), Inds.device_data(), BT, V, C);
    cudaDeviceSynchronize();
    embd.sync_host();
    classifier.sync_host();
    Inds.sync_host();
    indexNegDot.sync_host();

    for (int i = 0; i < BT; i++) {
        std::cout << indexNegDot.at({ i, 0 }) << std::endl;
    }

    // for (int i = 0; i < BT; i++) {
    //     for (int j = 0; j < C; j++) {
    //         std::cout << embd.at({ i, j }) << " ";
    //     }
    //     std::cout << "\n ";
    // }
}
