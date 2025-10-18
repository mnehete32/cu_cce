#include "cce_frwd.cuh"

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
    cutlass::HostTensor<Element, cutlass::layout::RowMajor> lse({ BT, cute::_1{} });

    for (int i = 0; i < BT; i++) {
        for (int j = 0; j < C; j++) {
            embd.at({ i, j }) = 0.00001 * ((i * C) + j);
        }

        lse.at({ i, 0 }) = -INFINITY;
    }

    for (int i = 0; i < V; i++) {
        for (int j = 0; j < C; j++) {
            classifier.at({ i, j }) = 0.00001 * ((i * C) + j);
        }
    }

    embd.sync_device();
    classifier.sync_device();
    lse.sync_device();
    // for (int i = 0; i < 1024; i++) {
    cce_fwd(lse.device_data(), embd.device_data(), classifier.device_data(), BT, V, C);
    // }

    embd.sync_host();
    classifier.sync_host();
    lse.sync_host();

    for (int i = 0; i < BT; i++) {
        // std::cout << "Index i:" << i << "->: ";
        std::cout << std::setprecision(10) << lse.at({ i, 0 }) << std::endl;
    }
}
