
#include "cce_backward.cuh"

#include <cutlass/layout/matrix.h>
#include <cutlass/util/host_tensor.h>

#include <cute/tensor.hpp>

int main() {
    cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 1024 * 1024 * 1000);
    using Element = float;
    int BT        = 128;
    int C         = 128;
    int V         = 128;

    auto BT_V_C_SHAPE = cute::make_shape(BT, V, C);

    cutlass::HostTensor<Element, cutlass::layout::RowMajor> embd({ BT, C }), d_embd({ BT, C });

    cutlass::HostTensor<Element, cutlass::layout::RowMajor> classifier({ V, C }), d_classifier({ V, C });

    cutlass::HostTensor<Element, cutlass::layout::RowMajor> lse({ BT, cute::_1{} });
    // dlse({ BT, cute::_1{} });

    cutlass::HostTensor<long, cutlass::layout::RowMajor> Inds({ BT, cute::_1{} });

    float dlse = 1.0f / ((float) BT);
    for (int i = 0; i < BT; i++) {
        for (int j = 0; j < C; j++) {
            embd.at({ i, j })   = 0.00001 * ((i * C) + j);
            d_embd.at({ i, j }) = 0.0f;
        }
        // dlse.at({ i, 0 }) = 1.0f / ((float) BT);
        Inds.at({ i, 0 }) = i;
    }

    for (int i = 0; i < V; i++) {
        for (int j = 0; j < C; j++) {
            classifier.at({ i, j })   = 0.00001 * ((i * C) + j);
            d_classifier.at({ i, j }) = 0.0f;
        }
    }
    fill_ptr("lse.txt", lse.host_data(), BT);
    embd.sync_device();
    classifier.sync_device();
    lse.sync_device();
    Inds.sync_device();
    d_embd.sync_device();
    d_classifier.sync_device();

    // for (int i = 0; i < 1024; i++) {
    cce_backward(dlse,
                 lse.device_data(),
                 embd.device_data(),
                 classifier.device_data(),
                 d_embd.device_data(),
                 d_classifier.device_data(),
                 Inds.device_data(),
                 BT,
                 V,
                 C);
    // }

    embd.sync_host();
    classifier.sync_host();
    lse.sync_host();
    d_embd.sync_host();
    d_classifier.sync_host();

    std::cout << "d_embd: " << std::endl;
    for (int i = 0; i < BT; i++) {
        std::cout << "Index i:" << i << "->: ";
        for (int j = 0; j < V; j++) {
            std::cout << std::setprecision(10) << d_embd.at({ i, j }) << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "d_classifier: " << std::endl;
    for (int i = 0; i < V; i++) {
        std::cout << "Index i:" << i << "->: ";
        for (int j = 0; j < V; j++) {
            std::cout << std::setprecision(10) << d_classifier.at({ i, j }) << " ";
        }
        std::cout << std::endl;
    }
}
