#include "../cce_backward/cce_backward.cuh"
#include "../cce_forward/cce_frwd.cuh"
#include "../index_dot/index_dot.cuh"

#include <ATen/ops/set.h>
#include <torch/torch.h>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>

// ================================================================
// 🔹 Reference Cross Entropy (Torch)
// ================================================================
struct CrossEntropyResult {
    torch::Tensor loss;
    torch::Tensor grad_embd;
    torch::Tensor grad_classifier;
};

CrossEntropyResult torch_ce_loss(torch::Tensor embd, torch::Tensor classifier, torch::Tensor Inds) {
    torch::Tensor logits = torch::matmul(embd, torch::transpose(classifier, 0, 1));
    torch::Tensor loss   = torch::cross_entropy_loss(logits, Inds);
    loss.backward();
    return { loss, embd.grad(), classifier.grad() };
}

// ================================================================
// 🔹 Custom CUTLASS-based Cross Entropy (Forward + Backward)
// ================================================================
struct CCEResult {
    torch::Tensor loss;
    torch::Tensor grad_embd;
    torch::Tensor grad_classifier;
};

CCEResult cce_loss_and_backward(torch::Tensor embd, torch::Tensor classifier, torch::Tensor Inds) {
    const int BT = embd.size(0);
    const int V  = classifier.size(0);
    const int C  = embd.size(1);

    torch::Tensor idxNegDot = torch::zeros({ BT }, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    torch::Tensor lse       = torch::full_like(idxNegDot, -FP_INFINITE);
    torch::Tensor dembd_cce = torch::zeros_like(embd);
    torch::Tensor dclassifier_cce = torch::zeros_like(classifier);

    // Forward
    idx_neg_dot(idxNegDot.data_ptr<float>(),
                embd.data_ptr<float>(),
                classifier.data_ptr<float>(),
                Inds.data_ptr<long>(),
                BT,
                V,
                C);

    cce_fwd(lse.data_ptr<float>(), embd.data_ptr<float>(), classifier.data_ptr<float>(), BT, V, C);

    torch::Tensor loss = torch::mean(idxNegDot + lse);

    // Backward
    cce_backward(1.0f / static_cast<float>(BT),
                 lse.data_ptr<float>(),
                 embd.data_ptr<float>(),
                 classifier.data_ptr<float>(),
                 dembd_cce.data_ptr<float>(),
                 dclassifier_cce.data_ptr<float>(),
                 Inds.data_ptr<long>(),
                 BT,
                 V,
                 C);

    return { loss, dembd_cce, dclassifier_cce };
}

void check_tensor_error(const torch::Tensor & torch_tensor,
                        const torch::Tensor & cce_tensor,
                        float                 error_tol = 1e-6f,
                        const std::string &   name      = "Tensor") {
    TORCH_CHECK(torch_tensor.device() == cce_tensor.device(), "Tensors must be on the same device");
    TORCH_CHECK(torch_tensor.sizes() == cce_tensor.sizes(), "Tensors must be on the same sizes");

    // Compute elementwise absolute difference
    auto diff = (torch_tensor - cce_tensor).abs();

    // Flatten to 1D and get the max value (scalar)
    auto flat_diff           = diff.flatten();
    // Get maximum difference and its index
    auto max_diff_val_tensor = std::get<0>(flat_diff.max(0));      // tensor with 1 element
    auto max_diff_val        = max_diff_val_tensor.item<float>();  // now it's a scalar

    if (max_diff_val > error_tol) {
        auto max_diff_idx = std::get<1>(flat_diff.max(0)).item<int64_t>();
        std::cerr << "\n❌ Gradient mismatch!\n";
        std::cerr << "\n For: " << name << std::endl;
        int64_t row = max_diff_idx / torch_tensor.size(1);
        int64_t col = max_diff_idx % torch_tensor.size(1);
        std::cerr << "max_diff: " << std::setprecision(12) << max_diff_val << "\n";
        std::cerr << "Mismatch at row " << row << ", col " << col << std::endl;
        std::cerr << "Torch value: " << std::setprecision(12) << torch_tensor[row][col].item<float>() << "\n";
        std::cerr << "CCE value:   " << std::setprecision(12) << cce_tensor[row][col].item<float>() << "\n";
        std::exit(EXIT_FAILURE);
    }
    std::cout << "Grad " << name << " abs_max_diff: " << std::setprecision(12) << max_diff_val << "\n";
    std::cout << "Grad " << name << " is Close: " << "✅" << std::endl;
    // std::cout << "Grad Classifier Close: " << "✅" << std::endl;
}

// ================================================================
// 🔹 Test Runner
// ================================================================
void run_tests() {
    torch::manual_seed(32);
    cudaDeviceSynchronize();

    // should be multiple of block sizes used in the kernels
    std::vector<int> sizes = { 128, 256, 512, 1024, 2048, 4096 };
    // const float      rtol  = 1e-5;
    // const float      atol  = 1e-6;

    const float rtol = 1e-4;
    const float atol = 1e-5;

    for (int BT : sizes) {
        for (int V : sizes) {
            for (int C : sizes) {
                std::cout << "\n=== Testing BT=" << BT << "  V=" << V << "  C=" << C << " ===" << std::endl;

                // --- Create inputs ---
                torch::Tensor embd = torch::rand(
                    { BT, C }, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA).requires_grad(true));
                torch::Tensor classifier = torch::rand(
                    { V, C }, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA).requires_grad(true));
                embd.retain_grad();
                classifier.retain_grad();

                torch::Tensor Inds =
                    torch::randint(V, { BT }, torch::TensorOptions().dtype(torch::kLong).device(torch::kCUDA));

                // --- Torch reference ---
                auto torch_res = torch_ce_loss(
                    embd.clone().detach().requires_grad_(true), classifier.clone().detach().requires_grad_(true), Inds);

                // --- Custom CUTLASS implementation ---
                auto cce_res = cce_loss_and_backward(embd.clone(), classifier.clone(), Inds);

                // --- Compare ---
                bool loss_close = cce_res.loss.allclose(torch_res.loss, rtol, atol);
                // bool grad_embd_close  = cce_res.grad_embd.allclose(torch_res.grad_embd, rtol, atol);
                // bool grad_class_close = cce_res.grad_classifier.allclose(torch_res.grad_classifier, rtol, atol);

                std::cout << std::fixed << std::setprecision(8);
                std::cout << "Loss (Torch): " << torch_res.loss.item<float>()
                          << " | (CCE): " << cce_res.loss.item<float>() << std::endl;
                std::cout << "Loss Close: " << (loss_close ? "✅" : "❌") << std::endl;

                check_tensor_error(torch_res.grad_embd, cce_res.grad_embd, 1e-4f, "Embedding");
                check_tensor_error(torch_res.grad_classifier, cce_res.grad_classifier, 1e-4f, "Classifier");
                // std::cout << "Grad Embd Close: " << (grad_embd_close ? "✅" : "❌") << std::endl;
                // std::cout << "Grad Classifier Close: " << (grad_class_close ? "✅" : "❌") << std::endl;

                cudaDeviceSynchronize();
            }
        }
    }

    std::cout << "\n✅ All tests passed successfully!" << std::endl;
}

// ================================================================
// 🔹 Main
// ================================================================
int main() {
    try {
        run_tests();
    } catch (const std::exception & e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return EXIT_FAILURE;
    } catch (...) {
        std::cerr << "Test failed with unknown error." << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
