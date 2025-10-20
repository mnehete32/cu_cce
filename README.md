# CU_CCE — Cut Your Losses in Large-Vocabulary Language Models

This repository provides a **CUTLASS-CUTE based implementation** of the paper **[Cut Your Losses in Large-Vocabulary Language Models](https://arxiv.org/abs/2411.09009)**. The goal is to reimplement the proposed *"loss-cutting"* technique using NVIDIA’s CUTLASS/CUTE CUDA library, to avoid ever materializing the large `(B × T, V)` logits tensor in GPU memory.

> “The cross-entropy loss is responsible for up to **90% of the memory footprint** of modern LLM training.”  
> — *Wijmans et al., 2025*

This implementation explores a low-level CUTLASS-CUTE approach (as opposed to the original Triton kernel version)  
to understand and optimize memory and compute efficiency at the CUDA kernel level.


# Environment:
1. CUDA 12.6
2. cmake 4.1.1
3. gcc-12

# Getting Started:
1. Clone and Set Up Dependencies

```bash
# Clone this repository
git clone https://github.com/mnehete32/cu_cce.git
cd cu_cce

# Clone CUTLASS (v4.2.0)
git clone --depth 1 --branch v4.2.0 https://github.com/NVIDIA/cutlass.git

# Download and extract LibTorch (PyTorch C++ API)
wget https://download.pytorch.org/libtorch/cu126/libtorch-shared-with-deps-2.8.0%2Bcu126.zip
unzip libtorch-shared-with-deps-2.8.0+cu126.zip
```


2. Build project
```bash
cmake -B build
cmake --build build --config Release
```

3. Run the Cross-Entropy Comparison Test
To compare the CUTLASS-based implementation against the PyTorch reference
```bash
./build/src/cce_test/cce_test
```


# Notes 

- Tested with CUDA 12.6, CUTLASS v4.2.0, and LibTorch 2.8.0+cu126.
- This codebase focuses on kernel-level design and numerical validation.
- The comparison test validates both loss and gradient correctness versus PyTorch reference results.


# Citation
If reference this work or build on the original idea, please cite the paper:
```
@inproceedings{wijmans2025cut,
  author       = {Erik Wijmans and
                  Brody Huval and
                  Alexander Hertzberg and
                  Vladlen Koltun and
                  Philipp Kr\"ahenb\"uhl},
  title        = {Cut Your Losses in Large-Vocabulary Language Models},
  booktitle    = {International Conference on Learning Representations},
  year         = {2025},
}
```