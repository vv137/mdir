// Standalone Schur-update microbenchmark, NOT a complete constraint solver.
// Supplied with tc_constraint_solver_reference.zip; compiled and exercised
// on an RTX 3090 during the MDIR constraint experiment (see README.md).
// Uses cuBLAS's documented TF32-enabled compute mode, with a strict FP32 control.
// A profiling run is still needed to verify which hardware instructions cuBLAS
// selects for a particular shape/device/library version.
//
// Build (CUDA 11+; target NVIDIA compute capability >= 8.0):
// nvcc -O3 -std=c++17 -gencode arch=compute_80,code='[sm_80,compute_80]' \
//      schur_update.cu -lcublas -o schur_update
// Run: ./schur_update [separator_size=32] [interior_size=64] [batch=1024]
//
// Column-major blocks: F[s,p], T[p,s], C[s,s]. The update is C <- C - F*T.
// T must be formed by solving D*T=E, not by explicitly inverting D.
// Library reference: https://docs.nvidia.com/cuda/cublas/index.html

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

static void cuda_check(cudaError_t e) {
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
static void blas_check(cublasStatus_t e) {
    if (e != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("cuBLAS error status " + std::to_string(int(e)));
}
struct DeviceBuffer {
    float* ptr = nullptr;
    explicit DeviceBuffer(size_t n) {
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&ptr), n*sizeof(float)));
    }
    ~DeviceBuffer() { if (ptr) cudaFree(ptr); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};
struct BlasHandle {
    cublasHandle_t value = nullptr;
    BlasHandle() { blas_check(cublasCreate(&value)); }
    ~BlasHandle() { if (value) cublasDestroy(value); }
};
struct Event {
    cudaEvent_t value = nullptr;
    Event() { cuda_check(cudaEventCreate(&value)); }
    ~Event() { if (value) cudaEventDestroy(value); }
};

static void schur_update(cublasHandle_t handle, const float* f, const float* t,
                         float* c, int s, int p, int batches,
                         cublasComputeType_t compute) {
    if (!f || !t || !c || s<=0 || p<=0 || batches<=0)
        throw std::invalid_argument("invalid Schur-update input");
    const float alpha = -1.0f, beta = 1.0f;
    const long long stride_ft = static_cast<long long>(s)*p;
    const long long stride_c = static_cast<long long>(s)*s;
    blas_check(cublasGemmStridedBatchedEx(
        handle, CUBLAS_OP_N, CUBLAS_OP_N,
        s, s, p,
        &alpha,
        f, CUDA_R_32F, s, stride_ft,
        t, CUDA_R_32F, p, stride_ft,
        &beta,
        c, CUDA_R_32F, s, stride_c,
        batches, compute, CUBLAS_GEMM_DEFAULT));
}

int main(int argc, char** argv) {
    try {
        const int s = argc>1 ? std::stoi(argv[1]) : 32;
        const int p = argc>2 ? std::stoi(argv[2]) : 64;
        const int batch = argc>3 ? std::stoi(argv[3]) : 1024;
        if (s<1 || s>1024 || p<1 || p>4096 || batch<1 || batch>1000000)
            throw std::invalid_argument("require 1<=s<=1024, 1<=p<=4096, 1<=batch<=1000000");
        int device = 0;
        cuda_check(cudaGetDevice(&device));
        cudaDeviceProp props{};
        cuda_check(cudaGetDeviceProperties(&props, device));
        if (props.major < 8)
            throw std::runtime_error("TF32 requires NVIDIA compute capability >= 8.0");
        const size_t nft = size_t(s)*size_t(p)*size_t(batch);
        const size_t nc = size_t(s)*size_t(s)*size_t(batch);
        size_t free_bytes=0, total_bytes=0;
        cuda_check(cudaMemGetInfo(&free_bytes, &total_bytes));
        if ((2*nft+nc)*sizeof(float) > free_bytes*0.75)
            throw std::runtime_error("requested buffers exceed 75% of free GPU memory");
        std::vector<float> f(nft), t(nft), c(nc), out(nc);
        std::mt19937 rng(17);
        std::uniform_real_distribution<float> uniform(-0.2f, 0.2f);
        for (auto& x:f) x=uniform(rng);
        for (auto& x:t) x=uniform(rng);
        for (auto& x:c) x=uniform(rng);
        for (int b=0; b<batch; ++b)
            for (int i=0; i<s; ++i) c[size_t(b)*s*s + i + size_t(i)*s] += 2.0f;
        DeviceBuffer df(nft), dt(nft), dc(nc);
        cuda_check(cudaMemcpy(df.ptr, f.data(), nft*sizeof(float), cudaMemcpyHostToDevice));
        cuda_check(cudaMemcpy(dt.ptr, t.data(), nft*sizeof(float), cudaMemcpyHostToDevice));
        BlasHandle handle;
        int version=0;
        blas_check(cublasGetVersion(handle.value, &version));
        std::cout << "Device: " << props.name << "; cuBLAS: " << version
                  << "; s=" << s << ", p=" << p << ", batch=" << batch << '\n';
        std::cout << "Only the product/update is timed; no factorization, packing, "
                     "residuals, or MD integration.\n";
        for (int variant=0; variant<2; ++variant) {
            const auto compute = variant==0 ? CUBLAS_COMPUTE_32F_PEDANTIC
                                            : CUBLAS_COMPUTE_32F_FAST_TF32;
            cuda_check(cudaMemcpy(dc.ptr, c.data(), nc*sizeof(float), cudaMemcpyHostToDevice));
            schur_update(handle.value, df.ptr, dt.ptr, dc.ptr, s, p, batch, compute);
            cuda_check(cudaMemcpy(out.data(), dc.ptr, nc*sizeof(float), cudaMemcpyDeviceToHost));
            double error=0.0, reference_scale=0.0;
            for (int b=0; b<std::min(batch,4); ++b) {
                const size_t oft=size_t(b)*s*p, oc=size_t(b)*s*s;
                for (int j=0; j<s; ++j) for (int i=0; i<s; ++i) {
                    double value=c[oc+i+size_t(j)*s];
                    for (int k=0; k<p; ++k)
                        value -= double(f[oft+i+size_t(k)*s])*double(t[oft+k+size_t(j)*p]);
                    error=std::max(error,std::abs(double(out[oc+i+size_t(j)*s])-value));
                    reference_scale=std::max(reference_scale,std::abs(value));
                }
            }
            for (int i=0; i<5; ++i)
                schur_update(handle.value, df.ptr, dt.ptr, dc.ptr, s, p, batch, compute);
            cuda_check(cudaDeviceSynchronize());
            Event start, stop;
            constexpr int repeat=50;
            cuda_check(cudaEventRecord(start.value));
            for (int i=0; i<repeat; ++i)
                schur_update(handle.value, df.ptr, dt.ptr, dc.ptr, s, p, batch, compute);
            cuda_check(cudaEventRecord(stop.value));
            cuda_check(cudaEventSynchronize(stop.value));
            float ms=0;
            cuda_check(cudaEventElapsedTime(&ms,start.value,stop.value));
            std::cout << (variant==0 ? "strict FP32" : "TF32 permitted")
                      << ": ms/update-batch=" << ms/repeat
                      << ", sampled one-update max absolute error=" << error
                      << ", sampled relative max error=" << error/std::max(reference_scale,1e-300)
                      << '\n';
        }
        std::cout << "A TF32-enabled call alone does not prove a full solver speedup.\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return EXIT_FAILURE;
    }
}
