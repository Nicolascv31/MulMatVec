#include <iostream>
#include <cuda_runtime.h>

__global__ void matVecMul(const int* A, const int* x, int* y, int M, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M) {
        int sum = 0;
        for (int j = 0; j < N; ++j) {
            sum += A[row * N + j] * x[j];
        }
        y[row] = sum;
    }
}

int main() {
    int M = 10; 
    int N = 10; 

    size_t size_A = M * N * sizeof(int);
    size_t size_x = N * sizeof(int);
    size_t size_y = M * sizeof(int);

    
    int* h_A = (int*)malloc(size_A);
    int* h_x = (int*)malloc(size_x);
    int* h_y = (int*)malloc(size_y);

    
    for (int i = 0; i < M * N; ++i) h_A[i] = 6;
    for (int i = 0; i < N; ++i) h_x[i] = 2;

    
    int* d_A, * d_x, * d_y;
    cudaMalloc((void**)&d_A, size_A);
    cudaMalloc((void**)&d_x, size_x);
    cudaMalloc((void**)&d_y, size_y);

    
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x, size_x, cudaMemcpyHostToDevice);

   
    int threadsPerBlock = 256;
    int blocksPerGrid = (M + threadsPerBlock - 1) / threadsPerBlock;
    matVecMul << <blocksPerGrid, threadsPerBlock >> > (d_A, d_x, d_y, M, N);

    
    cudaMemcpy(h_y, d_y, size_y, cudaMemcpyDeviceToHost);

    
    std::cout << "Resultado del producto matriz * vector:" << std::endl;
    for (int i = 0; i < M; ++i) {
        std::cout << h_y[i] << " ";
    }
    std::cout << std::endl;

    
    cudaFree(d_A);
    cudaFree(d_x);
    cudaFree(d_y);
    free(h_A);
    free(h_x);
    free(h_y);

    return 0;
}
