#include <cuda_runtime.h>
#include <stdio.h>

__global__ void matMulKernel(const int *m, const int *n, int *o, int size) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (row < size && col < size) {
    int acc = 0;
    for (int i = 0; i < size; ++i) {
      acc += m[row * size + i] * n[size * i + col];
    }
    o[row * size + col] = acc;
  }
}

void matMul(const int *h_m, const int *h_n, int *h_o, const int size) {
  int mallocSize = size * size * sizeof(int);
  int *d_m, *d_n, *d_o;

  cudaMalloc((void **)&d_m, mallocSize);
  cudaMalloc((void **)&d_n, mallocSize);
  cudaMalloc((void **)&d_o, mallocSize);

  cudaMemcpy(d_m, h_m, mallocSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_n, h_n, mallocSize, cudaMemcpyHostToDevice);

  dim3 dimBlock(16, 16, 1);
  dim3 dimGrid(ceil(size/16.0), ceil(size/16.0), 1);
  matMulKernel<<<dimGrid, dimBlock>>>(d_m, d_n, d_o, size);

  cudaMemcpy(h_o, d_o, mallocSize, cudaDeviceToHost);

  cudaFree(d_m);
  cudaFree(d_n);
  cudaFree(d_o);
}

int main() {
  // Squared matrix only
  const int size = 3;
  int m[size * size] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  int n[size * size] = {9, 8, 7, 6, 5, 4, 3, 2, 1};
  int *o = (int *)malloc(size * size * sizeof(int));

  matMul(m, n, o, size);

  printf("Result matrix:\n");
  for (int i = 0; i < size; ++i) {
      for (int j = 0; j < size; ++j) {
          printf("%d ", o[i * size + j]);
      }
      printf("\n");
  }

  return 0;
}
