#include <cuda_runtime.h>
#include <stdio.h>

#define TILE_SIZE 2
__global__ void tiledMatMulKernel(int *d_m, int *d_n, int *d_p, int matSize) {
	__shared__ int Ms[TILE_SIZE][TILE_SIZE];
	__shared__ int Ns[TILE_SIZE][TILE_SIZE];

	int row = blockIdx.y * TILE_SIZE + threadIdx.y;
	int col = blockIdx.x * TILE_SIZE + threadIdx.x;

	int acc = 0;
	for (int iterIdx = 0; iterIdx < matSize/TILE_SIZE; iterIdx++) {
		Ms[threadIdx.y][threadIdx.x] = d_m[row * matSize + TILE_SIZE * iterIdx + threadIdx.x];
		Ns[threadIdx.y][threadIdx.x] = d_n[col + iterIdx * TILE_SIZE * matSize + threadIdx.y * matSize];
		__syncthreads();

		for (int i = 0; i < TILE_SIZE; i++) {
			acc += Ms[threadIdx.y][i] * Ns[i][threadIdx.x];
		}
		__syncthreads();
	}
	d_p[row * matSize + col] = acc;
}

void tiledMatMul(int *h_m, int *h_n, int *h_p, int size) {
	int mallocSize = size * size * sizeof(int);
	int *d_m, *d_n, *d_p;

	cudaMalloc((void **)&d_m, mallocSize);
	cudaMalloc((void **)&d_n, mallocSize);
	cudaMalloc((void **)&d_p, mallocSize);

	cudaMemcpy(d_m, h_m, mallocSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_n, h_n, mallocSize, cudaMemcpyHostToDevice);

	dim3 dimBlock(2, 2, 1);
	dim3 dimGrid(ceil(size / 2.0), ceil(size / 2.0), 1);
	tiledMatMulKernel<<<dimGrid, dimBlock>>>(d_m, d_n, d_p, size);

	cudaMemcpy(h_p, d_p, mallocSize, cudaMemcpyDeviceToHost);

	cudaFree(d_m);
	cudaFree(d_n);
	cudaFree(d_p);
}

int main(int argc, char **argv) {
  const int size = 4;
  int m[size * size] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  int n[size * size] = {16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1};
  int pExp[size * size] = {80,  70,  60,  50,  240, 214, 188, 162,
                        400, 358, 316, 274, 560, 502, 444, 386};

  int *pRes = (int *)malloc(4 * 4 * sizeof(int));

  tiledMatMul(m, n, pRes, size);

  printf("Result matrix:\n");
  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < size; ++j) {
      printf("%d ", pRes[i * size + j]);
    }
    printf("\n");
  }

  printf("\n");

  printf("Expected matrix:\n");
  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < size; ++j) {
      printf("%d ", pRes[i * size + j]);
    }
    printf("\n");
  }

  tiledMatMul(m, n, pRes, size);
}
