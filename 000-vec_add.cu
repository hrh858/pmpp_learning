#include <cuda_runtime.h>
#include <stdio.h>

#define N 4000

__global__ void vecAddKernel(float* inA, float *inB, float *outC, int n) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < n) {
        outC[i] = inA[i] + inB[i];
    }
}

void vecAdd(float *h_inA, float *h_inB, float *h_outC, int n) {
    int size = n * sizeof(float);
    float *d_inA, *d_inB, *d_outC;

    cudaMalloc((void**) &d_inA, size);
    cudaMalloc((void**) &d_inB, size);
    cudaMalloc((void**) &d_outC, size);

    cudaMemcpy(d_inA, h_inA, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_inB, h_inB, size, cudaMemcpyHostToDevice);

    vecAddKernel<<<ceil(n/256.0), 256>>>(d_inA, d_inB, d_outC, n);

    cudaMemcpy(h_outC, d_outC, size, cudaMemcpyDeviceToHost);

    cudaFree(d_inA);
    cudaFree(d_inB);
    cudaFree(d_outC);
}


int main() {
    float inA[N];
    float inB[N];
    float outC[N];

    for (int i = 0; i < N; i++) {
        inA[i] = i;
        inB[i] = i;
    }

    vecAdd(inA, inB, outC, N);

    for (int i = 0; i < N; i++) {
        printf("Value at position %d is %.2f\n", i, outC[i]);
    }
}