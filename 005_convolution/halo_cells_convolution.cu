#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#define CONV_FILTER_R 2
__constant__ unsigned char d_CONV_FILTER[(CONV_FILTER_R * 2 + 1) * (CONV_FILTER_R * 2 + 1)];
const unsigned char h_CONV_FILTER[(CONV_FILTER_R * 2 + 1) * (CONV_FILTER_R * 2 + 1)] = {
    1, 2, 3, 2, 1,
    2, 3, 4, 3, 2,
    3, 4, 5, 4, 3,
    2, 3, 4, 3, 2,
    1, 2, 3, 2, 1
};

__global__ void convolutionKernel(unsigned char *in, unsigned char *out, int w, int h) {
    int outCol = blockDim.x * blockIdx.x + threadIdx.x;
    int outRow = blockDim.y * blockIdx.y + threadIdx.y;

    if (outCol >= w || outRow >= h) return;

    int outVal = 0;
    for (int fOffX = -CONV_FILTER_R; fOffX < CONV_FILTER_R + 1; ++fOffX) {
     for (int fOffY = -CONV_FILTER_R; fOffY < CONV_FILTER_R + 1; ++fOffY) {
         int offInCol = outCol + fOffX;
         int offInRow = outRow + fOffY;
         if (offInCol >= 0 && offInCol < w && offInRow >= 0 && offInRow < h) {
            unsigned char fVal = d_CONV_FILTER[(fOffY + CONV_FILTER_R)  * (CONV_FILTER_R * 2 + 1) + fOffX];
            unsigned char inVal = in[offInRow * w + offInCol];
            outVal += inVal * fVal;
         }
     }
    }

    outVal /= 65;
    out[outRow * w + outCol] = (unsigned char)(outVal > 255 ? 255 : outVal);
};

void convolution(unsigned char *h_inImg, unsigned char *h_outImg, int w, int h) {
    int sizeIn = w * h * sizeof(unsigned char);
    int sizeOut = sizeIn;

    unsigned char *d_inImg, *d_outImg;
    cudaMalloc((void **) &d_inImg, sizeIn);
    cudaMalloc((void **) &d_outImg, sizeOut);

    cudaMemcpy(d_inImg, h_inImg, sizeIn, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_CONV_FILTER, h_CONV_FILTER, sizeof(h_CONV_FILTER));

    dim3 dimBlock(16, 16, 1);
    dim3 dimGrid(ceil(w / 16.0), ceil(h / 16.0), 1);

    convolutionKernel<<<dimGrid, dimBlock>>>(d_inImg, d_outImg, w, h);

    cudaMemcpy(h_outImg, d_outImg, sizeOut, cudaMemcpyDeviceToHost);
    cudaFree(d_inImg);
    cudaFree(d_outImg);
}

int main(int argc, char** argv) {
    char *filename = argv[1];

    int x, y, n;
    unsigned char *in_im = stbi_load(filename, &x, &y, &n, 1);
    if (in_im == NULL) {
        printf("Error ocurred when loading image\n");
        return 1;
    }

    printf("x: %d, y: %d, n: %d\n", x, y, n);
    unsigned char *out_im = (unsigned char *)malloc(x * y * sizeof(unsigned char));
    if (out_im == NULL) {
        printf("Error allocating memory for the result image\n");
        stbi_image_free(in_im);
        return 1;
    }

    convolution(in_im, out_im, x, y);
    stbi_write_png("convolution_simple_output.png", x, y, 1, out_im, 0);

    free(out_im);
    stbi_image_free(in_im);
    return 0;
}
