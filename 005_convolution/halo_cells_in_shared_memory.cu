#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#define CONV_FILTER_R 2
#define TILE_SIZE 28
__constant__ unsigned char
    d_CONV_FILTER[(CONV_FILTER_R * 2 + 1) * (CONV_FILTER_R * 2 + 1)];
const unsigned char
    h_CONV_FILTER[(CONV_FILTER_R * 2 + 1) * (CONV_FILTER_R * 2 + 1)] = {
        1, 2, 3, 2, 1, // 9
        2, 3, 4, 3, 2, // 14
        3, 4, 5, 4, 3, // 19
        2, 3, 4, 3, 2, // 14
        1, 2, 3, 2, 1  // 9
};

// const unsigned char h_CONV_FILTER[(CONV_FILTER_R * 2 + 1) * (CONV_FILTER_R *
// 2 + 1)] = {
//     5, 5, 5, 5, 5, // 25
//     1, 1, 1, 1, 1, // 5
//     10, 10, 10, 10, 10, // 50
//     1, 1, 1, 1, 1, // 5
//     5, 5, 5, 5, 5 // 25
// };

__global__ void convolutionKernel(unsigned char *in, unsigned char *out, int w,
                                  int h) {
  int inTileSize = TILE_SIZE + CONV_FILTER_R * 2; // Same as blockDim
  int outTileSize = TILE_SIZE;

  // Calculate the col and row to load, displace it by the filter radius.
  int col = outTileSize * blockIdx.x + threadIdx.x - CONV_FILTER_R;
  int row = outTileSize * blockIdx.y + threadIdx.y - CONV_FILTER_R;

  // Declare shared memory to load the input values. It has to have the size of
  // the input tiles.
  __shared__ float blockSharedEls[(TILE_SIZE + CONV_FILTER_R * 2) *
                                  (TILE_SIZE + CONV_FILTER_R * 2)];

  // Load the value to shared memory. 0 if out of bounds.
  if ((col < 0 || col >= w) || (row < 0 || row >= h)) {
    blockSharedEls[threadIdx.y * inTileSize + threadIdx.x] = 0.0;
  } else {
    blockSharedEls[threadIdx.y * inTileSize + threadIdx.x] =
        (float)(in[row * w + col]);
  }
  // Barrier sync before to guarantee that all the loads happened.
  __syncthreads();

  // Early return from threads that may fall outside the output.
  if ((row < 0 || row >= h) || (col < 0 || col >= w))
    return;

  int tileCol = threadIdx.x - CONV_FILTER_R;
  int tileRow = threadIdx.y - CONV_FILTER_R;
  // Early return from threads belonging to the halo part of the block.
  if ((tileRow < 0 || tileRow >= outTileSize) ||
      (tileCol < 0 || tileCol >= outTileSize))
    return;

  // Computer the convolution
  float acc = 0.0;
  for (int convOffY = 0; convOffY < CONV_FILTER_R * 2 + 1; ++convOffY) {
    for (int convOffX = 0; convOffX < CONV_FILTER_R * 2 + 1; ++convOffX) {
      float inVal = blockSharedEls[(tileRow + convOffX) * inTileSize +
                                   (tileCol + convOffX)];
      float filterVal =
          (float)d_CONV_FILTER[convOffY * (CONV_FILTER_R * 2 + 1) + convOffX];
      acc += inVal * filterVal;
    }
  }
  acc /= 65.0;
  acc = acc > 255.0 ? 255.0 : acc;

  out[row * w + col] = (unsigned char)acc;
};

void convolution(unsigned char *h_inImg, unsigned char *h_outImg, int w,
                 int h) {
  int sizeIn = w * h * sizeof(unsigned char);
  int sizeOut = sizeIn;

  unsigned char *d_inImg, *d_outImg;
  cudaMalloc((void **)&d_inImg, sizeIn);
  cudaMalloc((void **)&d_outImg, sizeOut);

  cudaMemcpy(d_inImg, h_inImg, sizeIn, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(d_CONV_FILTER, h_CONV_FILTER, sizeof(h_CONV_FILTER));

  int blockSize = TILE_SIZE + CONV_FILTER_R * 2;
  dim3 dimBlock(blockSize, blockSize, 1);
  dim3 dimGrid(ceil(w / float(TILE_SIZE)), ceil(h / float(TILE_SIZE)), 1);

  convolutionKernel<<<dimGrid, dimBlock>>>(d_inImg, d_outImg, w, h);

  cudaMemcpy(h_outImg, d_outImg, sizeOut, cudaMemcpyDeviceToHost);
  cudaFree(d_inImg);
  cudaFree(d_outImg);
}

int main(int argc, char **argv) {
  char *filename = argv[1];

  int x, y, n;
  unsigned char *in_im = stbi_load(filename, &x, &y, &n, 1);
  if (in_im == NULL) {
    printf("Error ocurred when loading image\n");
    return 1;
  }

  unsigned char *out_im =
      (unsigned char *)malloc(x * y * sizeof(unsigned char));
  if (out_im == NULL) {
    printf("Error allocating memory for the result image\n");
    stbi_image_free(in_im);
    return 1;
  }

  convolution(in_im, out_im, x, y);
  stbi_write_png("convolution_tiled1_output.png", x, y, 1, out_im, 0);

  free(out_im);
  stbi_image_free(in_im);
  return 0;
}
