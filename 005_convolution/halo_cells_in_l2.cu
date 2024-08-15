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
#define TILE_SIZE 16
__constant__ float d_CONV_FILTER[CONV_FILTER_R * 2 + 1][CONV_FILTER_R * 2 + 1];
const float h_CONV_FILTER[CONV_FILTER_R * 2 + 1][CONV_FILTER_R * 2 + 1] = {
    {1.0 / 25.0, 1.0 / 25.0, 1.0 / 25.0, 1.0 / 25.0, 1.0 / 25.0},
    {1.0 / 25.0, 1.0 / 25.0, 1.0 / 25.0, 1.0 / 25.0, 1.0 / 25.0},
    {1.0 / 25.0, 1.0 / 25.0, 1.0 / 25.0, 1.0 / 25.0, 1.0 / 25.0},
    {1.0 / 25.0, 1.0 / 25.0, 1.0 / 25.0, 1.0 / 25.0, 1.0 / 25.0},
    {1.0 / 25.0, 1.0 / 25.0, 1.0 / 25.0, 1.0 / 25.0, 1.0 / 25.0},
};

__global__ void convolutionKernel(unsigned char *in, unsigned char *out, int w,
                                  int h) {
  int col = blockIdx.x * TILE_SIZE + threadIdx.x;
  int row = blockIdx.y * TILE_SIZE + threadIdx.y;

  __shared__ float sharedMem[TILE_SIZE][TILE_SIZE];
  if (col < w && row < h) {
    sharedMem[threadIdx.y][threadIdx.x] = in[row * w + col];
  } else {
    sharedMem[threadIdx.y][threadIdx.x] = 0.0;
  }
  __syncthreads();

  if (col < w && row < h) {
    float acc = 0.0f;

    for (int fRow = 0; fRow < CONV_FILTER_R * 2 + 1; fRow++) {
      for (int fCol = 0; fCol < CONV_FILTER_R * 2 + 1; fCol++) {
        if (threadIdx.x - CONV_FILTER_R + fCol >= 0 &&
            threadIdx.x - CONV_FILTER_R + fCol < TILE_SIZE &&
            threadIdx.y - CONV_FILTER_R + fRow >= 0 &&
            threadIdx.y - CONV_FILTER_R + fRow < TILE_SIZE) {
          acc += d_CONV_FILTER[fRow][fCol] *
                 sharedMem[threadIdx.y - CONV_FILTER_R + fRow]
                          [threadIdx.x - CONV_FILTER_R + fCol];
          // acc += d_CONV_FILTER[fRow][fCol] *
          //          sharedMem[threadIdx.y + fRow]
          //                   [threadIdx.x + fCol];

        } else {
          if (row - CONV_FILTER_R + fRow >= 0 &&
              row - CONV_FILTER_R + fRow < h &&
              col - CONV_FILTER_R + fCol >= 0 &&
              col - CONV_FILTER_R + fCol < w) {
            acc += d_CONV_FILTER[fRow][fCol] *
                   in[(row - CONV_FILTER_R + fRow) * w +
                      (col - CONV_FILTER_R + fCol)];
          }
        }
      }
    }

    out[row * w + col] = (unsigned char)(acc);
  }
}

// __global__ void convolutionKernel(unsigned char *in, unsigned char *out, int
// w,
//                                   int h) {
//   // Calculate current thread's position in the input/output.
//   int col = blockIdx.x * TILE_SIZE + threadIdx.x;
//   int row = blockIdx.y * TILE_SIZE + threadIdx.y;

//   // Declare a shared memory matrix where the threads collaboratively load
//   // their corresponding element.
//   __shared__ float prefInTile[TILE_SIZE][TILE_SIZE];

//   // Load this thread's value if it's inside the bounds of the input/output
//   // matrix. Either, just store a 0.
//   if (col < w && row < h) {
//     prefInTile[threadIdx.y][threadIdx.x] = (float)(in[row * w + col]);
//   } else {
//     prefInTile[threadIdx.y][threadIdx.x] = 0.0f;
//   }

//   // Wait for all the threads to load their value before computing the
//   // convolution.
//   __syncthreads();

//   // Check if we're inside the input/output bounds.
//   if (col < w && row < h) {
//     float outputValue = 0.0f;

//     // Iterate through all of the input elements and their filter value.
//     for (int fRow = 0; fRow < CONV_FILTER_R * 2 + 1; fRow++) {
//       for (int fCol = 0; fCol < CONV_FILTER_R * 2 + 1; fCol++) {
//         // Check if the value is inside the tile (if branch) or is part
//         // of the halo cells (else branch).
//         if (threadIdx.x - CONV_FILTER_R + fCol >= 0 &&
//             threadIdx.x - CONV_FILTER_R + fCol < TILE_SIZE &&
//             threadIdx.y - CONV_FILTER_R + fRow >= 0 &&
//             threadIdx.y - CONV_FILTER_R + fRow < TILE_SIZE) {
//           // Retrieve the value for the prefetched matrix.
//           outputValue += d_CONV_FILTER[fRow][fCol] *
//                          prefInTile[threadIdx.y + fRow][threadIdx.x + fCol];
//         } else {
//           if (row - CONV_FILTER_R + fRow >= 0 &&
//               row - CONV_FILTER_R + fRow < h &&
//               col - CONV_FILTER_R + fCol >= 0 &&
//               col - CONV_FILTER_R + fCol < w) {
//             // Retrieve the value from the input matrix (probably already in
//             L2
//             // cache).
//             outputValue += d_CONV_FILTER[fRow][fCol] *
//                            in[(row - CONV_FILTER_R + fRow) * w +
//                               (col - CONV_FILTER_R + fCol)];
//           }
//         }
//       }
//     }

//     out[row * w + col] = (unsigned char)(outputValue);
//   }
// };

void convolution(unsigned char *h_inImg, unsigned char *h_outImg, int w,
                 int h) {
  int sizeIn = w * h * sizeof(unsigned char);
  int sizeOut = sizeIn;

  unsigned char *d_inImg, *d_outImg;
  cudaMalloc((void **)&d_inImg, sizeIn);
  cudaMalloc((void **)&d_outImg, sizeOut);

  cudaMemcpy(d_inImg, h_inImg, sizeIn, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(d_CONV_FILTER, h_CONV_FILTER, sizeof(h_CONV_FILTER));

  dim3 dimBlock(16, 16, 1);
  dim3 dimGrid(ceil(w / 16.0), ceil(h / 16.0), 1);

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

  printf("x: %d, y: %d, n: %d\n", x, y, n);
  unsigned char *out_im =
      (unsigned char *)malloc(x * y * sizeof(unsigned char));
  if (out_im == NULL) {
    printf("Error allocating memory for the result image\n");
    stbi_image_free(in_im);
    return 1;
  }

  convolution(in_im, out_im, x, y);
  stbi_write_png("cached_halo_output.png", x, y, 1, out_im, 0);

  free(out_im);
  stbi_image_free(in_im);
  return 0;
}
