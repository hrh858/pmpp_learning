#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <png.h>

// The PNG library should be linked like so:
// nvcc colorToGrayscale.cu -lpng
// can be installed in Ubuntu like so:
// sudo apt install libpng-dev

__global__ void colorToGrayscaleKernel(unsigned char *inImage, unsigned char *outImage, int nX, int nY) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < nY && row < nX) {
        int outPixel = row * nY + col;
        int inPixel = outPixel * 3;

        unsigned char r = inImage[inPixel];
        unsigned char g = inImage[inPixel + 1];
        unsigned char b = inImage[inPixel + 2];

        outImage[outPixel] = 0.21*r + 0.71*g + 0.07*b;
    }
}

void colorToGrayscale(unsigned char *h_inImage, unsigned char* h_outImage, int nX, int nY) {
    int sizeIn = 3 * nX * nY * sizeof(unsigned char);
    int sizeOut = nX * nY * sizeof(unsigned char);
    unsigned char *d_inImage, *d_outImage;

    cudaMalloc((void **) &d_inImage, sizeIn);
    cudaMalloc((void **) &d_outImage, sizeOut);

    cudaMemcpy(d_inImage, h_inImage, sizeIn, cudaMemcpyHostToDevice);

    // Arbitrary block dimensions of 16x16 (256)
    dim3 dimBlock(16,16,1);
    dim3 dimGrid(ceil(nY/16.0), ceil(nX/16.0), 1);

    colorToGrayscaleKernel<<<dimGrid, dimBlock>>>(d_inImage, d_outImage, nX, nY);

    cudaMemcpy(h_outImage, d_outImage, sizeOut, cudaMemcpyDeviceToHost);

    cudaFree(d_inImage);
    cudaFree(d_outImage);
}

// ChatGPT generated function to load the image, did not want to spend the time since my focus is in CUDA
void loadLennaPng(unsigned char **inImage, int *xDim, int *yDim) {
    FILE *fp = fopen("lenna.png", "rb");
    if (!fp) {
        perror("File could not be opened for reading");
        return;
    }

    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png) {
        perror("png_create_read_struct failed");
        fclose(fp);
        return;
    }

    png_infop info = png_create_info_struct(png);
    if (!info) {
        perror("png_create_info_struct failed");
        png_destroy_read_struct(&png, NULL, NULL);
        fclose(fp);
        return;
    }

    if (setjmp(png_jmpbuf(png))) {
        perror("Error during init_io");
        png_destroy_read_struct(&png, &info, NULL);
        fclose(fp);
        return;
    }

    png_init_io(png, fp);
    png_read_info(png, info);

    *xDim = png_get_image_width(png, info);
    *yDim = png_get_image_height(png, info);
    png_byte bit_depth = png_get_bit_depth(png, info);
    png_byte color_type = png_get_color_type(png, info);

    if (bit_depth == 16)
        png_set_strip_16(png);

    if (color_type == PNG_COLOR_TYPE_PALETTE)
        png_set_palette_to_rgb(png);

    if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8)
        png_set_expand_gray_1_2_4_to_8(png);

    if (png_get_valid(png, info, PNG_INFO_tRNS))
        png_set_tRNS_to_alpha(png);

    if (color_type == PNG_COLOR_TYPE_GRAY ||
        color_type == PNG_COLOR_TYPE_GRAY_ALPHA)
        png_set_gray_to_rgb(png);

    // Comment out or remove the following line to avoid adding an alpha channel
    // png_set_filler(png, 0xFF, PNG_FILLER_AFTER);

    png_read_update_info(png, info);

    // Adjust the memory allocation for 3 bytes per pixel (RGB)
    *inImage = (unsigned char *)malloc(*xDim * *yDim * 3 * sizeof(unsigned char));

    png_bytep *row_pointers = (png_bytep *)malloc(sizeof(png_bytep) * *yDim);
    for (int y = 0; y < *yDim; y++) {
        row_pointers[y] = (png_byte *)((*inImage) + y * *xDim * 3);
    }

    png_read_image(png, row_pointers);

    fclose(fp);

    png_destroy_read_struct(&png, &info, NULL);
    free(row_pointers);
}

// Same as the function above, hehe
void saveGrayscaleLennaPng(unsigned char *outImage, int xDim, int yDim) {
    FILE *fp = fopen("grayscale_lenna.png", "wb");
    if (!fp) {
        perror("File could not be opened for writing");
        return;
    }

    png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png) {
        perror("png_create_write_struct failed");
        fclose(fp);
        return;
    }

    png_infop info = png_create_info_struct(png);
    if (!info) {
        perror("png_create_info_struct failed");
        png_destroy_write_struct(&png, NULL);
        fclose(fp);
        return;
    }

    if (setjmp(png_jmpbuf(png))) {
        perror("Error during init_io");
        png_destroy_write_struct(&png, &info);
        fclose(fp);
        return;
    }

    png_init_io(png, fp);

    png_set_IHDR(
        png,
        info,
        xDim, yDim,
        8,
        PNG_COLOR_TYPE_GRAY,
        PNG_INTERLACE_NONE,
        PNG_COMPRESSION_TYPE_DEFAULT,
        PNG_FILTER_TYPE_DEFAULT
    );

    png_bytep *row_pointers = (png_bytep *)malloc(sizeof(png_bytep) * yDim);
    for (int y = 0; y < yDim; y++) {
        row_pointers[y] = (png_byte *)(outImage + y * xDim);
    }

    png_set_rows(png, info, row_pointers);
    png_write_png(png, info, PNG_TRANSFORM_IDENTITY, NULL);

    fclose(fp);
    png_destroy_write_struct(&png, &info);
    free(row_pointers);
}

int main() {
    unsigned char *inImage, *outImage;
    int xDim, yDim;

    loadLennaPng(&inImage, &xDim, &yDim);
    outImage = (unsigned char *)malloc(xDim * yDim * sizeof(unsigned char));

    colorToGrayscale(inImage, outImage, xDim, yDim);

    saveGrayscaleLennaPng(outImage, xDim,  yDim);

    free(inImage);
    free(outImage);
    return 0;
}