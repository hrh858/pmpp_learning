#include <cuda_runtime.h>
#include <png.h>

__global__ void blurFilterKernel(unsigned char *inImage, unsigned char
*outImage, int cols, int rows, int filterSize) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < cols && row < rows) {
        int pixelIdx = (row * cols + col) * 3;

        int pixelsCount = 0;
        // TODO: Careful with overflows in these:
        int rAcc = 0;
        int gAcc = 0;
        int bAcc = 0;

        for (int blurRowOffset = -filterSize; blurRowOffset <= filterSize; ++blurRowOffset) {
            for (int blurColOffset = -filterSize; blurColOffset <= filterSize; ++blurColOffset) {
                int blurCol = col + blurColOffset;
                int blurRow = row + blurRowOffset;
                
                if (blurRow >= 0 && blurRow < rows && blurCol >= 0 && blurCol < cols) {
                    ++pixelsCount;
                    int blurPixelIdx = (blurRow * cols + blurCol)*3;

                    rAcc += inImage[blurPixelIdx + 0];
                    gAcc += inImage[blurPixelIdx + 1];
                    bAcc += inImage[blurPixelIdx + 2];
                }
            }
        }

        outImage[pixelIdx + 0] = (unsigned char) (rAcc / pixelsCount);
        outImage[pixelIdx + 1] = (unsigned char) (gAcc / pixelsCount);
        outImage[pixelIdx + 2] = (unsigned char) (bAcc / pixelsCount);
    }
}

void blurFilter(unsigned char *h_inImage, unsigned char *h_outImage, int cols, int rows, int filterSize) {
    int size = cols * rows * 3 * sizeof(unsigned char);
    unsigned char *d_inImage, *d_outImage;
    
    cudaMalloc((void **) &d_inImage, size);
    cudaMalloc((void **) &d_outImage, size);

    cudaMemcpy(d_inImage, h_inImage, size, cudaMemcpyHostToDevice);

    dim3 dimBlock(16, 16, 1);
    dim3 dimGrid(ceil(rows/16.0), ceil(cols/16.0), 1);
    blurFilterKernel<<<dimGrid, dimBlock>>>(d_inImage, d_outImage, cols, rows, filterSize);

    cudaMemcpy(h_outImage, d_outImage, size, cudaMemcpyDeviceToHost);

    cudaFree(d_inImage);
    cudaFree(d_outImage);
}

// ChatGPT generated function to load the image, did not want to spend the time since my focus is in CUDA
void loadInImage(const char* filename, unsigned char **inImage, int *cols, int *rows) {
    FILE *fp = fopen(filename, "rb");
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

    *cols = png_get_image_width(png, info);
    *rows = png_get_image_height(png, info);
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
    *inImage = (unsigned char *)malloc(*cols * *rows * 3 * sizeof(unsigned char));

    png_bytep *row_pointers = (png_bytep *)malloc(sizeof(png_bytep) * *rows);
    for (int y = 0; y < *rows; y++) {
        row_pointers[y] = (png_byte *)((*inImage) + y * *cols * 3);
    }

    png_read_image(png, row_pointers);

    fclose(fp);

    png_destroy_read_struct(&png, &info, NULL);
    free(row_pointers);
}

void storeOutImage(const char* filename, unsigned char* imOut, int cols, int rows) {
    FILE *fp = fopen(filename, "wb");
    if(!fp) {
        fprintf(stderr, "Could not open file for writing\n");
        return;
    }

    png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png) {
        fprintf(stderr, "Could not allocate write struct\n");
        fclose(fp);
        return;
    }

    png_infop info = png_create_info_struct(png);
    if (!info) {
        fprintf(stderr, "Could not allocate info struct\n");
        png_destroy_write_struct(&png, (png_infopp)NULL);
        fclose(fp);
        return;
    }

    if (setjmp(png_jmpbuf(png))) {
        fprintf(stderr, "Error during png creation\n");
        png_destroy_write_struct(&png, &info);
        fclose(fp);
        return;
    }

    png_init_io(png, fp);

    // Output is 8bit depth, RGB format
    png_set_IHDR(
        png,
        info,
        cols, rows,
        8,
        PNG_COLOR_TYPE_RGB,
        PNG_INTERLACE_NONE,
        PNG_COMPRESSION_TYPE_DEFAULT,
        PNG_FILTER_TYPE_DEFAULT
    );

    png_write_info(png, info);

    png_bytep row = (png_bytep) malloc(3 * cols * sizeof(png_byte));
    for(int y = 0; y < rows; y++) {
        for(int x = 0; x < cols; x++) {
            row[x*3 + 0] = imOut[(y*cols + x)*3 + 0]; // red
            row[x*3 + 1] = imOut[(y*cols + x)*3 + 1]; // green
            row[x*3 + 2] = imOut[(y*cols + x)*3 + 2]; // blue
        }
        png_write_row(png, row);
    }

    png_write_end(png, NULL);
    fclose(fp);

    if (png && info)
        png_destroy_write_struct(&png, &info);
    if (row)
        free(row);
}

int main() { 
    unsigned char *inImage, *outImage;
    int cols, rows;
    int filterSize = 5; // A 11x11 filter

    loadInImage("lenna.png", &inImage, &cols, &rows);
    outImage = (unsigned char *)malloc(cols * rows * 3 * sizeof(unsigned char));

    blurFilter(inImage, outImage, cols, rows, filterSize);

    storeOutImage("blurred_lenna.png", outImage, cols, rows);

    return 0;
}