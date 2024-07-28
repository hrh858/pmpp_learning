simple:
	nvcc -o convolution_simple -I. convolution_simple.cu

run_simple:
	./convolution_simple ../001_colorToGrayscale/grayscale_lenna.png

tiled1:
	nvcc -o convolution_tiled1 -I. convolution_tiled1.cu

run_tiled1:
	./convolution_tiled1 ../001_colorToGrayscale/grayscale_lenna.png

clean:
	rm -f convolution_simple 