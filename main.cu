#include <iostream>
#include <cuda_runtime.h>


__global__ void gaussian_blur(int width, int height, unsigned char* input, unsigned char* output){
    int pixel_col = blockIdx.x * blockDim.x + threadIdx.x;
    int pixel_row = blockIdx.y * blockDim.y + threadIdx.y;
    int window_size = 2
    if (pixel_col < width && pixel_row < height) {
        //zmienimy wyzerowanie na rozwijanie/lustro
        float pixel_sum = 0.0f;
        int pixel_count = 0;

        for (int blur_row = -window_size; blur_row <= window_size; blur_row++) {
            for (int blur_col = -window_size; blur_col <= window_size; blur_col++) {
                //tu trzeba przemnożyć przez macierz może zmienić pętle
            }
        }

    }
}


int main() {
    
    gaussian_blur<<<blocks, threads>>>(d_input, d_output, WIDTH, HEIGHT, BLUR_SIZE);
    //cuda errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    return 0;
}