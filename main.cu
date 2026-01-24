%%writefile pyrDown.cu
#include <iostream>
#include <cuda_runtime.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

const float h_gaussian_window[5][5] = {{1, 4, 7, 4, 1},
                                    {4, 16, 26, 16, 4},
                                    {7, 26, 41, 26, 7},
                                    {4, 16, 26, 16, 4},
                                    {1, 4, 7, 4, 1}
};

__constant__ float gaussian_window[5][5];

__global__ void gaussian_blur(unsigned char* input, unsigned char* output, int width, int height){
    int pixel_col = blockIdx.x * blockDim.x + threadIdx.x;
    int pixel_row = blockIdx.y * blockDim.y + threadIdx.y;
    int window_size = 2;

    if (pixel_col < width && pixel_row < height) {
        float pixel_sum_b = 0.0f;
        float pixel_sum_g = 0.0f;
        float pixel_sum_r = 0.0f;

        for (int blur_row = - window_size; blur_row <= window_size; blur_row++) {
            for (int blur_col = - window_size; blur_col <=  window_size; blur_col++) {
                //handle dla border cases jako lustro
                int curr_row = blur_row + pixel_row;
                int curr_col = blur_col + pixel_col;
                int pixel_idx = curr_row * width + curr_col;

                if (curr_row < 0) {
                curr_row = -curr_row;
                }
                else if (curr_row >= height) {
                    curr_row = height - 1 - (curr_row - height - 1);
                }
                if (curr_col < 0) {
                    curr_col = -curr_col;
                }
                else if (curr_col >= width) {
                    curr_col = width - 1 - (curr_col - height - 1);
                }
                pixel_sum_b += input[pixel_idx * 3]* gaussian_window[blur_col + window_size][blur_row + window_size];
                pixel_sum_g += input[pixel_idx * 3+1]* gaussian_window[blur_col + window_size][blur_row + window_size];
                pixel_sum_r += input[pixel_idx * 3+2]* gaussian_window[blur_col + window_size][blur_row + window_size];

            }
        }

        int b_pixel = (pixel_row * width + (pixel_col)) * 3;
        int g_pixel = b_pixel + 1;
        int r_pixel = b_pixel + 2;
        float output_b_pixel = (1.0/273.0) * pixel_sum_b;
        float output_g_pixel = (1.0/273.0) * pixel_sum_g;
        float output_r_pixel = (1.0/273.0) * pixel_sum_r;
        output[b_pixel] = (unsigned char)output_b_pixel;
        output[g_pixel] = (unsigned char)output_g_pixel;
        output[r_pixel] = (unsigned char)output_r_pixel;
    }

}

__global__ void resize_img(unsigned char* input, unsigned char* output, int width, int height, int h, int w){
    
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    int in_col = col*2 + 1;
    int in_row = row*2 + 1;

    int channel = 3;

    if(row < h && col < w){

        output[(row*w + col)*channel] = input[(in_row*width + in_col)*channel];
        output[(row*w + col)*channel + 1] = input[(in_row*width + in_col)*channel + 1];
        output[(row*w + col)*channel + 2] = input[(in_row*width + in_col)*channel + 2];

    }
        
}



int main() {
    std::string filename = "/content/cat.jpg";
    //mat zapisuje już jako tablice 1D ALE trzy kanały b, g, r
    cv::Mat image = cv::imread(filename, cv::IMREAD_COLOR);
    unsigned char *image_arr = image.data;
    unsigned char *h_output;
    unsigned char *d_input, *d_output;
    unsigned char *dr_output;
    int image_width = image.size().width;
    int image_height = image.size().height;
    int image_size = image_width*image_height * 3 * sizeof(unsigned char);
    h_output = (unsigned char*)malloc(image_size);


    cudaError_t err = cudaMalloc(&d_input, image_size);
    if (err != cudaSuccess) {
        printf("cudaMalloc d_input failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    err = cudaMemcpyToSymbol(gaussian_window, h_gaussian_window, 25 * sizeof(float));
    if (err != cudaSuccess) {
        printf("cudaMemcpyToSymbol failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    err = cudaMalloc(&d_output, image_size);
    if (err != cudaSuccess) {
        printf("cudaMalloc d_output failed: %s\n", cudaGetErrorString(err));
        return 1;
    }


    err = cudaMemcpy(d_input, image_arr, image_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("cudaMemcpy host to device failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    cudaGetLastError();

    dim3 threads(16, 16);
    dim3 blocks((image_width + threads.x - 1) / threads.x,
                (image_height + threads.y - 1) / threads.y);

    printf("Launching kernel with blocks(%d,%d) threads(%d,%d)\n",
           blocks.x, blocks.y, threads.x, threads.y);

    //stworzenie stream do asynchronicznego policzenia gauss blur,a potem skopiowania do host
    cudaStream_t compute_stream;
    cudaStreamCreate(&compute_stream);

    gaussian_blur<<<blocks, threads, 0, compute_stream>>>(d_input, d_output, image_width, image_height);
   
    int new_size = image_size/2; 
    int new_height = image_height/2;
    int new_width = image_width/2;

    dim3 re_threads(16,16);
    dim3 re_blocks((new_width + threads.x - 1)/threads.x, 
                    (new_height + threads.y - 1)/threads.y);
    //alokacja miejsca na zmniejszone zdjęcie
    err = cudaMalloc(&dr_output, new_size);
    if (err != cudaSuccess) {
        printf("cudaMalloc dr_output failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    resize_img<<<re_blocks, re_threads, 0, compute_stream>>>(d_output, dr_output, image_width, image_height, new_height, new_width);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Kernel execution error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    cudaMemcpy(h_output, dr_output, new_size, cudaMemcpyDeviceToHost);

    cv::Mat output_image(new_height, new_width, CV_8UC3, h_output);
    cv::imwrite("blurred_cat.jpg", output_image);


    cudaFree(d_output);
    cudaFree(dr_output);
    free(h_output);

    return 0;
}

