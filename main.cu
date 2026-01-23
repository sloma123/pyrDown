#include <iostream>
#include <cuda_runtime.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <thrust/device_vector.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/execution_policy.h>

__constant__ float h_window[5][5];

const float gaussian_window[5][5] = {{1, 4, 7, 4, 1},
                                    {4, 16, 26, 16, 4},
                                    {7, 26, 41, 26, 7},
                                    {4, 16, 26, 16, 4},
                                    {1, 4, 7, 4, 1}
}; 
//trzeba zedytować dla trzech kanałów
__global__ void gaussian_blur(unsigned char* input, unsigned char* output, int width, int height){
    int pixel_col = blockIdx.x * blockDim.x + threadIdx.x;
    int pixel_row = blockIdx.y * blockDim.y + threadIdx.y;
    int window_size = 2;

    //__shared__ unsigned char shared[]
    //naraze input jako tablica int, moze potem unsigned char
    if (pixel_col < width && pixel_row < height) {
        //zmienimy wyzerowanie na rozwijanie/lustro
        float pixel_sum_b = 0.0f;
        float pixel_sum_g = 0.0f;
        float pixel_sum_r = 0.0f;
        int pixel_count = 0;
        
        for (int blur_row = - window_size; blur_row <= window_size; blur_row++) {
            for (int blur_col = - window_size; blur_col <=  window_size; blur_col++) {
                //handle dla border cases jako lustro 
                int curr_row = blur_row + pixel_row;
                int curr_column = blur_col + pixel_col;
                if (curr_row < 0) {
                curr_row = -curr_row; 
                }
                else if (curr_row >= height) {
                    curr_row = height - 1 - curr_row;
                }
                if (curr_col < 0) {
                    curr_col = -curr_col;
                }
                else if (curr_col >= width) {
                    curr_col = width - 1 - curr_col;
                }
                pixel_sum_b += input[(curr_row) * width + curr_column]* gaussian_window[blur_col + window_size][blur_row + window_size];
                pixel_sum_g += input[(curr_row) * width + curr_column]* gaussian_window[blur_col + window_size][blur_row + window_size];
                pixel_sum_r += input[(curr_row) * width + curr_column]* gaussian_window[blur_col + window_size][blur_row + window_size];
                b_pixel = (pixel_row * width + pixel_col) * 3;
                g_pixel = b_pixel + 1;
                r_pixel = b_pixel + 2;

            }
        }
        output[b_pixel] = (unsigned char)(1.0/273.0) * pixel_sum;
        output[g_pixel] = (unsigned char)(1.0/273.0) * pixel_sum;
        output[r_pixel] = (unsigned char)(1.0/273.0) * pixel_sum;
    }
    
    __syncthreads();  
}


void resize(unsigned char* d_input_in, unsigned char* r_output, int height, int width, int h, int w){

    uchar3* d_input = (uchar3*)d_input_in;
    thrust::device_vector<uchar3> in_img(d_input, d_input + height*width);
    thrust::device_vector<uchar3> out_img(h*w);

    uchar3* pt_in = thrust::raw_pointer_cast(in_img.data());
    uchar3* pt_out = thrust::raw_pointer_cast(out_img.data());

    thrust::for_each(thrust::device, thrust::make_counting_iterator(0), thrust::make_counting_iterator(height*width),
        [=]__device__ (int idx) {
            if ((idx % 4 == 0)) {
                if(idx == 0){
                    pt_out[idx] = pt_in[idx];
                }
                else{
                    pt_out[idx/4] = pt_in[idx];
                }
            }
    });

    unsigned char* r_input = (unsigned char*)out_img;

    
}

int main() {
    string filename = "/teamspace/studios/this_studio/cat.jpg";
    //mat zapisuje już jako tablice 1D ALE trzy kanały b, g, r
    cv::Mat image = imread(filename, IMREAD_COLOR);
    unsigned char *image_arr = image.data;
    //puste wskaźniki które mają przechowywać tablice dla zdjęcia dla cpu i gpu
    unsigned char *h_output;
    unsigned char *d_input, *d_output;
    unsigned char *dr_output;
    int image_width = image.size().width;
    int image_height = image.size().height;
    int image_size = image_width*image_height * 3 * sizeof(unsigned char);
    //alokacja miejsca na cpu w rozmiarze zdjęcia x3 bo 3 kanały
    h_output = (unsigned char*)malloc(image_size);
    
    //alokacja miejca na tablice z brg zdjęcia na gpu
    cudaError_t err = cudaMalloc(&d_input, image_size);
    if (err != cudaSuccess) {
        printf("cudaMalloc d_input failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    //alokacja miejsca na zblurrowane zdjęcie na gpu
    err = cudaMalloc(&d_output, image_size);
    if (err != cudaSuccess) {
        printf("cudaMalloc d_output failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    //kopiowanie tablicy z z bgrbgrbgr do device
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
    
    //zmiana rozmiaru tablicy na gpu
    int new_size = image_size/2; 
    int new_height = image_height/2
    int new_width = image_width/2
    //alokacja miejsca na zmniejszone zdjęcie
    err = cudaMalloc(&dr_output, new_size);
    if (err != cudaSuccess) {
        printf("cudaMalloc dr_output failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    //dr_output = (unsigned char*)malloc(re_size/2);
    resize(d_output, dr_output, image_height, image_width, new_height, new_width);


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
    //kopiowanie tablicy z gpu na cpu
    cudaMemcpy(h_output, dr_output, new_size, cudaMemcpyDeviceToHost, compute_stream);

    cv::Mat output_image(new_height, new_width, CV_8UC3, h_output);
    cv::imwrite("blurred_cat.jpg", output_image);

    cudaFree(d_input);
    cudaFree(d_output);
    free(h_output);

    return 0;
}
