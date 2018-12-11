/*
============================================================================
Filename    : algorithm.c
Author      : Your name goes here
SCIPER      : Your SCIPER number
============================================================================
*/

#include <iostream>
#include <iomanip>
#include <sys/time.h>
#include <cuda_runtime.h>
#include <math.h>
using namespace std;

// CPU Baseline
void array_process(double *input, double *output, int length, int iterations)
{
    double *temp;

    for(int n=0; n<(int) iterations; n++)
    {
        for(int i=1; i<length-1; i++)
        {
            for(int j=1; j<length-1; j++)
            {
                output[(i)*(length)+(j)] = (input[(i-1)*(length)+(j-1)] +
                                            input[(i-1)*(length)+(j)]   +
                                            input[(i-1)*(length)+(j+1)] +
                                            input[(i)*(length)+(j-1)]   +
                                            input[(i)*(length)+(j)]     +
                                            input[(i)*(length)+(j+1)]   +
                                            input[(i+1)*(length)+(j-1)] +
                                            input[(i+1)*(length)+(j)]   +
                                            input[(i+1)*(length)+(j+1)] ) / 9;

            }
        }
        output[(length/2-1)*length+(length/2-1)] = 1000;
        output[(length/2)*length+(length/2-1)]   = 1000;
        output[(length/2-1)*length+(length/2)]   = 1000;
        output[(length/2)*length+(length/2)]     = 1000;

        temp = input;
        input = output;
        output = temp;
    }
}

__global__ void GPU_processing(double *input, double *output, int length) {
    //for(int i = 0; i<iterations; i++) {
        int x = (blockIdx.x*blockDim.x) + threadIdx.x;
        int y = (blockIdx.y*blockDim.y) + threadIdx.y;
        int element_id = (y*length) + x;

        if (x >= length || y >= length || x%(length-1) == 0 || y%(length-1) == 0 ||
            (x==length/2 - 1 && (y==length/2 || y==length/2-1)) ||
            (x==length/2 && (y==length/2 || y==length/2-1))) return;

        output[element_id] = (input[(y-1)*(length)+(x-1)] +
                                            input[(y-1)*(length)+(x)]   +
                                            input[(y-1)*(length)+(x+1)] +
                                            input[(y)*(length)+(x-1)]   +
                                            input[(y)*(length)+(x)]     +
                                            input[(y)*(length)+(x+1)]   +
                                            input[(y+1)*(length)+(x-1)] +
                                            input[(y+1)*(length)+(x)]   +
                                            input[(y+1)*(length)+(x+1)] ) / 9;
    //}
}

// GPU Optimized function
void GPU_array_process(double *input, double *output, int length, int iterations)
{
    //Cuda events for calculating elapsed time
    cudaEvent_t cpy_H2D_start, cpy_H2D_end, comp_start, comp_end, cpy_D2H_start, cpy_D2H_end;
    cudaEventCreate(&cpy_H2D_start);
    cudaEventCreate(&cpy_H2D_end);
    cudaEventCreate(&cpy_D2H_start);
    cudaEventCreate(&cpy_D2H_end);
    cudaEventCreate(&comp_start);
    cudaEventCreate(&comp_end);

    /* Preprocessing goes here */
    double* gpu_output;
    cudaMalloc((void**)&gpu_output, length*length*sizeof(double));
    double* gpu_input;
    cudaMalloc((void**)&gpu_input, length*length*sizeof(double));
    double* temp;

    cudaEventRecord(cpy_H2D_start);

    /* Copying array from host to device goes here */
    cudaMemcpy((void*)gpu_input, (void*)input, length*length*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy((void*)gpu_output, (void*)output, length*length*sizeof(double), cudaMemcpyHostToDevice);

    cudaEventRecord(cpy_H2D_end);
    cudaEventSynchronize(cpy_H2D_end);

    //Copy array from host to device
    cudaEventRecord(comp_start);

    /* GPU calculation goes here */
    dim3 thrsPerBlock(16,16);//256 threads par blocks
    int nbTB = ceil(sqrt(ceil(length*length/256)));
    dim3 nBlks(nbTB, nbTB);
	
	
    for(int i = 0; i < iterations; i++) {
        GPU_processing<<< nBlks, thrsPerBlock>>>(gpu_input, gpu_output, length);
        cudaThreadSynchronize();
        temp = gpu_input;
        gpu_input = gpu_output;
        gpu_output = temp;
    }


    cudaEventRecord(comp_end);
    cudaEventSynchronize(comp_end);

    cudaEventRecord(cpy_D2H_start);

    /* Copying array from device to host goes here */
    cudaMemcpy((void*)output, (void*)gpu_input, length*length*sizeof(double), cudaMemcpyDeviceToHost);

    cudaEventRecord(cpy_D2H_end);
    cudaEventSynchronize(cpy_D2H_end);

    /* Postprocessing goes here */
    cudaFree(gpu_input);
    cudaFree(gpu_output);


    float time;
    cudaEventElapsedTime(&time, cpy_H2D_start, cpy_H2D_end);
    cout<<"Host to Device MemCpy takes "<<setprecision(4)<<time/1000<<"s"<<endl;

    cudaEventElapsedTime(&time, comp_start, comp_end);
    cout<<"Computation takes "<<setprecision(4)<<time/1000<<"s"<<endl;

    cudaEventElapsedTime(&time, cpy_D2H_start, cpy_D2H_end);
    cout<<"Device to Host MemCpy takes "<<setprecision(4)<<time/1000<<"s"<<endl;
}
