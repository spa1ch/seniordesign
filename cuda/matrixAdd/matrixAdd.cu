/**
 * Copyright 1993-2013 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/**
 * Matrix addition: C = A + B.
 * Host code.
 *
 * This sample implements matrix multiplication as described in Chapter 3
 * of the programming guide.
 * It has been written for clarity of exposition to illustrate various CUDA
 * programming principles, not with the goal of providing the most
 * performant generic kernel for matrix multiplication.
 *
 * See also:
 * V. Volkov and J. Demmel, "Benchmarking GPUs to tune dense linear algebra,"
 * in Proc. 2008 ACM/IEEE Conf. on Superconducting (SC '08),
 * Piscataway, NJ: IEEE Press, 2008, pp. Art. 31:1-11.
 */

// System includes
#include <stdio.h>

#include "Stopwatch.h"

// CUDA runtime
#include <cuda_runtime.h>

/**
 * Matrix multiplication (CUDA Kernel) on the device: C = A * B
 * wA is A's width and wB is B's width
 */
template <int BLOCK_SIZE> __global__ void
matrixAddCUDA(float *C, float *A, float *B, int N, int nIter)
{
  int i =blockIdx.x*blockDim.x+threadIdx.x; 
  int j=blockIdx.y*blockDim.y+threadIdx.y; 
  int index =i+j*N; 
  if( i <N && j <N) 
  C[index]=A[index]+B[index]; 
  for (int it=1; it<nIter; ++it)
    C[index]+=B[index];
}

void constantInit(float *data, int size, float val)
{
  for (int i = 0; i < size; ++i)
  {
    data[i] = val;
  }
}

/**
 * Run a simple test of matrix multiplication using CUDA
 */
int matrixAddCUDABench(int argc, char **argv, int block_size, dim3 &dimsA, dim3 &dimsB, int nIter)
{
  // Allocate host memory for matrices A and B
  unsigned int size_A = dimsA.x * dimsA.y;
  unsigned int mem_size_A = sizeof(float) * size_A;
  float *h_A = (float *)malloc(mem_size_A);
  unsigned int size_B = dimsB.x * dimsB.y;
  unsigned int mem_size_B = sizeof(float) * size_B;
  float *h_B = (float *)malloc(mem_size_B);

  // Initialize host memory
  const float valB = 0.01f;
  constantInit(h_A, size_A, 1.0f);
  constantInit(h_B, size_B, valB);

  // Allocate device memory
  float *d_A, *d_B, *d_C;

  // Allocate host matrix C
  dim3 dimsC(dimsB.x, dimsA.y, 1);
  unsigned int mem_size_C = dimsC.x * dimsC.y * sizeof(float);
  float *h_C = (float *) malloc(mem_size_C);

  cudaMalloc((void **) &d_A, mem_size_A);
  cudaMalloc((void **) &d_B, mem_size_B);
  cudaMalloc((void **) &d_C, mem_size_C);

  // copy host memory to device
  cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice);

  // Setup execution parameters
  dim3 threads(block_size, block_size);
  dim3 grid(dimsB.x / threads.x, dimsA.y / threads.y);

  // Performs warmup operation using matrixAdd CUDA kernel
  if (block_size == 16)
  {
      matrixAddCUDA<16><<< grid, threads >>>(d_C, d_A, d_B, dimsA.x, nIter);
  }
  else
  {
      matrixAddCUDA<32><<< grid, threads >>>(d_C, d_A, d_B, dimsA.x, nIter);
  }

  cudaDeviceSynchronize();

  Stopwatch sw;
  sw.restart();
  int ncount = 0;
  double maxtime = 2.0;

  for (ncount = 0; sw.getTime()<maxtime; ++ncount) {
    // Execute the kernel
    matrixAddCUDA<32><<< grid, threads >>>(d_C, d_A, d_B, dimsA.x, nIter);
    // Copy result from device to host
    cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost);
  }
  sw.stop();

  double megaflops = 1.0e-6*dimsA.x*dimsA.y*ncount*nIter/sw.getTime();
  printf("%f\n",megaflops);



  // Clean up memory
  free(h_A);
  free(h_B);
  free(h_C);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  cudaDeviceReset();

  return EXIT_SUCCESS;
}


/**
 * Program main
 */
int main(int argc, char **argv)
{
  //printf("[Matrix Addition Using CUDA] - Starting...\n");

  // By default, we use device 0, otherwise we override the device ID based on what is provided at the command line
  int devID = 0;

  cudaDeviceProp deviceProp;
  cudaGetDevice(&devID);


  cudaGetDeviceProperties(&deviceProp, devID);

  //printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", devID, deviceProp.name, deviceProp.major, deviceProp.minor);

  // Use a larger block size for Fermi and above
  int block_size = (deviceProp.major < 2) ? 16 : 32;
  int msize;
  int nIter;
  
  for (msize = 100; msize<=1000; msize+=100){
  for (nIter = 2; nIter<=2048; nIter*=2){

  //printf("msize = %i nIter = %i \n",msize,nIter);
  //printf("nIter = %i \n",nIter);
  dim3 dimsA(msize, msize, 1);
  dim3 dimsB(msize, msize, 1);

  if (dimsA.x != dimsB.y)
  {
    printf("Error: outer matrix dimensions must be equal. (%d != %d)\n",
           dimsA.x, dimsB.y);
    exit(EXIT_FAILURE);
  }

  //printf("MatrixA(%d,%d), MatrixB(%d,%d)\n", dimsA.x, dimsA.y, dimsB.x, dimsB.y);
  matrixAddCUDABench(argc, argv, block_size, dimsA, dimsB, nIter);


  }
  } 
  return 0;
}
