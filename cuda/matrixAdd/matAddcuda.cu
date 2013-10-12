#include "stdio.h"
#include "cuda_runtime.h"
#include "matAddcuda.h"

/**
 * CUDA Kernel Device code
 * Computes the matrix addition of a and b into c. 
 */


__global__ void matrixAdd(float *a, float *b, float *c, int n2)
{
  int x = blockIdx.x;
  int y = blockIdx.y;
  int i = (n2*y) + x;
  c[i] = a[i] + b[i];
}
  
