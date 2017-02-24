/* Vector-Matrix multiplication: Y = A * X.
 * Device code.
 */

#ifndef _MATRIXMUL_KERNEL_H_
#define _MATRIXMUL_KERNEL_H_

#include <stdio.h>
#include "vec_mat_mult.h"

__global__ void vec_mat_kernel_naive(float *Ad, float *Xd, float *Yd)
{
	//Multiply A and X
	int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
	
	float Y_temp = 0;
	if (thread_id < MATRIX_SIZE){
		for (int k = 0; k < MATRIX_SIZE; k++){
			float Ad_element = Ad[k + MATRIX_SIZE * thread_id];
			float Xd_element = Xd[k];
			Y_temp += Ad_element * Xd_element;
		}
		Yd[thread_id] = Y_temp;
	}
}


__global__ void vec_mat_kernel_optimized(float *Ad, float *Xd, float *Yd)
{
	__shared__ float Asub[TILE_SIZE][TILE_SIZE];
    __shared__ float Xsub[TILE_SIZE];
	
	int tx = threadIdx.x; // Obtain the x-index within the thread block
    int ty = threadIdx.y; // Obtain the y-index within the thread block
    int row = blockDim.y * blockIdx.y + ty; // Perform the thread to data ID mapping
	int k = 0;
    int temp;
    double Ysub = 0.0f;
	
	while(k < MATRIX_SIZE){
		if (tx < TILE_SIZE && row < MATRIX_SIZE){
			Asub[ty][tx] = Ad[row * MATRIX_SIZE + k + tx];
			Xsub[tx] = Xd[k + tx]; 
		}
		
		__syncthreads();
		
		for (temp = 0; temp < TILE_SIZE; temp++)
			Ysub += Asub[ty][temp] * Xsub[temp];
		
		__syncthreads();
		
		k += TILE_SIZE;
	}
	
	Yd[row] = (float)Ysub;
}



#endif // #ifndef _MATRIXMUL_KERNEL_H_
