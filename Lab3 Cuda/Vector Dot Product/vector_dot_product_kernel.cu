#ifndef _VECTOR_DOT_PRODUCT_KERNEL_H_
#define _VECTOR_DOT_PRODUCT_KERNEL_H_

#define THREAD_BLOCK_SIZE 256
#define GRID_SIZE 240

/* Edit this function to complete the functionality of dot product on the GPU using atomics. 
	You may add other kernel functions as you deem necessary. 
 */
__device__ void lock(int *mutex);
__device__ void unlock(int *mutex);

__global__ void vector_dot_product_kernel(float* A, float* B, float* C, unsigned int num_elements, int *mutex)
{	
	__shared__ float temp[THREAD_BLOCK_SIZE];
	
	int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x; 

	float local_sum = 0.0f;
	while(thread_id < num_elements){
		local_sum += A[thread_id] * B[thread_id];
		thread_id += stride;
	}
	
	temp[threadIdx.x] = local_sum;
	__syncthreads();
	
	unsigned int i = THREAD_BLOCK_SIZE/2;
	while(i != 0){
		if(threadIdx.x < i){
			temp[threadIdx.x] += temp[threadIdx.x + i];
		}
		__syncthreads();
		i = i/2;
	}
	
	if(threadIdx.x == 0){
		lock(mutex);
		C[0] += temp[0];
		unlock(mutex);
	}
}

__device__ void lock(int *mutex)
{
    while(atomicCAS(mutex, 0, 1) != 0);
}

/* Using exchange to release mutex. */
__device__ void unlock(int *mutex)
{
	atomicExch(mutex, 0);
}


#endif // #ifndef _VECTOR_DOT_PRODUCT_KERNEL_H
