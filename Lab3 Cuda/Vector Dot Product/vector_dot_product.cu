#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <time.h>
#include <sys/time.h>

// includes, kernels
#include "vector_dot_product_kernel.cu"

void run_test(unsigned int);
float compute_on_device(float *, float *,int);
void check_for_error(char *);
extern "C" float compute_gold( float *, float *, unsigned int);

int 
main( int argc, char** argv) 
{
	if(argc != 2){
		printf("Usage: vector_dot_product <num elements> \n");
		exit(0);	
	}
	unsigned int num_elements = atoi(argv[1]);
	run_test(num_elements);
	return 0;
}

void 
run_test(unsigned int num_elements) 
{
	// Obtain the vector length
	unsigned int size = sizeof(float) * num_elements;

	// Allocate memory on the CPU for the input vectors A and B
	float *A = (float *)malloc(size);
	float *B = (float *)malloc(size);
	
	// Randomly generate input data. Initialize the input data to be floating point values between [-.5 , 5]
	printf("Generating random vectors with values between [-.5, .5]. \n");	
	srand(time(NULL));
	for(unsigned int i = 0; i < num_elements; i++){
		A[i] = (float)rand()/(float)RAND_MAX - 0.5;
		B[i] = (float)rand()/(float)RAND_MAX - 0.5;
		//printf("A = %d;  B = %d. \n", A[i], B[i]);
	}
	
	printf("Generating dot product on the CPU. \n");
	struct timeval start, stop;	
	gettimeofday(&start, NULL);
	
	float reference = compute_gold(A, B, num_elements);
    
	gettimeofday(&stop, NULL);
	printf("Execution time on CPU = %fs. \n", (float)(stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec)/(float)1000000));
	
	/* Edit this function to compute the result vector on the GPU. 
       The result should be placed in the gpu_result variable. */
	float gpu_result = compute_on_device(A, B, num_elements);

	printf("Result on CPU: %f, result on GPU: %f. \n", reference, gpu_result);
    printf("Epsilon: %f. \n", fabsf(reference - gpu_result));

	// cleanup memory
	free(A);
	free(B);
	
	return;
}

/* Edit this function to compute the dot product on the device using atomic intrinsics. */
float 
compute_on_device(float *A_on_host, float *B_on_host, int num_elements)
{
	float *A_on_device = NULL;
    float *B_on_device = NULL;
	float *C_on_device = NULL;
	
    cudaMalloc ((void **) &A_on_device, num_elements * sizeof (float));
    cudaMemcpy (A_on_device, A_on_host, num_elements * sizeof (float), cudaMemcpyHostToDevice);
	
    cudaMalloc ((void **) &B_on_device, num_elements * sizeof (float));
    cudaMemcpy (B_on_device, B_on_host, num_elements * sizeof (float), cudaMemcpyHostToDevice);
	
	cudaMalloc ((void **) &C_on_device, num_elements * sizeof (float));
	cudaMemset( C_on_device, 0.0f, GRID_SIZE * sizeof(float) );
	
	int *mutex = NULL;
    cudaMalloc((void **)&mutex, sizeof(int));
    cudaMemset(mutex, 0, sizeof(int));
		
    dim3 thread_block (THREAD_BLOCK_SIZE, 1, 1);
    dim3 grid (GRID_SIZE, 1);
   
    struct timeval start, stop;	
	gettimeofday(&start, NULL);
	
    vector_dot_product_kernel <<< grid, thread_block >>> (A_on_device, B_on_device, C_on_device, num_elements, mutex);	 
	cudaThreadSynchronize();
	
	gettimeofday(&stop, NULL);
	printf("Execution time on GPU = %fs. \n", (float)(stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec)/(float)1000000));
	
	float gpu_result = 0.0f;
    cudaMemcpy(&gpu_result, C_on_device, sizeof(float), cudaMemcpyDeviceToHost);
    
	cudaFree(A_on_device); 
	cudaFree(B_on_device);
	cudaFree(C_on_device);
	
	return gpu_result;  
}
 
// This function checks for errors returned by the CUDA run time
void 
check_for_error(char *msg)
{
	cudaError_t err = cudaGetLastError();
	if(cudaSuccess != err){
		printf("CUDA ERROR: %s (%s). \n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
} 
