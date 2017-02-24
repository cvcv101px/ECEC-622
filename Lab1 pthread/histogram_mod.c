/* Pthread Lab: Histrogram generation
 * Author: Naga Kandasamy
 * Date modified: 10/11/2015
 *
 * compile as follows: 
 * gcc -o histogram histogram.c -std=c99 -lpthread -lm
 */
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <math.h>
#include <float.h>
#include <pthread.h>
#include <semaphore.h>

void run_test(int);
void compute_gold(int *, int *, int, int);
void compute_using_pthreads(int *, int *, int, int);
void *my_histogram(void *);
void check_histogram(int *, int, int);

#define HISTOGRAM_SIZE 500      /* Number of histrogram bins. */
#define NUM_THREADS 4           /* Number of threads. */

/* Data structure defining what to pass to each worker thread. */
typedef struct args_for_thread{
		  int thread_id; // The thread ID
		  int offset; // Starting offset for each thread within the vectors 
		  int counter;
		  int *input_data;
		  int *histogram;
		  int *buffer;
} ARGS_FOR_THREAD;

sem_t sem_histogram;

int 
main( int argc, char** argv) 
{
	if(argc != 2){
		printf("Usage: histogram <num elements> \n");
		exit(0);	
	}
	int num_elements = atoi(argv[1]);
	run_test(num_elements);
	return 0;
}

void 
run_test(int num_elements) 
{
	float diff;
	int i; 

    /* Allocate memory for the histrogram structures. */
	int *reference_histogram = (int *)malloc(sizeof(int) * HISTOGRAM_SIZE);
	int *histogram_using_pthreads = (int *)malloc(sizeof(int) * HISTOGRAM_SIZE); 

	/* Generate input data---integer values between 0 and (HISTOGRAM_SIZE - 1). */
    int size = sizeof(int) * num_elements;
	int *input_data = (int *)malloc(size);

	for(i = 0; i < num_elements; i++)
		input_data[i] = floorf((HISTOGRAM_SIZE - 1) * (rand()/(float)RAND_MAX));

    /* Compute the reference solution on the CPU. */
	printf("Creating the reference histogram. \n"); 
	struct timeval start, stop;	
	gettimeofday(&start, NULL);

	compute_gold(input_data, reference_histogram, num_elements, HISTOGRAM_SIZE);

	gettimeofday(&stop, NULL);
	printf("CPU run time for reference= %0.4f s. \n", (float)(stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec)/(float)1000000));
	/* check_histogram(reference_histogram, num_elements, HISTOGRAM_SIZE); 
	
	/* Compute the histogram using pthreads. The result histogram should be stored in the 
     * histogram_using_pthreads array. */
	printf("Creating histogram using pthreads. \n");
	gettimeofday(&start, NULL);
	
	compute_using_pthreads(input_data, histogram_using_pthreads, num_elements, HISTOGRAM_SIZE);
	
	gettimeofday(&stop, NULL);
	printf("CPU run time for pthread (4 threads) = %0.4f s. \n", (float)(stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec)/(float)1000000));
	/* check_histogram(histogram_using_pthreads, num_elements, HISTOGRAM_SIZE); */

	/* Compute the differences between the reference and pthread results. */
	diff = 0.0;
    for(i = 0; i < HISTOGRAM_SIZE; i++)
		diff = diff + abs(reference_histogram[i] - histogram_using_pthreads[i]);

	printf("Difference between the reference and pthread results: %f. \n", diff);
   
	/* cleanup memory. */
	free(input_data);
	free(reference_histogram);
	free(histogram_using_pthreads);

	pthread_exit(NULL);
}

/* This function computes the reference solution. */
void 
compute_gold(int *input_data, int *histogram, int num_elements, int histogram_size)
{
  int i;
  
   for(i = 0; i < histogram_size; i++)   /* Initialize histogram. */
       histogram[i] = 0; 

   for(i = 0; i < num_elements; i++)     /* Bin the elements. */
			 histogram[input_data[i]]++;
}


/* Write the function to compute the histogram using pthreads. */
void 
compute_using_pthreads(int *input_data, int *histogram, int num_elements, int histogram_size)
{
	int chunk_size;
	chunk_size = (int)floor((int)num_elements/(int)NUM_THREADS);
	
	ARGS_FOR_THREAD *args_for_thread;
	
    pthread_t thread_id[NUM_THREADS];
	
	sem_init(&sem_histogram, 0, 1);
	
	int i; int j;
	int **buffer = (int **)malloc(NUM_THREADS * sizeof(int *));
	for(i = 0; i < NUM_THREADS; i++)
		buffer[i] = (int *)malloc(histogram_size * sizeof(int));
	
	for(i = 0; i < NUM_THREADS; i++)
		for(j = 0; j < histogram_size; j++ )
			buffer[i][j] = 0;
	
	for(i = 0; i < NUM_THREADS; i++){
		args_for_thread = (ARGS_FOR_THREAD *)malloc(sizeof(ARGS_FOR_THREAD));
		args_for_thread->thread_id = i;
		args_for_thread->offset = chunk_size * i;
		args_for_thread->counter = (i == (NUM_THREADS - 1) ? (chunk_size + ( num_elements % NUM_THREADS)) : chunk_size);
		
		args_for_thread->input_data = input_data;
		args_for_thread->histogram = histogram;
		args_for_thread->buffer = buffer[i];
		
		if((pthread_create(&thread_id[i], NULL, my_histogram, (void *)args_for_thread)) != 0)
			exit(-1);
	}
	
	for(i = 0; i < NUM_THREADS; i++)
		pthread_join(thread_id[i], NULL);
}

void
*my_histogram(void *args)
{
	ARGS_FOR_THREAD *args_for_me = (ARGS_FOR_THREAD *)args;
	
	int i;
	for(i = args_for_me->offset; i < (args_for_me->counter + args_for_me->offset); i++)
		args_for_me->buffer[args_for_me->input_data[i]]++;
	
	sem_wait(&sem_histogram);
	
	for(i = 0; i < HISTOGRAM_SIZE; i++)
		args_for_me->histogram[i] += args_for_me->buffer[i];
	
	sem_post(&sem_histogram);
	
	pthread_exit(0);
}
/* Helper function to check for correctness of the resulting histogram. */
void 
check_histogram(int *histogram, int num_elements, int histogram_size)
{
	int sum = 0;
	for(int i = 0; i < histogram_size; i++)
		sum += histogram[i];

	printf("Number of histogram entries = %d. \n", sum);
	if(sum == num_elements)
		printf("Histogram generated successfully. \n");
	else
		printf("Error generating histogram. \n");
}



