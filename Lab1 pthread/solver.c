
/* Code for the equation solver. 
 * Author: Naga Kandasamy, 10/20/2012
 * Date last modified: 10/11/2015
 *
 * Compile as follows:
 * gcc -o solver solver.c solver_gold.c -std=c99 -lm -lpthread 
 */

#include <stdio.h>
#include <string.h>
#include <malloc.h>
#include <time.h>
#include <stdlib.h>
#include <pthread.h>
#include <math.h>
#include "grid.h" 
#include <semaphore.h>

typedef struct barrier_struct{
		  sem_t counter_sem; // Protects access to the counter
		  sem_t barrier_sem; // Signals that barrier is safe to cross
		  int counter; // The value itself
} BARRIER;
// Create the barrier data structure
BARRIER barrier;

void barrier_sync(BARRIER *);
void* compute_jacobi(void *);
void* compute_redblack(void *);

extern int compute_gold(GRID_STRUCT *);
int compute_using_pthreads_jacobi(GRID_STRUCT *);
int compute_using_pthreads_red_black(GRID_STRUCT *);
void compute_grid_differences(GRID_STRUCT *, GRID_STRUCT *, GRID_STRUCT *);


int NUM_THREADS;
GRID_STRUCT * grid_2_temp;
GRID_STRUCT * grid_3_temp;
int n;//the number of lines that each thread compute



/* This function prints the grid on the screen. */
void 
display_grid(GRID_STRUCT *my_grid)
{
    for(int i = 0; i < my_grid->dimension; i++)
        for(int j = 0; j < my_grid->dimension; j++)
            printf("%f \t", my_grid->element[i * my_grid->dimension + j]);
   		
    printf("\n");
}


/* Print out statistics for the converged values, including min, max, and average. */
void 
print_statistics(GRID_STRUCT *my_grid)
{
    float min = INFINITY;
    float max = 0.0;
    double sum = 0.0; 
    
    for(int i = 0; i < my_grid->dimension; i++){
        for(int j = 0; j < my_grid->dimension; j++){
            sum += my_grid->element[i * my_grid->dimension + j];
           
            if(my_grid->element[i * my_grid->dimension + j] > max) 
                max = my_grid->element[i * my_grid->dimension + j];

				if(my_grid->element[i * my_grid->dimension + j] < min) 
                    min = my_grid->element[i * my_grid->dimension + j];
				 
        }
    }

    printf("AVG: %f \n", sum/(float)my_grid->num_elements);
	printf("MIN: %f \n", min);
	printf("MAX: %f \n", max);

	printf("\n");
}

/* Calculate the differences between grid elements for the various implementations. */
void compute_grid_differences(GRID_STRUCT *grid_1, GRID_STRUCT *grid_2, GRID_STRUCT *grid_3)
{
    float diff_12, diff_13;
    int dimension = grid_1->dimension;
    int num_elements = dimension*dimension;

    diff_12 = 0.0;
    diff_13 = 0.0;
    for(int i = 0; i < grid_1->dimension; i++){
        for(int j = 0; j < grid_1->dimension; j++){
            diff_12 += fabsf(grid_1->element[i * dimension + j] - grid_2->element[i * dimension + j]);
            diff_13 += fabsf(grid_1->element[i * dimension + j] - grid_3->element[i * dimension + j]);
        }
    }
    printf("Average difference in grid elements for Gauss Seidel and Red-Black methods = %f. \n", \
            diff_12/num_elements);

    printf("Average difference in grid elements for Gauss Seidel and Jacobi methods = %f. \n", \
            diff_13/num_elements);


}

/* Create a grid of random floating point values bounded by UPPER_BOUND_ON_GRID_VALUE */
void 
create_grids(GRID_STRUCT *grid_1, GRID_STRUCT *grid_2, GRID_STRUCT *grid_3)
{
	printf("Creating a grid of dimension %d x %d. \n", grid_1->dimension, grid_1->dimension);
	grid_1->element = (float *)malloc(sizeof(float) * grid_1->num_elements);
	grid_2->element = (float *)malloc(sizeof(float) * grid_2->num_elements);
	grid_3->element = (float *)malloc(sizeof(float) * grid_3->num_elements);
	srand((unsigned)time(NULL));
	
	float val;
	for(int i = 0; i < grid_1->dimension; i++)
		for(int j = 0; j < grid_1->dimension; j++){
			val =  ((float)rand()/(float)RAND_MAX) * UPPER_BOUND_ON_GRID_VALUE;
			grid_1->element[i * grid_1->dimension + j] = val; 	
			grid_2->element[i * grid_2->dimension + j] = val; 
			grid_3->element[i * grid_3->dimension + j] = val; 
			
		}
}



/* Edit this function to use the jacobi method of solving the equation. The final result should 
 * be placed in the grid_2 data structure */
int 
compute_using_pthreads_jacobi(GRID_STRUCT *grid_2)
{
	pthread_t thread_id[NUM_THREADS];
	int num_iter = 0;
	int done = 0;
	float diff;

	*grid_2_temp = *grid_2;
	
	int i; int j;
	while(!done)
	{
		diff = 0;
		barrier.counter=0;
		for(i = 0; i < NUM_THREADS; i++)
			pthread_create(&thread_id[i], NULL, compute_jacobi, grid_2);
		for(i = 0; i < NUM_THREADS; i++)
			pthread_join(thread_id[i], NULL);
	
		for( i = 1; i < (grid_2->dimension-1); i++)
            for( j = 1; j < (grid_2->dimension-1); j++)
           			diff += fabs(grid_2->element[i * grid_2->dimension + j] - grid_2_temp->element[i * grid_2_temp->dimension + j] );  
		
		printf("Iteration %d. Diff: %f. \n", num_iter, diff);
		num_iter++;
		
		if((float)diff/((float)(grid_2->dimension*grid_2->dimension)) < (float)TOLERANCE) 
			done = 1;
		
		*grid_2 = *grid_2_temp;
	}
	
	return num_iter;
}

//the compute module for the jacobi method
void 
*compute_jacobi(void *arg){
	int end;	
	GRID_STRUCT * my_grid = (GRID_STRUCT *)arg;
	int i; int j;
	
	if(barrier.counter == NUM_THREADS - 1)
		end = GRID_DIMENSION - 1;
	else 
		end = barrier.counter * n + 1 + n;
		
	for(i = barrier.counter * n + 1; i < end; i++)
		for(j = 1; j < GRID_DIMENSION - 1; j++){
			/* Apply the update rule. */	
            grid_2_temp->element[i * my_grid->dimension + j] = \
                               0.20*(my_grid->element[i * my_grid->dimension + j] + \
                                       my_grid->element[(i - 1) * my_grid->dimension + j] +\
                                       my_grid->element[(i + 1) * my_grid->dimension + j] +\
                                       my_grid->element[i * my_grid->dimension + (j + 1)] +\
                                       my_grid->element[i * my_grid->dimension + (j - 1)]);
                			
		}
	barrier_sync(&barrier);
}

/* Edit this function to use the red-black method of solving the equation. The final result 
 * should be placed in the grid_3 data structure */
int 
compute_using_pthreads_red_black(GRID_STRUCT *grid_3)
{	
	pthread_t thread_id[NUM_THREADS];
	int num_iter = 0;
	int done = 0;
	float diff;
	
	*grid_3_temp = *grid_3;
	
	int i; int j;
	while(!done)
	{
		diff = 0;
		barrier.counter = 0;
		for(i = 0; i < NUM_THREADS; i++)
			pthread_create(&thread_id[i], NULL, compute_redblack, grid_3);
		for(i = 0; i < NUM_THREADS; i++)
			pthread_join(thread_id[i], NULL);
	
		for(i = 1; i < (grid_3->dimension-1); i++)
            for(j = 1; j < (grid_3->dimension-1); j++)
           			diff += fabs(grid_3->element[i * grid_3->dimension + j] - grid_3_temp->element[i * grid_3->dimension + j] );  
		
		printf("Iteration %d. Diff: %f. \n", num_iter, diff);
		num_iter++;

		if((float)diff/((float)(grid_3->dimension*grid_3->dimension)) < (float)TOLERANCE) 
			done = 1;
		
		*grid_3 = *grid_3_temp;
	}
	
	return num_iter;	
}

//the compute module for the red black method
void
*compute_redblack(void *arg){
	int end;
	int offset;
	GRID_STRUCT * my_grid = (GRID_STRUCT *)arg;
	int i; int j;
	
	//check if the current box is red box or black box
	if (barrier.counter % 2 == 0) 
		offset = 1;
	else
		offset = 2;
	//check if this is the last thread that compute the last several lines of the matrix
	if(barrier.counter == NUM_THREADS - 1)
		end = GRID_DIMENSION - 1;
	else 
		end = barrier.counter * n + offset + n;
		
	for(i = barrier.counter * n + offset; i < end; i += 2)
		for(j = 1; j < GRID_DIMENSION - 1; j += 2){
			/* Apply the update rule. */	
            grid_3_temp->element[i * my_grid->dimension + j] = \
                               0.20*(my_grid->element[i * my_grid->dimension + j] + \
                                       my_grid->element[(i - 1) * my_grid->dimension + j] +\
                                       my_grid->element[(i + 1) * my_grid->dimension + j] +\
                                       my_grid->element[i * my_grid->dimension + (j + 1)] +\
                                       my_grid->element[i * my_grid->dimension + (j - 1)]);
                			
		}
	barrier_sync(&barrier);
}

void barrier_sync(BARRIER *barrier)
{
		  sem_wait(&(barrier->counter_sem)); // Try to obtain the lock on the counter

		  // Check if all threads before us, that is NUM_THREADS-1 threads have reached this point
		  if(barrier->counter == (NUM_THREADS - 1)){
					 barrier->counter = 0; // Reset the counter
					 sem_post(&(barrier->counter_sem)); 
					 // Signal the blocked threads that it is now safe to cross the barrier
					 for(int i = 0; i < (NUM_THREADS - 1); i++)
								sem_post(&(barrier->barrier_sem));
		  } else{
					 barrier->counter++; // Increment the counter
					 sem_post(&(barrier->counter_sem)); // Release the lock on the counter
					 sem_wait(&(barrier->barrier_sem)); // Block on the barrier semaphore and wait for someone to signal us when it is safe to cross
		  }
}



/* The main function */
int 
main(int argc, char **argv)
{	
	int num_iter;
	/*get the num of thread from the std input*/
	NUM_THREADS=strtol(argv[1],NULL,10);
	/*how many lines each thread need to compute*/
	n=floor((GRID_DIMENSION-2)/NUM_THREADS);
	
 /* Initialize the barrier data structure */
	barrier.counter = 0;
	sem_init(&barrier.counter_sem, 0, 1); // Initialize the semaphore protecting the counter to 1
	sem_init(&barrier.barrier_sem, 0, 0); // Initialize the semaphore protecting the barrier to 0

	
	/* Generate the grids and populate them with the same set of random values. */
	GRID_STRUCT *grid_1 = (GRID_STRUCT *)malloc(sizeof(GRID_STRUCT)); 
	GRID_STRUCT *grid_2 = (GRID_STRUCT *)malloc(sizeof(GRID_STRUCT)); 
	GRID_STRUCT *grid_3 = (GRID_STRUCT *)malloc(sizeof(GRID_STRUCT)); 

	grid_1->dimension = GRID_DIMENSION;
	grid_1->num_elements = grid_1->dimension * grid_1->dimension;
	grid_2->dimension = GRID_DIMENSION;
	grid_2->num_elements = grid_2->dimension * grid_2->dimension;
	grid_3->dimension = GRID_DIMENSION;
	grid_3->num_elements = grid_3->dimension * grid_3->dimension;

 	create_grids(grid_1, grid_2, grid_3);

	
	/* Compute the reference solution using the single-threaded version. */
	printf("Using the single threaded version to solve the grid. \n");
	num_iter = compute_gold(grid_1);
	printf("Convergence achieved after %d iterations. \n", num_iter);

	/* Use pthreads to solve the equation uisng the red-black parallelization technique. */
	printf("Using pthreads to solve the grid using the red-black parallelization method. \n");
	num_iter = compute_using_pthreads_red_black(grid_2);
	printf("Convergence achieved after %d iterations. \n", num_iter);

	
	/* Use pthreads to solve the equation using the jacobi method in parallel. */
	printf("Using pthreads to solve the grid using the jacobi method. \n");
	num_iter = compute_using_pthreads_jacobi(grid_3);
	printf("Convergence achieved after %d iterations. \n", num_iter);

	/* Print key statistics for the converged values. */
	printf("\n");
	printf("Reference: \n");
	print_statistics(grid_1);

	printf("Red-black: \n");
	print_statistics(grid_2);
		
	printf("Jacobi: \n");
	print_statistics(grid_3);

    /* Compute grid differences. */
    compute_grid_differences(grid_1, grid_2, grid_3);

	/* Free up the grid data structures. */
	free((void *)grid_1->element);	
	free((void *)grid_1); 
	
	free((void *)grid_2->element);	
	free((void *)grid_2);

	free((void *)grid_3->element);	
	free((void *)grid_3);

	exit(0);
}
