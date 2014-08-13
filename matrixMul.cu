// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <windows.h>

// includes, project
#include <cutil.h>

// includes, kernels
#include "matrixMul.h"
#include "matrixMul_kernel.cu"


void runTest(int argc, char** argv);
void randomInit(float*, int);
void nonrandomInit(float*, int);
void printDiff(float*, float*, int, int);

extern "C"
void computeGold(float*, const float*, const float*, unsigned int, unsigned int, unsigned int);



int
main(int argc, char** argv)
{
    runTest(argc, argv);

    CUT_EXIT(argc, argv);
}



void
runTest(int argc, char** argv)
{
	
	//int hA = 8*BLOCK_SIZE_A;
	//int wB = 4*BLOCK_SIZE_B;
	//int wAhB = 4*BLOCK_SIZE_K;

	//int hA = 16*BLOCK_SIZE_A ;     /*height of matrix A rows A(hA*wA)*/
	//int wB = 16*BLOCK_SIZE_B ;     /*width of matrix B columns*/
	//int wAhB = 64*BLOCK_SIZE_K;   /*shared dimension*/

	int hA = 16386;     /*height of matrix A(hA*wA)*/  
	int wB = 512 ;     /*width of matrix B*/
	int wAhB =hA;   /*shared dimension*/

	
    srand(2007);


    CUT_DEVICE_INIT();

	int size_A = hA*wAhB;
	int size_B = wAhB*wB;
	int size_C = hA*wB;
	int mem_size_A = size_A *sizeof(float);
	int mem_size_B = size_B *sizeof(float);
	int mem_size_C = size_C *sizeof(float);

    // allocate host memory for matrices A and B
    float* h_A = (float*) malloc (mem_size_A);
    float* h_B = (float*) malloc (mem_size_B);
    float* h_C = (float*) malloc (mem_size_C);

    // initialize host memory
    /*/
	randomInit(h_A, size_A);/*/nonrandomInit(h_A, size_A);//*/
    /*/
	randomInit(h_B, size_B);/*/nonrandomInit(h_B, size_B);//*/
    /*/
	randomInit(h_C, size_C);/*/nonrandomInit(h_C, size_C);//*/

    // allocate device memory
    float* d_A;
    CUDA_SAFE_CALL(cudaMalloc((void**) &d_A, mem_size_A));
    float* d_B;
    CUDA_SAFE_CALL(cudaMalloc((void**) &d_B, mem_size_B));
    float* d_C;
    CUDA_SAFE_CALL(cudaMalloc((void**) &d_C, mem_size_C));
	
    // copy host memory to device
    CUDA_SAFE_CALL(cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice) );
    CUDA_SAFE_CALL(cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice) );


	//bind textures
	//CUDA_SAFE_CALL(cudaBindTexture(0, tex_A, d_A, mem_size_A));
	CUDA_SAFE_CALL(cudaBindTexture(0, tex_B, d_B, mem_size_B));
    
	//set kernel parameters
	dim3 threads(BLOCK_SIZE_B/OUTPUTS_X, BLOCK_SIZE_A/OUTPUTS_Y);
	dim3 grid(wB/BLOCK_SIZE_B, hA/BLOCK_SIZE_A);


	//warmup
		matrixMul_opt<<<grid, threads>>>(d_A, d_B, d_C, wB, wAhB);
		cudaThreadSynchronize();

    //create and start timer
		unsigned int timer = 0;
		cutCreateTimer(&timer);
		cutStartTimer(timer);
	//execute
		double const ops = 2.0e11;
		int const reps = (int)(ops/(2.0*hA*wB*wAhB));
		for(int i=0; i< reps; i++)
			matrixMul_opt<<<grid, threads>>>(d_A, d_B, d_C, wB, wAhB);
		cudaThreadSynchronize();
    //stop and destroy timer
		cutStopTimer(timer);
		float duration = cutGetTimerValue(timer);
		cutDeleteTimer(timer);
		printf("\n");
		printf("Problem size:  %d MB  %.1f Billion calculations\n", (mem_size_A+mem_size_B+mem_size_C)/1024/1024, (2.0*hA*wB*wAhB)/1e9);
		printf("Iterations: %d\n", reps);
		printf("Total processing time: %f (ms) \n", duration );
		printf("Average processing time: %f (ms) \n", duration / reps );
		printf("Performance: %f (GFLOPS) \n", (2.*hA*wB*wAhB)/duration/1e6 *reps );


	printf("\n");
	/*int col = 4;
	for(int i=0; i< 3; i++)
		printf("h_C[%i][%i] = %f\n", i, col, h_C[i*wB+col]);
	for(int i=0; i< 3; i++)
		printf("h_C[%i][%i] = %f\n", i, col+1, h_C[i*wB+col+1]);*/
	if (0) {
		//copy result from device to host
		CUDA_SAFE_CALL(cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost) );
		double start = clock();
		// compute reference solution
		float* reference = (float*) malloc(mem_size_C);
		computeGold(reference, h_A, h_B, hA, wAhB, wB);
		double end = clock();
		//for(int i=0; i< 3; i++)
		//	printf("ref[%i][%i] = %f\n", i, col, reference[i*wB+col]);
		printf("%f ms, %fGflops\n",end-start,(2.*hA*wB*wAhB)/(end-start)/1e6  );
		// check result
		CUTBoolean res = cutCompareL2fe(reference, h_C, size_C, 1e-6f);
		printf("\nTest %s \n", (1 == res) ? "PASSED" : "FAILED");
		//if (res!=1) printDiff(reference, h_C, wB, hA);
		
		free(reference);
	}



    // clean up memory
    CUDA_SAFE_CALL(cudaFree(d_A));
    CUDA_SAFE_CALL(cudaFree(d_B));
    CUDA_SAFE_CALL(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C);
}

// Allocates a matrix with random float entries.
void randomInit(float* data, int size)
{
    for (int i = 0; i < size; ++i)
        data[i] = rand() / (float)RAND_MAX;
		//data[i] = 1.0f;
}
void nonrandomInit(float* data, int size)
{
    for (int i = 0; i < size; ++i)
        data[i] = 1.0f;
}

void printDiff(float *data1, float *data2, int width, int height)
{
  int x, y, k;
  int error_count=0;
  for (y=0; y<height; y++) {
    for (x=0; x<width; x++) {
      k = y*width+x;
      //if (data1[k] - data2[k] > 1e-2f) {
         printf("diff(%d,%d) CPU=%4.4f, GPU=%4.4f \n", x,y, data1[k], data2[k]);
		 // printf("diff(%d,%d) \t", x,j);
         error_count++;
      //}
    }
	 printf("\n");
  }
  printf("\nTotal Errors = %d \n", error_count);
}

