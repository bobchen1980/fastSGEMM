// copyright aleksandr dubinsky

#ifndef _MATRIXMUL_KERNEL_H_
#define _MATRIXMUL_KERNEL_H_


#include <stdio.h>


#define unrolled_loop_1(_)	\
	_(0)
#define unrolled_loop_2(_)	\
	_(0);	_(1)
#define unrolled_loop_4(_)	\
	_(0);	_(1);				\
	_(2);	_(3)
#define unrolled_loop_6(_)	\
	_(0);	_(1);				\
	_(2);	_(3);				\
	_(4);	_(5)
#define unrolled_loop_8(_)	\
	_(0);	_(1);				\
	_(2);	_(3);				\
	_(4);	_(5);				\
	_(6);	_(7)
#define unrolled_loop_12(_)	\
	_(0);	_(1);				\
	_(2);	_(3);				\
	_(4);	_(5);				\
	_(6);	_(7);				\
	_(8);	_(9);				\
	_(10); _(11)
#define unrolled_loop_16(_)	\
	_(0);	_(1);				\
	_(2);	_(3);				\
	_(4);	_(5);				\
	_(6);	_(7);				\
	_(8);	_(9);				\
	_(10); _(11);				\
	_(12); _(13);				\
	_(14); _(15)

#define unrolled_loop(_)\
	unrolled_loop_8(_)




texture<float2, 1, cudaReadModeElementType> tex_A;
texture<float2, 1, cudaReadModeElementType> tex_B;


__global__ void
matrixMul_opt (float* A, float* B, float* C, int wB, int wAhB)
{
    __shared__ float Atemp[BLOCK_SIZE_A][BLOCK_SIZE_K];
	__shared__ float Btemp[BLOCK_SIZE_K];//[BLOCK_SIZE_B];

	#define op_decl(__)\
		float out##__##0 = 0.0f;\
		float out##__##1 = 0.0f
	unrolled_loop (op_decl);

	for (int k_slow =0; k_slow < wAhB; k_slow +=BLOCK_SIZE_K) {
		
		//* Load part of the matrix A into shared memory;
        __syncthreads();
		if(threadIdx.x*OUTPUTS_X < BLOCK_SIZE_K){ // However, don't do it if i'm an "extra thread." This happens when wB > wAhB. Note: wB<wAhB causes a crash.
        	for(int y=0; y<OUTPUTS_Y; y++){
				int Atemp_y = threadIdx.y*OUTPUTS_Y + y;
				int Atemp_x = threadIdx.x*OUTPUTS_X;

				int A_y = blockIdx.y*BLOCK_SIZE_A + Atemp_y;
				int A_x = k_slow + Atemp_x;
				
				*((float2*)&Atemp[Atemp_y][Atemp_x]) = *((float2*)&A[A_y * wAhB + A_x]);
			}
		}
        __syncthreads();
		//*/

		//* Perform calculations
		float a;
		int B_y, B_x;
		float2 b2, b1;

		for (int k=0; k<BLOCK_SIZE_K; k+=1){
			/*
			B_y = k;
			B_x = OUTPUTS_X*threadIdx.x;
			//b1 = *((float2*)&(Btemp[B_y][B_x]));
			b1 = *((float2*)&(Btemp[(B_y + B_x)%BLOCK_SIZE_K]));
			//b1.x = Btemp[(B_y + B_x)%BLOCK_SIZE_K];
			//b1.y = Btemp[(B_y + B_x+64)%BLOCK_SIZE_K];
			/*/
			B_y = k_slow + k;
			B_x = BLOCK_SIZE_B*blockIdx.x + OUTPUTS_X*threadIdx.x;
			b1 = *((float2*)&(B[B_y*wB + B_x]));
			//b1.x = b1.y = B[B_y*wB + B_x -threadIdx.x];
			//b1 = tex1Dfetch(tex_B, (B_y*wB + B_x)/OUTPUTS_X);
			//b1.x = 1.0f; b1.y = 1.0f;
			//*/

			#define op_compute1(__)\
				a = Atemp[threadIdx.y*OUTPUTS_Y + __][k];\
				out##__##0 += b1.x * a;\
				out##__##1 += b1.y * a
			unrolled_loop (op_compute1);
		}
		/*/
		float a;
		int B_y = k_slow + 0;
		int B_x = BLOCK_SIZE_B*blockIdx.x + OUTPUTS_X*threadIdx.x;
		float2 b2, b1 = *((float2*)&(B[B_y*wB + B_x]));

		for (int k=0; k<BLOCK_SIZE_K-2; k+=2){
			B_y = k_slow + k+1;
			B_x = BLOCK_SIZE_B*blockIdx.x + OUTPUTS_X*threadIdx.x;
			b2 = *((float2*)&(B[B_y*wB + B_x]));

			#define op_compute1(__)\
				a = Atemp[threadIdx.y*OUTPUTS_Y + __][k];\
				out##__##0 += b1.x * a;\
				out##__##1 += b1.y * a
			unrolled_loop (op_compute1);


			B_y = k_slow + k+2;
			B_x = BLOCK_SIZE_B*blockIdx.x + OUTPUTS_X*threadIdx.x;
			b1 = *((float2*)&(B[B_y*wB + B_x]));

			#define op_compute2(__)\
				a = Atemp[threadIdx.y*OUTPUTS_Y + __][k+1];\
				out##__##0 += b2.x * a;\
				out##__##1 += b2.y * a
			unrolled_loop (op_compute2);
		}

		B_y = k_slow + BLOCK_SIZE_K-1;
		B_x = BLOCK_SIZE_B*blockIdx.x + OUTPUTS_X*threadIdx.x;
		b2 = *((float2*)&(B[B_y*wB + B_x]));

		#define op_compute3(__)\
			a = Atemp[threadIdx.y*OUTPUTS_Y + __][BLOCK_SIZE_K-2];\
			out##__##0 += b1.x * a;\
			out##__##1 += b1.y * a
		unrolled_loop (op_compute3);


		#define op_compute4(__)\
			a = Atemp[threadIdx.y*OUTPUTS_Y + __][BLOCK_SIZE_K-1];\
			out##__##0 += b2.x * a;\
			out##__##1 += b2.y * a
		unrolled_loop (op_compute4);
		//*/
    }


    // Write the block sub-matrix to device memory;
	int C_x, C_y;
	#define op_save(__)																\
    	C_y = blockIdx.y*BLOCK_SIZE_A + threadIdx.y*OUTPUTS_Y + __;					\
		C_x = blockIdx.x*BLOCK_SIZE_B + threadIdx.x*OUTPUTS_X;						\
		C[C_y*wB + C_x] = out##__##0;												\
		C[C_y*wB + C_x+1] = out##__##1
	unrolled_loop (op_save);
}


#endif // #ifndef _MATRIXMUL_KERNEL_H_
