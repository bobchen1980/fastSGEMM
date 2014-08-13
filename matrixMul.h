
#ifndef _MATRIXMUL_H_
#define _MATRIXMUL_H_

//                      b
//               **************
//               **************
//               **************
//               **************
//             c ****MTRX B****
//               **************
//               **************
//               **************
//               **************
//       c
//   *********   **************
// a **MTX A**   ****MTRX C**** a
//   *********   **************
//                     b
//   * == element of a matrix
//
// notes:
//   BLOCK_SIZE_A must be a multiple of OUTPUTS_PER_THREAD
//   shared memory must be large enough for  BLOCK_SIZE_A * BLOCK_SIZE_C *sizeof(float)
//   number of threads will be BLOCK_SIZE_A * BLOCK_SIZE_B /OUTPUTS_PER_THREAD
//   register count per thread is OUTPUTS_PER_THREAD + 


#define OUTPUTS_X		2 //changing this requires rewriting matrixMul_opt()
#define OUTPUTS_Y		8 //this can be changed, but must be manually kept in sync with unrolled_loop() macro

#define BLOCK_SIZE_A	OUTPUTS_Y

/*/
#define BLOCK_SIZE_B	128
#define BLOCK_SIZE_K	128
/*/
#define BLOCK_SIZE_B	(128+128)  //assert BLOCKSIZE_B >= BLOCK_SIZE_K
#define BLOCK_SIZE_K	(128+64)   //here is best performance
//*/

#endif // _MATRIXMUL_H_

// 62.8
// 61.7
//

/*
at 4x1. When 256 B,K, 16 GFLOPS but at 192, 20 !!!
	?? Is B fitting into some sort of cache???
	!! no, more blocks can fit into one multiprocessor

*/
