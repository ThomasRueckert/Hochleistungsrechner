#define min(x,y) (((x) < (y)) ? (x) : (y))

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include <cublas_v2.h>

extern void dgemm_ (char *transa, char *transb,
		    int *m, int *n, int *k, double *alpha,
		    double *A, int *lda,
		    double *B, int *ldb, double *beta, double *C, int *ldc);

int
main (int argc, char **argv)
{
  double *A, *B, *C;
  int m, n, k, i;
  double alpha, beta;
  char trans = 'N';
  struct timeval begin, end;

  m = k = n = 8192;
  if (argc == 2)
    {
      m = k = n = atoi (argv[1]);
    }
  alpha = 1.0;
  beta = 0.0;

  A = (double *) malloc (m * k * sizeof (double));
   B = (double *) malloc (k * n * sizeof (double));
   C = (double *) malloc (m * n * sizeof (double));
   double *d_A, *d_B, *d_C;
  cudaError_t initError = cudaMalloc ((void **) &d_A, m * k * sizeof (double));
  if (initError != cudaSuccess) {
	  printf("initError");
	  return;
  }
  cudaMalloc ((void **) &d_B, k * n * sizeof (double));
  cudaMalloc ((void **) &d_C, m * n * sizeof (double));

  if (A == NULL || B == NULL || C == NULL)
    {
      printf
	("\n ERROR: Can't allocate memory for matrices. Aborting... \n\n");
      free (A);
      free (B);
      free (C);
      return 1;
    }

  for (i = 0; i < (m * k); i++)
    {
      A[i] = (double) (i + 1);
    }

  for (i = 0; i < (k * n); i++)
    {
      B[i] = (double) (-i - 1);
    }

  for (i = 0; i < (m * n); i++)
    {
      C[i] = 0.0;
    }

  cudaError_t cpyerror = cudaMemcpy(d_A, A, m * k * sizeof (double), cudaMemcpyHostToDevice);
  if (cpyerror != cudaSuccess) {
	  printf("copyerror");return;
  }
  cpyerror = cudaMemcpy(d_B, B, k * n * sizeof (double), cudaMemcpyHostToDevice);
  cpyerror = cudaMemcpy(d_C, C, m * n * sizeof (double), cudaMemcpyHostToDevice);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	/**
	* dgemm -> cublas
	*/
	//dgemm_ (&trans, &trans, &m, &n, &k, &alpha, A, &k, B, &n, &beta, C, &n);
	cublasHandle_t handle;
	cublasCreate(&handle);
	cudaError_t error = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, d_A, k, d_B, n, &beta, d_C, n);
	cublasDestroy(handle);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float time;
	cudaEventElapsedTime(&time, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	if (error != cudaSuccess) {
		printf ("error");
		return;
	}

  float msecPerMatrixMul = time;
  double flopsPerMatrixMul = 2.0 * (double) m * (double) k * (double) n;
  double gigaFlops =  (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);

  printf ("%d %.2f\n", m, gigaFlops);

  free (A);
  free (B);
  free (C);

  return 0;
}
