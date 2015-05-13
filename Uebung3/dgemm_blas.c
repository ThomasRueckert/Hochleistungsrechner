#define min(x,y) (((x) < (y)) ? (x) : (y))

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>

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

  if (gettimeofday (&begin, (struct timezone *) 0))
    {
      fprintf (stderr, "ERROR: gettimeofday() failed!\n");
      exit (1);
    }

  dgemm_ (&trans, &trans, &m, &n, &k, &alpha, A, &k, B, &n, &beta, C, &n);

  if (gettimeofday (&end, (struct timezone *) 0))
    {
      fprintf (stderr, "ERROR: gettimeofday() failed!\n");
      exit (1);
    }

  float msecPerMatrixMul =
    ((end.tv_sec - begin.tv_sec) * 1000.0 +
     ((end.tv_usec - begin.tv_usec) / 1000.0));
  double flopsPerMatrixMul = 2.0 * (double) m * (double) k * (double) n;
  double gigaFlops =
    (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);

  printf ("%d %.2f\n", m, gigaFlops);

  free (A);
  free (B);
  free (C);

  return 0;
}
