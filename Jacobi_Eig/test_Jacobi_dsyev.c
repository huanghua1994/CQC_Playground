#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include <mkl.h>

#include "Jacobi_dsyev.h"

void test_mkl_eig(double *A, double *A0, double *eigval, const int n)
{
    // Warm up
    memcpy(A, A0, sizeof(double) * n * n);
    LAPACKE_dsyev(LAPACK_ROW_MAJOR, 'V', 'U', n, A, n, eigval);
    
    for (int i = 0; i < 5; i++)
    {
        memcpy(A, A0, sizeof(double) * n * n);
        double st = omp_get_wtime();
        LAPACKE_dsyev(LAPACK_ROW_MAJOR, 'V', 'U', n, A, n, eigval);
        double ut = omp_get_wtime() - st;
        printf("LAPACKE_dsyev %2d: %.1lf ms\n", i, ut * 1e3);
    }
}

int main(int argc, char **argv)
{
    int n = atoi(argv[1]);
    int nthread = omp_get_max_threads();
    
    size_t mat_msize = sizeof(double) * n * n;
    double *A   = (double*) malloc(mat_msize);
    double *A0  = (double*) malloc(mat_msize);
    double *VT  = (double*) malloc(mat_msize);
    double *D   = (double*) malloc(mat_msize);
    double *DVT = (double*) malloc(mat_msize);
    double *A1  = (double*) malloc(mat_msize);
    assert(A != NULL && A0  != NULL && VT != NULL);
    assert(D != NULL && DVT != NULL && A1 != NULL);
    
    srand48(time(NULL));
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < i; j++)
        {
            double val = drand48() + 1.0;
            A0[i * n + j] = val;
            A0[j * n + i] = val;
        }
        A0[i * n + i] += 2.0;
    }
    
    memcpy(A, A0, sizeof(double) * n * n);
    Jacobi_dsyev(n, A, n, VT, n, D, 15, 1e-14, nthread);
    // Verify Jacobi method's result
    for (int i = 0; i < n; i++)
    {
        double *VT_i  = VT  + i * n;
        double *DVT_i = DVT + i * n;
        for (int j = 0; j < n; j++) DVT_i[j] = D[i] * VT_i[j];
    }
    cblas_dgemm(
        CblasRowMajor, CblasTrans, CblasNoTrans, n, n, n,
        1.0, VT, n, DVT, n, 0.0, A1, n
    );
    double A_2norm = 0.0, rel_norm = 0.0;
    for (int i = 0; i < n * n; i++)
    {
        double diff = A0[i] - A1[i];
        A_2norm  += A0[i] * A0[i];
        rel_norm += diff * diff;
    }
    A_2norm  = sqrt(A_2norm);
    rel_norm = sqrt(rel_norm);
    rel_norm /= A_2norm;
    printf("Jacobi ||V * D * V' - A||_fro / ||A||_fro = %e\n", rel_norm);
    
    test_mkl_eig(A, A0, D, n);
    
    free(A);
    free(A0);
    free(VT);
    free(D);
    free(DVT);
    free(A1);
}