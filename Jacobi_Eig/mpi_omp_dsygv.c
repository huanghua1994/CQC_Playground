#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <mkl.h>
#include <mpi.h>

static void generate_AB(const int n, double *A, double *B, double *BV1, double *BV2, double *BD)
{
    // Make A B symmetric
    for (int i = 0; i < n; i++)
    {
        for (int j = i; j < n; j++)
        {
            double val0 = drand48() + 1.0;
            double val1 = drand48() + 1.0;
            A[i * n + j] = val0;
            A[j * n + i] = val0;
            B[i * n + j] = val1;
            B[j * n + i] = val1;
        }
    }
    
    // Make B SPD
    LAPACKE_dsyev(LAPACK_ROW_MAJOR, 'V', 'U', n, B, n, BD);
    memcpy(BV1, B, sizeof(double) * n * n);
    // Make all eigenvalues positive and right multiple D to V
    for (int i = 0; i < n; i++) BD[i] = fabs(BD[i]) + 0.01;
    for (int i = 0; i < n; i++)
    {
        double *BV1_row = BV1 + i * n;
        double *BV2_row = BV2 + i * n;
        for (int j = 0; j < n; j++) BV2_row[j] = BD[j] * BV1_row[j];
    }
    // Form new B
    cblas_dgemm(
        CblasRowMajor, CblasNoTrans, CblasTrans, n, n, n,
        1.0, BV1, n, BV2, n, 0.0, B, n
    );
}

static void print_double_mat(double *mat, const int ldm, const int nrows, const int ncols, const char *mat_name)
{
    printf("%s:\n", mat_name);
    for (int i = 0; i < nrows; i++)
    {
        for (int j = 0; j < ncols; j++) printf("% .3lf ", mat[i * ldm + j]);
        printf("\n");
    }
    printf("\n");
}

void dsygv_decompose(const int n, double *A, double *B, double *D)
{
    double st, ut;
    
    st = omp_get_wtime();
    // B = chol(B, 'lower'), output B is lower triangle matrix
    LAPACKE_dpotrf(LAPACK_ROW_MAJOR, 'L', n, B, n);
    ut = (omp_get_wtime() - st) * 1000.0;
    printf("Calc B = L * L^T         used %.3lf ms\n", ut);
    
    
    st = omp_get_wtime();
    // B = inv(B), input & output B is lower triangle matrix
    LAPACKE_dtrtri(LAPACK_ROW_MAJOR, 'L', 'N', n, B, n);
    ut = (omp_get_wtime() - st) * 1000.0;
    printf("Calc L^-1                used %.3lf ms\n", ut);
    
    st = omp_get_wtime();
    // A = A * B', input B is lower triangle matrix
    cblas_dtrmm(
        CblasRowMajor, CblasRight, CblasLower, CblasTrans,
        CblasNonUnit, n, n, 1.0, B, n, A, n
    );
    // A = B * A, input B is lower triangle matrix
    cblas_dtrmm(
        CblasRowMajor, CblasLeft, CblasLower, CblasNoTrans,
        CblasNonUnit, n, n, 1.0, B, n, A, n
    );
    ut = (omp_get_wtime() - st) * 1000.0;
    printf("Calc C = L^-1 * A * L^-T used %.3lf ms\n", ut);
    
    st = omp_get_wtime();
    LAPACKE_dsyev(LAPACK_ROW_MAJOR, 'V', 'U', n, A, n, D);
    ut = (omp_get_wtime() - st) * 1000.0;
    printf("Calc C = V * D * V'      used %.3lf ms\n", ut);
    
    st = omp_get_wtime();
    cblas_dtrmm(
        CblasRowMajor, CblasLeft, CblasLower, CblasTrans,
        CblasNonUnit, n, n, 1.0, B, n, A, n
    );
    ut = (omp_get_wtime() - st) * 1000.0;
    printf("Calc Z = L^-T * V        used %.3lf ms\n", ut);
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    
    int rank, nproc;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    int n = 1000;
    if (argc >= 2) n = atoi(argv[1]);
    if (rank == 0) printf("Matrix size = %d\n", n);
    
    srand48(114514);
    
    int my_nthreads = 1;
    char *env_ntheads = getenv("NTHREADS"); 
    if (env_ntheads != NULL) my_nthreads = atoi(env_ntheads);
    if (my_nthreads < 1) my_nthreads = 1;
    
    if (rank == 0)
    {
        int save = mkl_get_max_threads();
        mkl_set_dynamic(0);
        mkl_set_num_threads(my_nthreads);
        
        size_t mat_msize = sizeof(double) * n * n;
        double *A, *B, *BV1, *BV2, *BD;
        A   = (double*) malloc(mat_msize);
        B   = (double*) malloc(mat_msize);
        BV1 = (double*) malloc(mat_msize);
        BV2 = (double*) malloc(mat_msize);
        BD  = (double*) malloc(mat_msize);
        generate_AB(n, A, B, BV1, BV2, BD);
        
        double *A0 = BV1, *B0 = BV2;
        double *D_dsygv = BD, *D_dsyev = BD + n;
        memcpy(A0, A, mat_msize);
        memcpy(B0, B, mat_msize);
    
        memcpy(A, A0, mat_msize);
        memcpy(B, B0, mat_msize);
        LAPACKE_dsygv(LAPACK_ROW_MAJOR, 1, 'V', 'U', n, A, n, B, n, D_dsygv);
        //print_double_mat(A, n, n, n, "Z from dsygv");
        //print_double_mat(D_dsygv, n, 1, n, "D from dsygv");
        for (int i = 0; i < 5; i++)
        {
            memcpy(A, A0, mat_msize);
            memcpy(B, B0, mat_msize);
            double st = omp_get_wtime();
            LAPACKE_dsygv(LAPACK_ROW_MAJOR, 1, 'V', 'U', n, A, n, B, n, D_dsygv);
            double ut = (omp_get_wtime() - st) * 1000.0;
            printf("LAPACKE_dsygv use %.3lf ms\n", ut);
        }
        
        printf("\n\n");
        
        memcpy(A, A0, mat_msize);
        memcpy(B, B0, mat_msize);
        dsygv_decompose(n, A, B, D_dsyev);
        //print_double_mat(A, n, n, n, "Z from dsygv_decompose");
        //print_double_mat(D_dsyev, n, 1, n, "D from dsygv_decompose");
        for (int i = 0; i < 5; i++)
        {
            memcpy(A, A0, mat_msize);
            memcpy(B, B0, mat_msize);
            double st = omp_get_wtime();
            dsygv_decompose(n, A, B, D_dsyev);
            double ut = (omp_get_wtime() - st) * 1000.0;
            printf("dsygv_decompose use %.3lf ms\n\n", ut);
        }
        
        free(A);
        free(B);
        free(BV1);
        free(BV2);
        free(BD);
        
        mkl_set_dynamic(1);
        mkl_set_num_threads(1);
    }
    
    
    int flag;
    MPI_Request req;
    MPI_Status status;
    MPI_Ibarrier(MPI_COMM_WORLD, &req);
    MPI_Test(&req, &flag, &status);
    while (!flag)
    {
        usleep(10000); // in micro-seconds
        MPI_Test(&req, &flag, &status);
    }
    
    MPI_Finalize(); 
}