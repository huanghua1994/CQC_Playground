#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <time.h>
#include <omp.h>

#include <mkl.h>

#define MAX(a, b) ((a)>(b))?(a):(b)
#define MIN(a, b) ((a)<(b))?(a):(b)

// Parallel Jacobi method for eigen decompisition
// Input parameters:
//   A       : Symmetric matrix to be decomposed, Row-major, 
//             size >= ldA * nrow, will be overwritten when exit
//   nrow    : Number of rows and columns
//   ldA     : Leading dimension of A, size >= nrow
//   ldV     : Leading dimension of V, size >= nrow
//   nthread : Number of threads
//   workbuf : Working buffer, size >= nthread * 4 * nrow + nrow
// Output parameters:
//   V : Eigenvectors, each row is an eigenvector, size >= ldV * nrow
//   D : Array, size >= nrow
void eig_jacobi_onesided(
    double *A, const int nrow, const int ldA,
    double *V, const int ldV, double *D,
    const int nthread, double *workbuf
)
{
    const int ncol = nrow;
    const int semi_n = nrow / 2;
    int *top = (int*) &workbuf[nthread * 4 * nrow];
    int *bot = top + semi_n;
    for (int i = 0; i < semi_n; i++)
    {
        top[i] = 2 * i;
        bot[i] = 2 * i + 1;
    }
    
    memset(D, 0, sizeof(double) * nrow);
    memset(V, 0, sizeof(double) * ldV * nrow);
    for (int i = 0; i < nrow; i++)
        V[i * ldV + i] = 1.0;
    
    double A_2norm = 0.0, D_2norm = 0.0, relres_norm;
    #pragma omp parallel for num_threads(nthread) reduction(+:A_2norm)
    for (int i = 0; i < nrow; i++)
    {
        double tmp = 0.0;
        double *A_row = A + i * ldA;
        for (int j = 0; j < ncol; j++)
            tmp += A_row[j] * A_row[j];
        A_2norm += tmp;
    }
    A_2norm = sqrt(A_2norm);
    
    // Conceptually, G = V' * A
    double *G = A;
    const int ldG = ldA;
    int sweep = 0;
    size_t row_msize = sizeof(double) * nrow;
    relres_norm = fabs(A_2norm - D_2norm) / A_2norm; 
    while (relres_norm > 1e-14)
    {
        double st = omp_get_wtime();
        for (int subsweep = 0; subsweep < nrow - 1; subsweep++)
        {
            #pragma omp parallel num_threads(nthread)
            {
                int tid = omp_get_thread_num();
                double *thread_buf = workbuf + tid * 4 * nrow;
                double *Gp = thread_buf + 0 * nrow;
                double *Gq = thread_buf + 1 * nrow;
                double *Vp = thread_buf + 2 * nrow;
                double *Vq = thread_buf + 3 * nrow;
                
                #pragma omp for schedule(guided)
                for (int k = 0; k < semi_n; k++)
                {
                    // Choose such that p < q
                    int p = MIN(top[k], bot[k]);
                    int q = MAX(top[k], bot[k]);
                    
                    double *Gp_ = G + p * ldG;
                    double *Gq_ = G + q * ldG;
                    double *Vp_ = V + p * ldV;
                    double *Vq_ = V + q * ldV;
                    memcpy(Gp, Gp_, row_msize);
                    memcpy(Gq, Gq_, row_msize);
                    memcpy(Vp, Vp_, row_msize);
                    memcpy(Vq, Vq_, row_msize);
                    
                    // Calculate block
                    double App = 0.0, Aqq = 0.0, Apq = 0.0;
                    #pragma omp simd
                    for (int l = 0; l < nrow; l++)
                    {
                        App += Gp[l] * Vp[l];
                        Aqq += Gq[l] * Vq[l];
                        Apq += Gp[l] * Vq[l];
                    }
                    
                    // Calculate J = [c s;-s c] such that J' * Apq * J = diagonal
                    // [c s] = symschur2([app apq; apq aqq]);
                    double c, s, tau, t;
                    if (Apq == 0)
                    {
                        c = 1.0; s = 0.0;
                    } else {
                        tau = (Aqq - App) / (2.0 * Apq);
                        if (tau > 0) t =  1.0 / ( tau + sqrt(1.0 + tau * tau));
                        else         t = -1.0 / (-tau + sqrt(1.0 + tau * tau));
                        c = 1.0 / sqrt(1 + t * t);
                        s = t * c;
                    }
                    
                    // Update G by applying J' on left
                    // Update V by applying J on right
                    #pragma omp simd
                    for (int l = 0; l < nrow; l++)
                    {
                        Gp_[l] = c * Gp[l] - s * Gq[l];
                        Gq_[l] = s * Gp[l] + c * Gq[l];
                        Vp_[l] = c * Vp[l] - s * Vq[l];
                        Vq_[l] = s * Vp[l] + c * Vq[l];
                    }
                }  // End of k loop
            }  // End of pragma omp parallel
            
            // [top, bot] = music(top, bot)
            int top_tail = top[semi_n - 1];
            int bot_head = bot[0];
            for (int l = semi_n - 1; l >= 2; l--) top[l] = top[l - 1];
            for (int l = 0; l < semi_n; l++) bot[l] = bot[l + 1];
            top[1] = bot_head;
            bot[semi_n - 1] = top_tail;
        }  // End of subsweep loop
        
        D_2norm = 0.0;
        for (int k = 0; k < nrow; k++)
        {
            double tmp = 0.0;
            double *Gk = G + k * ldG;
            double *Vk = V + k * ldV;
            for (int l = 0; l < nrow; l++) tmp += Gk[l] * Vk[l];
            D[k] = tmp;
            D_2norm += tmp * tmp;
        }
        D_2norm = sqrt(D_2norm);
        relres_norm = fabs(A_2norm - D_2norm) / A_2norm; 
        
        double ut = omp_get_wtime() - st;
        printf("Jacobi sweep %2d: %e %.3lf\n", ++sweep, relres_norm, ut);
        if (sweep >= 20) break;
    }  // End of while (relres_norm > 1e-14) loop 
}

void test_mkl_qr(double *A, double *A0, double *tau, const int n)
{
    // Warm up
    memcpy(A, A0, sizeof(double) * n * n);
    LAPACKE_dgeqrfp(LAPACK_ROW_MAJOR, n, n, A, n, tau);
    
    for (int i = 0; i < 10; i++)
    {
        memcpy(A, A0, sizeof(double) * n * n);
        double st = omp_get_wtime();
        LAPACKE_dgeqrfp(LAPACK_ROW_MAJOR, n, n, A, n, tau);
        double ut = omp_get_wtime() - st;
        printf("LAPACKE_dgeqrfp %2d: %.3lf (s)\n", i, ut);
    }
}

void test_mkl_eig(double *A, double *A0, double *eigval, const int n)
{
    // Warm up
    memcpy(A, A0, sizeof(double) * n * n);
    LAPACKE_dsyev(LAPACK_ROW_MAJOR, 'V', 'U', n, A, n, eigval);
    
    for (int i = 0; i < 10; i++)
    {
        memcpy(A, A0, sizeof(double) * n * n);
        double st = omp_get_wtime();
        LAPACKE_dsyev(LAPACK_ROW_MAJOR, 'V', 'U', n, A, n, eigval);
        double ut = omp_get_wtime() - st;
        printf("LAPACKE_dsyev %2d: %.3lf (s)\n", i, ut);
    }
}

int main(int argc, char **argv)
{
    int n = atoi(argv[1]);
    int nthread = omp_get_max_threads();
    
    double *A  = (double*) malloc(sizeof(double) * n * n);
    double *A0 = (double*) malloc(sizeof(double) * n * n);
    double *V  = (double*) malloc(sizeof(double) * n * n);
    double *D  = (double*) malloc(sizeof(double) * n);
    double *workbuf = (double*) malloc(sizeof(double) * (nthread * 4 * n + n));
    assert(A != NULL && A0 != NULL && V != NULL);
    assert(D != NULL && workbuf != NULL);
    
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
    eig_jacobi_onesided(A, n, n, V, n, D, nthread, workbuf);
    
    test_mkl_qr (A, A0, workbuf, n);
    test_mkl_eig(A, A0, workbuf, n);
    
    free(A);
    free(A0);
    free(V);
    free(D);
    free(workbuf);
}