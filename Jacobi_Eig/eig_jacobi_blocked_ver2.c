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

// Perform a Jacobi roration B = J^T * A * J where J = J(p, q, theta)
// Input parameters:
//   nrow   : Number of rows and columns of A
//   p, q   : Jacobi rotation index pair
//   G      : == V' * A, row-major, size >= ldA * nrow
//   ldG    : Leading dimension of G
//   V      : Culmulated Jacobi rotation matrix
//   ldV    : Leading dimension of V
// Output parameters:
//   G, V : The p-th and q-th rows will be updated
void jacobi_rotation_kernel(
    const int nrow, const int p, const int q, 
    double *G, const int ldG, double *V, const int ldV
)
{
    double *Gp = G + p * ldG;
    double *Gq = G + q * ldG;
    double *Vp = V + p * ldV;
    double *Vq = V + q * ldV;
    
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
        double Gpl = Gp[l], Gql = Gq[l];
        double Vpl = Vp[l], Vql = Vq[l];
        Gp[l] = c * Gpl - s * Gql;
        Gq[l] = s * Gpl + c * Gql;
        Vp[l] = c * Vpl - s * Vql;
        Vq[l] = s * Vpl + c * Vql;
    }
}

// Perform a "sweep" for a block and ignore lower triangle elements 
// Input parameters:
//   nrow      : Number of rows and columns of A
//   G         : == V' * A, will be overwritten when exit, size nrow * nrow
//   blk_s_row : Start row of the block 
//   blk_n_row : Number of rows of the block
//   blk_s_col : Start column of the block 
//   blk_n_col : Number of columns of the block
// Output parameters:
//   V : Culmulated Jacobi rotation matrix, size nrow * nrow
void jacobi_subblock_sweep(
    const int nrow, double *G, double *V, 
    const int blk_s_row, const int blk_n_row,
    const int blk_s_col, const int blk_n_col
)
{
    const int ldG = nrow, ldV = nrow;
    const int blk_e_row = blk_s_row + blk_n_row;
    const int blk_e_col = blk_s_col + blk_n_col;
    
    memset(V, 0, sizeof(double) * ldV * nrow);
    for (int i = 0; i < nrow; i++) V[i * ldV + i] = 1.0;
    
    for (int p = blk_s_row; p < blk_e_row; p++)
    {
        for (int q = blk_s_col; q < blk_e_col; q++)
        {
            if (p >= q) continue;
            jacobi_rotation_kernel(nrow, p, q, G, ldG, V, ldV);
        }
    }
}

static int blk_spos(const int nblock, const int blksize, const int blkrem, const int iblock)
{
    int res;
    if (iblock < blkrem) res = (blksize + 1) * iblock;
    else res = blksize * iblock + blkrem;
    return res;
}

void copy_matrix_block(
    double *dst, const int ldd, double *src, const int lds, 
    const int nrows, const int ncols
)
{
    size_t ncols_msize = sizeof(double) * ncols;
    for (int irow = 0; irow < nrows; irow++)
        memcpy(dst + irow * ldd, src + irow * lds, ncols_msize);
} 

// Generate next set of pairs for elimination
// Ref: Matrix Computation 4th edition, page 482
// Input parameters:
//   top, bot : (top[k], bot[k]) is a pair
//   npair    : Total number of pairs to be eliminated
// Output parameters:
//   top, bot : New sets of pairs
void next_elimination_pairs(int *top, int *bot, const int npair)
{
    int top_tail = top[npair - 1];
    int bot_head = bot[0];
    for (int l = npair - 1; l >= 2; l--) top[l] = top[l - 1];
    for (int l = 0; l < npair; l++) bot[l] = bot[l + 1];
    top[1] = bot_head;
    bot[npair - 1] = top_tail;
}

// Parallel Jacobi method for eigen decompisition, blocked version
// Input parameters:
//   A       : Symmetric matrix to be decomposed, row-major, 
//             size >= ldA * nrow, will be overwritten when exit
//   nrow    : Number of rows and columns
//   ldA     : Leading dimension of A, size >= nrow
//   ldV     : Leading dimension of V, size >= nrow
//   nthread : Number of threads
//   workbuf : Working buffer, size >= nrow
// Output parameters:
//   V : Eigenvectors, each row is an eigenvector, size >= ldV * nrow
//   D : Array, size >= nrow
void eig_jacobi_blocked(
    double *A, const int nrow, const int ldA,
    double *V, const int ldV, double *D,
    const int nthread, int *workbuf
)
{
    const int ncol   = nrow;
    const int semi_n = nrow / 2;
    
    // Initialize V = eye(nrow)
    memset(D, 0, sizeof(double) * nrow);
    memset(V, 0, sizeof(double) * ldV * nrow);
    for (int i = 0; i < nrow; i++) V[i * ldV + i] = 1.0;
    
    // We need to check the fro-norm of off-diagonal V' * A * V, but we don't want 
    // to form the matrix explicitly. ||A|_fro - ||diag(A)||_fro is equivalent.
    double A_2norm = 0.0, D_2norm = 0.0, relres_norm;
    #pragma omp parallel for num_threads(nthread) reduction(+:A_2norm)
    for (int i = 0; i < nrow; i++)
    {
        double tmp = 0.0;
        double *A_row = A + i * ldA;
        for (int j = 0; j < ncol; j++) tmp += A_row[j] * A_row[j];
        A_2norm += tmp;
    }
    A_2norm = sqrt(A_2norm);

    // Set the block size info
    int blksize = 32; //(256 * 1024) / (8 * 2 * 2 * nrow);
    int nblock  = nrow / blksize;  
    nblock = MAX(nblock, nthread);
    nblock = (nblock + 1) / 2 * 2; // Need to be even
    blksize = nrow / nblock;
    int blkrem = nrow % nblock;
    int semi_nblock = nblock / 2;
    int blksize1 = blksize + (blkrem > 0 ? 1 : 0);
    printf("[DEBUG] blksize, nblock, blkrem, semi_nblock = %d, %d, %d, %d\n", blksize, nblock, blkrem, semi_nblock);
    
    // Set of block pairs for elimination, no two pairs has the same element
    int *top = workbuf;
    int *bot = top + semi_nblock;
    for (int i = 0; i < semi_nblock; i++)
    {
        top[i] = 2 * i;
        bot[i] = 2 * i + 1;
    }

    size_t thread_buff_size = 4 * nrow * blksize1 + 8 * blksize1 * blksize1;
    double *buff = (double*) malloc(sizeof(double) * nthread * thread_buff_size);

    mkl_set_num_threads(1);

    // Conceptually, G = V' * A
    double *G = A;
    int ldG = ldA, sweep = 0;
    relres_norm = fabs(A_2norm - D_2norm) / A_2norm; 
    while (relres_norm > 1e-14)
    {
        double st = omp_get_wtime();
        #pragma omp parallel num_threads(nthread)
        {
            int tid = omp_get_thread_num();
            double *thread_buff = buff + tid * thread_buff_size;
            double *G_p_blk = thread_buff;
            double *G_q_blk = G_p_blk + blksize1 * nrow;
            double *V_p_blk = G_q_blk + blksize1 * nrow;
            double *V_q_blk = V_p_blk + blksize1 * nrow;
            double *A_blk   = V_q_blk + blksize1 * nrow;
            double *V_blk   = A_blk   + blksize1 * blksize1 * 4;
            double *G_k_blk = G_p_blk;
            double *V_k_blk = V_p_blk;
            
            // Eliminate off-diagonal blocks
            for (int subsweep = 0; subsweep < nblock - 1; subsweep++)
            {
                #pragma omp barrier
                #pragma omp for schedule(guided)
                for (int k = 0; k < semi_nblock; k++)
                {
                    int p_blk = MIN(top[k], bot[k]);
                    int q_blk = MAX(top[k], bot[k]);

                    int p_blk_spos = blk_spos(nblock, blksize, blkrem, p_blk);
                    int q_blk_spos = blk_spos(nblock, blksize, blkrem, q_blk);
                    int p_blk_epos = blk_spos(nblock, blksize, blkrem, p_blk + 1);
                    int q_blk_epos = blk_spos(nblock, blksize, blkrem, q_blk + 1);
                    int p_blk_size = p_blk_epos - p_blk_spos;
                    int q_blk_size = q_blk_epos - q_blk_spos;
                    int VAblk_size = p_blk_size + q_blk_size;
                    
                    double *G_p_blk_ = G + p_blk_spos * ldG;
                    double *G_q_blk_ = G + q_blk_spos * ldG;
                    double *V_p_blk_ = V + p_blk_spos * ldV;
                    double *V_q_blk_ = V + q_blk_spos * ldV;
                    
                    // GT_blk_p = GT(:, blk_p_s:blk_p_e);
                    copy_matrix_block(G_p_blk, nrow, G_p_blk_, ldG, p_blk_size, nrow);
                    // GT_blk_q = GT(:, blk_q_s:blk_q_e);
                    copy_matrix_block(G_q_blk, nrow, G_q_blk_, ldG, q_blk_size, nrow);
                    // V_blk_p  =  V(:, blk_p_s:blk_p_e);
                    copy_matrix_block(V_p_blk, nrow, V_p_blk_, ldV, p_blk_size, nrow);
                    // V_blk_q  =  V(:, blk_q_s:blk_q_e);
                    copy_matrix_block(V_q_blk, nrow, V_q_blk_, ldV, q_blk_size, nrow);
                    
                    int pp_offset = 0;
                    int pq_offset = p_blk_size;
                    int qp_offset = p_blk_size * VAblk_size;
                    int qq_offset = p_blk_size * (VAblk_size + 1);
                    
                    // A_blk(s0:e0, s0:e0) = GT_blk_p' * V_blk_p;
                    cblas_dgemm(
                        CblasRowMajor, CblasNoTrans, CblasTrans, 
                        p_blk_size, p_blk_size, nrow,
                        1.0, G_p_blk, nrow, V_p_blk, nrow, 
                        0.0, A_blk + pp_offset, VAblk_size
                    );
                    // A_blk(s0:e0, s1:e1) = GT_blk_p' * V_blk_q;
                    cblas_dgemm(
                        CblasRowMajor, CblasNoTrans, CblasTrans, 
                        p_blk_size, q_blk_size, nrow,
                        1.0, G_p_blk, nrow, V_q_blk, nrow, 
                        0.0, A_blk + pq_offset, VAblk_size
                    );
                    // A_blk(s1:e1, s0:e0) = GT_blk_q' * V_blk_p;
                    cblas_dgemm(
                        CblasRowMajor, CblasNoTrans, CblasTrans, 
                        q_blk_size, p_blk_size, nrow,
                        1.0, G_q_blk, nrow, V_p_blk, nrow, 
                        0.0, A_blk + qp_offset, VAblk_size
                    );
                    // A_blk(s1:e1, s1:e1) = GT_blk_q' * V_blk_q;
                    cblas_dgemm(
                        CblasRowMajor, CblasNoTrans, CblasTrans, 
                        q_blk_size, q_blk_size, nrow,
                        1.0, G_q_blk, nrow, V_q_blk, nrow, 
                        0.0, A_blk + qq_offset, VAblk_size
                    );
                    
                    // V_blk = jacobi_block_subsweep(A_blk, s0, e0, s1, e1);
                    jacobi_subblock_sweep(
                        VAblk_size, A_blk, V_blk, 
                        0, p_blk_size, p_blk_size, q_blk_size
                    );
                    
                    // Notice: V_blk in C is the transpose of V_blk in MATLAB
                    // GT(:, blk_p_s:blk_p_e) = GT_blk_p * V_blk(s0:e0, s0:e0) + GT_blk_q * V_blk(s1:e1, s0:e0);
                    cblas_dgemm(
                        CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                        p_blk_size, nrow, p_blk_size, 
                        1.0, V_blk + pp_offset, VAblk_size, G_p_blk, nrow, 
                        0.0, G_p_blk_, ldG
                    );
                    cblas_dgemm(
                        CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                        p_blk_size, nrow, q_blk_size, 
                        1.0, V_blk + pq_offset, VAblk_size, G_q_blk, nrow, 
                        1.0, G_p_blk_, ldG
                    );
                    // GT(:, blk_q_s:blk_q_e) = GT_blk_p * V_blk(s0:e0, s1:e1) + GT_blk_q * V_blk(s1:e1, s1:e1);
                    cblas_dgemm(
                        CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                        q_blk_size, nrow, p_blk_size, 
                        1.0, V_blk + qp_offset, VAblk_size, G_p_blk, nrow, 
                        0.0, G_q_blk_, ldG
                    );
                    cblas_dgemm(
                        CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                        q_blk_size, nrow, q_blk_size, 
                        1.0, V_blk + qq_offset, VAblk_size, G_q_blk, nrow, 
                        1.0, G_q_blk_, ldG
                    );
                    // V(:, blk_p_s:blk_p_e)  =  V_blk_p * V_blk(s0:e0, s0:e0) +  V_blk_q * V_blk(s1:e1, s0:e0);
                    cblas_dgemm(
                        CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                        p_blk_size, nrow, p_blk_size, 
                        1.0, V_blk + pp_offset, VAblk_size, V_p_blk, nrow, 
                        0.0, V_p_blk_, ldV
                    );
                    cblas_dgemm(
                        CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                        p_blk_size, nrow, q_blk_size, 
                        1.0, V_blk + pq_offset, VAblk_size, V_q_blk, nrow, 
                        1.0, V_p_blk_, ldV
                    );
                    // V(:, blk_q_s:blk_q_e)  =  V_blk_p * V_blk(s0:e0, s1:e1) +  V_blk_q * V_blk(s1:e1, s1:e1);
                    cblas_dgemm(
                        CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                        q_blk_size, nrow, p_blk_size, 
                        1.0, V_blk + qp_offset, VAblk_size, V_p_blk, nrow, 
                        0.0, V_q_blk_, ldV
                    );
                    cblas_dgemm(
                        CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                        q_blk_size, nrow, q_blk_size, 
                        1.0, V_blk + qq_offset, VAblk_size, V_q_blk, nrow, 
                        1.0, V_q_blk_, ldV
                    );
                }  // End of k loop

                #pragma omp master
                next_elimination_pairs(top, bot, semi_nblock);
            }  // End of subsweep loop

            // Eliminate diagonal blocks off-diagonal elements
            #pragma omp for schedule(guided)
            for (int k = 0; k < nblock; k++)
            {
                int k_blk_spos = blk_spos(nblock, blksize, blkrem, k);
                int k_blk_epos = blk_spos(nblock, blksize, blkrem, k + 1);
                int k_blk_size = k_blk_epos - k_blk_spos;
                
                double *G_k_blk_ = G + k_blk_spos * ldG;
                double *V_k_blk_ = V + k_blk_spos * ldV;
                copy_matrix_block(G_k_blk, nrow, G_k_blk_, ldG, k_blk_size, nrow);
                copy_matrix_block(V_k_blk, nrow, V_k_blk_, ldV, k_blk_size, nrow);
                
                // A_blk = GT(:, blk_p_s:blk_p_e)' * V(:, blk_p_s:blk_p_e);
                cblas_dgemm(
                    CblasRowMajor, CblasNoTrans, CblasTrans, 
                    k_blk_size, k_blk_size, nrow,
                    1.0, G_k_blk, nrow, V_k_blk, nrow, 
                    0.0, A_blk, k_blk_size
                );
                
                // V_blk = jacobi_block_subsweep(A_blk, 1, blk_p_n, 1, blk_p_n);
                jacobi_subblock_sweep(
                    k_blk_size, A_blk, V_blk, 
                    0, k_blk_size, 0, k_blk_size
                );
                
                // GT(:, blk_p_s:blk_p_e) = GT(:, blk_p_s:blk_p_e) * V_blk;
                cblas_dgemm(
                    CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                    k_blk_size, nrow, k_blk_size, 
                    1.0, V_blk, k_blk_size, G_k_blk, nrow, 
                    0.0, G_k_blk_, ldG
                );
                //  V(:, blk_p_s:blk_p_e) =  V(:, blk_p_s:blk_p_e) * V_blk;
                cblas_dgemm(
                    CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                    k_blk_size, nrow, k_blk_size, 
                    1.0, V_blk, k_blk_size, V_k_blk, nrow, 
                    0.0, V_k_blk_, ldV
                );
            }  // End of k loop
        }  // End of pragma omp parallel
        
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
        if (sweep >= 15) break;
    }  // End of while (relres_norm > 1e-14) loop 
    
    mkl_set_num_threads(nthread);
    free(buff);
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
    
    double *A   = (double*) malloc(sizeof(double) * n * n);
    double *A0  = (double*) malloc(sizeof(double) * n * n);
    double *VT  = (double*) malloc(sizeof(double) * n * n);
    double *D   = (double*) malloc(sizeof(double) * n);
    double *DVT = (double*) malloc(sizeof(double) * n * n);
    double *A1  = (double*) malloc(sizeof(double) * n * n);
    double *workbuf = (double*) malloc(sizeof(double) * n);
    assert(A != NULL && A0 != NULL && VT != NULL);
    assert(D != NULL && workbuf != NULL && DVT != NULL && A1 != NULL);

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
    eig_jacobi_blocked(A, n, n, VT, n, D, nthread, (int*) workbuf);
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
    
    test_mkl_eig(A, A0, workbuf, n);
    
    free(A);
    free(A0);
    free(VT);
    free(D);
    free(workbuf);
    free(DVT);
    free(A1);
}