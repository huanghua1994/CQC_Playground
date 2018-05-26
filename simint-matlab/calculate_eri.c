#include <stdio.h>
#include <mex.h>
#include "simint/simint.h"

#define MAX(a,b) \
   ({ __typeof__ (a) _a = (a); \
      __typeof__ (b) _b = (b); \
      _a > _b ? _a : _b; })

#define NCART(am) ((am>=0)?((((am)+2)*((am)+1))>>1):0)


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    int i, j;

    if (nrhs != 4)
        mexErrMsgTxt("Function needs input of four shells.");
    if (nlhs > 1)
        mexErrMsgTxt("Too many output arguments.");

    simint_init();

    struct simint_shell shell[4];
    int max_am = 0;

    for (i=0; i<4; i++)
    {
        int am    = (int)    *mxGetPr(mxGetField(prhs[i], 0, "am"));
        int nprim = (int)    *mxGetPr(mxGetField(prhs[i], 0, "nprim"));
        double x  = (double) *mxGetPr(mxGetField(prhs[i], 0, "x"));
        double y  = (double) *mxGetPr(mxGetField(prhs[i], 0, "y"));
        double z  = (double) *mxGetPr(mxGetField(prhs[i], 0, "z"));
        double *alpha =       mxGetPr(mxGetField(prhs[i], 0, "alpha"));
        double *coef  =       mxGetPr(mxGetField(prhs[i], 0, "coef"));

        simint_initialize_shell(&shell[i]);
        simint_allocate_shell(nprim, &shell[i]);

        max_am = MAX(max_am, am);

        shell[i].am    = am;
        shell[i].nprim = nprim;
        shell[i].x     = x;
        shell[i].y     = y;
        shell[i].z     = z;

        for (j=0; j<nprim; j++)
        {
            shell[i].alpha[j] = alpha[j];
            shell[i].coef[j]  = coef[j];
        }

    }

    // normalize
    // except for shells that have orbital exponent zero
    for (i=0; i<4; i++)
    {
        if (shell[i].alpha[0] != 0.)
            simint_normalize_shells(1, &shell[i]);
        else
        {
            if (shell[i].coef[0] != 1. || shell[i].nprim != 1)
                mexErrMsgTxt("Bad unit shell.");
        }
    }

    // multishell pairs
    struct simint_multi_shellpair bra_pair;
    struct simint_multi_shellpair ket_pair;

    simint_initialize_multi_shellpair(&bra_pair);
    simint_initialize_multi_shellpair(&ket_pair);

    simint_create_multi_shellpair(1, &shell[0], 1, &shell[1], &bra_pair, 0);
    simint_create_multi_shellpair(1, &shell[2], 1, &shell[3], &ket_pair, 0);

    // Arguments to simint_ostei_workmem: Derivative order, max am
    double *work = SIMINT_ALLOC(simint_ostei_workmem(0, max_am));

    // allocate output
    // compute number of scalar integrals to be computed
    int size = NCART(shell[0].am) * NCART(shell[1].am) 
             * NCART(shell[2].am) * NCART(shell[3].am);

    plhs[0] = mxCreateDoubleMatrix(size, 1, mxREAL);
    double *targets = mxGetPr(plhs[0]);

    // ncomputed should always be 1 in this code
    int ncomputed = 
        simint_compute_eri(&bra_pair, &ket_pair, 0.0, work, targets);

    simint_free_shell(&shell[0]);
    simint_free_shell(&shell[1]);
    simint_free_shell(&shell[2]);
    simint_free_shell(&shell[3]);
    simint_free_multi_shellpair(&bra_pair);
    simint_free_multi_shellpair(&ket_pair);
    SIMINT_FREE(work);
    simint_finalize();
}
