#include <stdio.h>
#include <mex.h>
#include "simint/simint.h"

#define MAX(a,b) \
   ({ __typeof__ (a) _a = (a); \
      __typeof__ (b) _b = (b); \
      _a > _b ? _a : _b; })

#define NCART(am) ((am>=0)?((((am)+2)*((am)+1))>>1):0)


/*
input arguments: 
    array of atomic numbers
    array of x coords
    array of y coords
    array of z coords
    shell a
    shell b
*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    int i, j;

    if (nrhs != 6)
        mexErrMsgTxt("Function needs input of six arguments: Z array, x array, y array, z array, shell1, shell2.");
    if (nlhs > 1)
        mexErrMsgTxt("Too many output arguments.");

    int     ncenter     = mxGetM (prhs[0]);
    double *atomic_nums = mxGetPr(prhs[0]);
    double *xcoords     = mxGetPr(prhs[1]);
    double *ycoords     = mxGetPr(prhs[2]);
    double *zcoords     = mxGetPr(prhs[3]);

    int     temp        = mxGetN (prhs[0]);
    if (temp > ncenter)
        mexErrMsgTxt("Atomic positions must be tall vectors.");

    simint_init();

    struct simint_shell shell[2];
    int max_am = 0;

    for (i=0; i<2; i++)
    {
        int am    = (int)    *mxGetPr(mxGetField(prhs[4+i], 0, "am"));
        int nprim = (int)    *mxGetPr(mxGetField(prhs[4+i], 0, "nprim"));
        double x  = (double) *mxGetPr(mxGetField(prhs[4+i], 0, "x"));
        double y  = (double) *mxGetPr(mxGetField(prhs[4+i], 0, "y"));
        double z  = (double) *mxGetPr(mxGetField(prhs[4+i], 0, "z"));
        double *alpha =       mxGetPr(mxGetField(prhs[4+i], 0, "alpha"));
        double *coef  =       mxGetPr(mxGetField(prhs[4+i], 0, "coef"));

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
    for (i=0; i<2; i++)
    {
        if (shell[i].alpha[0] != 0.)
            simint_normalize_shells(1, &shell[i]);
        else
        {
            if (shell[i].coef[0] != 1. || shell[i].nprim != 1)
                mexErrMsgTxt("Bad unit shell.");
        }
    }

    // allocate output
    // compute number of scalar integrals to be computed
    int size = NCART(shell[0].am) * NCART(shell[1].am);

    plhs[0] = mxCreateDoubleMatrix(size, 1, mxREAL);
    double *targets = mxGetPr(plhs[0]);

    // ncomputed should always be 1 in this code
    int ncomputed = 
        simint_compute_potential(ncenter, atomic_nums, 
                xcoords, ycoords, zcoords, &shell[0], &shell[1], targets);

    if (ncomputed < 0)
        mexErrMsgTxt("simint_compute_potential returned < 0.");

    simint_free_shell(&shell[0]);
    simint_free_shell(&shell[1]);
    simint_finalize();
}
