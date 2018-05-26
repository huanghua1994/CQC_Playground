#include <stdio.h>
#include <mex.h>


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    int ammap[128];
    ammap['S'] =  0;
    ammap['P'] =  1;
    ammap['D'] =  2;
    ammap['F'] =  3;
    ammap['G'] =  4;
    ammap['H'] =  5;
    ammap['I'] =  6;
    ammap['J'] =  7;
    ammap['K'] =  8;
    ammap['L'] =  9;
    ammap['M'] = 10;
    ammap['N'] = 11;
    ammap['O'] = 12;
    ammap['Q'] = 13;
    ammap['R'] = 14;
    ammap['T'] = 15;
    ammap['U'] = 16;
    ammap['V'] = 17;
    ammap['W'] = 18;
    ammap['X'] = 19;
    ammap['Y'] = 20;
    ammap['Z'] = 21;
    ammap['A'] = 22;
    ammap['B'] = 23;
    ammap['C'] = 24;
    ammap['E'] = 25;

    FILE *fp;
    int i, j, k, l;

    int natom;
    int ntot_shells = 0;

    if (nrhs != 1)
        mexErrMsgTxt("Function needs one filename parameter.");
    if (nlhs > 1)
        mexErrMsgTxt("Too many output arguments.");

    char *filename = mxArrayToString(prhs[0]);
    fp = fopen(filename, "r");
    if (fp == NULL)
        mexErrMsgTxt("Could not open input file.");

    fscanf(fp, "%d", &natom);
    // printf("natom %d\n", natom);

    for (i=0; i<natom; i++)
    {
        char sym[100];
        int nshell, nallprim, nallprimg;
        double x, y, z;

        fscanf(fp, "%s %d %d %d", sym, &nshell, &nallprim, &nallprimg);
        fscanf(fp, "%lf %lf %lf", &x, &y, &z);

        for (j=0; j<nshell; j++)
        {
            char type[100];
            int nprim, ngen;
            fscanf(fp, "%s %d %d", type, &nprim, &ngen);

            ntot_shells += ngen;

            for (k=0; k<nprim; k++)
            {
                double alpha;
                fscanf(fp, "%lf", &alpha);

                for (l=0; l<ngen; l++)
                {
                    double coef;
                    fscanf(fp, "%lf", &coef);

                    // printf("atom %d, shell %d, prim %d, coef %f\n", i, j, k, coef);
                }
            }
        }
    }

    fclose(fp);

    // printf("total number of shells %d\n", ntot_shells);

    const char *field_names[] = {"atom_ind", "atom_sym", 
        "am", "nprim", "x", "y", "z", "alpha", "coef"};
    mwSize dims[2] = {1, ntot_shells};

    plhs[0] = mxCreateStructArray(2, dims, 9, field_names);

    fp = fopen(filename, "r");
    fscanf(fp, "%d", &natom);

    int shell_index = 0;

    for (i=0; i<natom; i++)
    {
        char sym[100];
        int nshell, nallprim, nallprimg;
        double x, y, z;

        fscanf(fp, "%s %d %d %d", sym, &nshell, &nallprim, &nallprimg);
        fscanf(fp, "%lf %lf %lf", &x, &y, &z);

        for (j=0; j<nshell; j++)
        {
            char type[100];
            int nprim, ngen;
            fscanf(fp, "%s %d %d", type, &nprim, &ngen);

            if (ngen > 100)
                mexErrMsgTxt("Too many general shells.");

            double *alpha_p[100];
            double *coef_p[100];

            // fill structure, including allocating arrays for alpha and coef
            for (l=0; l<ngen; l++, shell_index++)
            {
                mxArray *field_value;

                // atom_ind
                field_value = mxCreateDoubleMatrix(1, 1, mxREAL);
                *mxGetPr(field_value) = i+1; // 1-based
                mxSetFieldByNumber(plhs[0], shell_index, 0, field_value);

                // atom_sym
                field_value = mxCreateString(sym);
                mxSetFieldByNumber(plhs[0], shell_index, 1, field_value);

                // am
                field_value = mxCreateDoubleMatrix(1, 1, mxREAL);
                *mxGetPr(field_value) = ammap[(char)type[l]];
                mxSetFieldByNumber(plhs[0], shell_index, 2, field_value);

                // nprim
                field_value = mxCreateDoubleMatrix(1, 1, mxREAL);
                *mxGetPr(field_value) = nprim;
                mxSetFieldByNumber(plhs[0], shell_index, 3, field_value);

                // x
                field_value = mxCreateDoubleMatrix(1, 1, mxREAL);
                *mxGetPr(field_value) = x;
                mxSetFieldByNumber(plhs[0], shell_index, 4, field_value);

                // y
                field_value = mxCreateDoubleMatrix(1, 1, mxREAL);
                *mxGetPr(field_value) = y;
                mxSetFieldByNumber(plhs[0], shell_index, 5, field_value);

                // z
                field_value = mxCreateDoubleMatrix(1, 1, mxREAL);
                *mxGetPr(field_value) = z;
                mxSetFieldByNumber(plhs[0], shell_index, 6, field_value);

                // alpha
                field_value = mxCreateDoubleMatrix(1, nprim, mxREAL);
                alpha_p[l] = mxGetPr(field_value);
                mxSetFieldByNumber(plhs[0], shell_index, 7, field_value);

                // coef
                field_value = mxCreateDoubleMatrix(1, nprim, mxREAL);
                coef_p[l] = mxGetPr(field_value);
                mxSetFieldByNumber(plhs[0], shell_index, 8, field_value);
            }

            for (k=0; k<nprim; k++)
            {
                double alpha;
                fscanf(fp, "%lf", &alpha);

                for (l=0; l<ngen; l++)
                {
                    double coef;
                    fscanf(fp, "%lf", &coef);
                    coef_p[l][k] = coef;
                    alpha_p[l][k] = alpha;

                    // printf("atom %d, shell %d, prim %d, coef %f\n", i, j, k, coef);
                }
            }
        }
    }

    fclose(fp);
}
