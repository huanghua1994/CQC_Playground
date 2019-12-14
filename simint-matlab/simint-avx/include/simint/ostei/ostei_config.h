#pragma once

#include "simint/vectorization/vectorization.h"

#define SIMINT_OSTEI_MAXAM 4
#define SIMINT_OSTEI_MAXDER 0
#define SIMINT_OSTEI_DERIV1_MAXAM (SIMINT_OSTEI_MAXDER > 0 ? 4 : -1)


static inline size_t simint_ostei_worksize(int derorder, int maxam)
{
    static const size_t nelements[1][5] = {
      {
        (SIMINT_SIMD_ROUND(SIMINT_NSHELL_SIMD*1) + SIMINT_SIMD_ROUND(0) + SIMINT_SIMD_LEN*1),
        (SIMINT_SIMD_ROUND(SIMINT_NSHELL_SIMD*81) + SIMINT_SIMD_ROUND(81) + SIMINT_SIMD_LEN*149),
        (SIMINT_SIMD_ROUND(SIMINT_NSHELL_SIMD*961) + SIMINT_SIMD_ROUND(4332) + SIMINT_SIMD_LEN*2405),
        (SIMINT_SIMD_ROUND(SIMINT_NSHELL_SIMD*5476) + SIMINT_SIMD_ROUND(57512) + SIMINT_SIMD_LEN*17273),
        (SIMINT_SIMD_ROUND(SIMINT_NSHELL_SIMD*21025) + SIMINT_SIMD_ROUND(418905) + SIMINT_SIMD_LEN*79965),
      },
    };
    return nelements[derorder][maxam];
}

static inline size_t simint_ostei_workmem(int derorder, int maxam)
{
    return simint_ostei_worksize(derorder, maxam)*sizeof(double);
}


