#if !defined(SEARCH_ALGS_H)
#define SEARCH_ALGS_H

#include "maxsat.h"

__global__ void search_alg1(MaxSATFormula_d f_d, float alpha, float beta, float sp, curandState_t *d_state);

#endif // SEARCH_ALGS_H
