#if !defined(SEARCH_ALGS_H)
#define SEARCH_ALGS_H

#include <curand_kernel.h>
#include "maxsat.h"

__global__ void search_alg1(MaxSATFormula_d f_d, unsigned hard_weight, float alpha, float beta, float sp, float flip_hard_prob, int n_iters, int restart_threshold, curandState_t *d_state);
// __global__ void search_alg2(MaxSATFormula_d f_d, unsigned hard_weight, float hard_prob, float sp, int n_iters, int restart_threshold, curandState_t *d_state);

#endif // SEARCH_ALGS_H
