#if !defined(SEARCH_ALGS_H)
#define SEARCH_ALGS_H

#include <curand_kernel.h>
#include "maxsat.h"

__global__ void search_alg(MaxSATFormula_d f_d, unsigned hard_weight,
                           float alpha, float beta, float sp,
                           float flip_hard_prob, int hard_inc, int n_iters,
                           int restart_threshold, int restart_flip,
                           curandState_t *d_state);

#endif  // SEARCH_ALGS_H
