#if !defined(SEARCH_ALGS_H)
#define SEARCH_ALGS_H

__global__ void search_alg1(int nVars, int nClauses, int hardWeight,
                           int *input_arr, int *clauses, int *weights,
                           int *model, int *best_model, float *scores,
                           int *d_sat, float alpha, float beta);

#endif // SEARCH_ALGS_H
