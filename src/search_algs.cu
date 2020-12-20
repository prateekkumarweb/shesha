// // for __syncthreads()
// #ifndef __CUDACC__
// #define __CUDACC__
// #endif

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <curand.h>
#include <device_launch_parameters.h>

#include <cstdio>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "maxsat.h"
#include "search_algs.h"

#define THR_INDEX threadIdx.x
#define BLK_INDEX blockIdx.x
#define BLK_DIMEN blockDim.x
#define TOT_INDEX (THR_INDEX + BLK_DIMEN * BLK_INDEX)

#define HARD_WEIGHTS_AT(index) \
  f_d.hard_weights[(index) + (BLK_INDEX * fmd.nHardClauses)]
#define SOFT_WEIGHTS_AT(index) \
  f_d.soft_weights[(index) + (BLK_INDEX * fmd.nSoftClauses)]
#define HARD_SCORES_AT(index) \
  f_d.hard_scores[(index) + (BLK_INDEX * fmd.nHardClauses)]
#define SOFT_SCORES_AT(index) \
  f_d.soft_scores[(index) + (BLK_INDEX * fmd.nSoftClauses)]
#define VAR_SCORES_AT(index) \
  f_d.var_scores[(index) + (BLK_INDEX * (fmd.nVars + 1))]
#define VAR_HARD_ORIG_AT(index) \
  f_d.var_hard_orig[(index) + (BLK_INDEX * (fmd.nVars + 1))]
#define VAR_SOFT_ORIG_AT(index) \
  f_d.var_soft_orig[(index) + (BLK_INDEX * (fmd.nVars + 1))]
#define MODEL_AT(index) f_d.model[(index) + (BLK_INDEX * (fmd.nVars + 1))]
#define BEST_MODEL_AT_0(index) \
  f_d.best_model[(index) + (BLK_INDEX * (2 * fmd.nVars + 1))]
#define BEST_MODEL_AT_1(index) \
  f_d.best_model[(index) + (BLK_INDEX * (2 * fmd.nVars + 1)) + fmd.nVars]
#define BEST_MODEL_FLAG BEST_MODEL_AT_0(0)
#define VAR_OF_LIT(lit) (lit >> 1)
#define IS_LIT_SAT(lit) ((MODEL_AT(VAR_OF_LIT(lit)) ^ 1) == (lit & 1))
#define NEG_LIT(lit) (lit ^ 1)

__device__ void get_random_model(MaxSATFormula_d &f_d, const FormulaMetadata &fmd, curandState_t *state,
                                 bool update_best, int restart_flip = -1) {
  int index = THR_INDEX;
  for (int var_index = index + 1; var_index <= fmd.nVars;
       var_index += BLK_DIMEN) {
    float randf = curand_uniform(&state[TOT_INDEX]);
    if (restart_flip == -1) {
      MODEL_AT(var_index) = (randf <= 0.5);
      if (update_best)
        BEST_MODEL_AT_0(var_index) = BEST_MODEL_AT_1(var_index) =
            MODEL_AT(var_index);
    } else {
      float p = float(restart_flip) / float(fmd.nVars);
      if (randf <= p) {
        MODEL_AT(var_index) ^= 1;
      }
    }
  }
}

__device__ void update_hard_scores(MaxSATFormula_d &f_d, const MaxSATFormulaConstant_d &fconst_d, const FormulaMetadata &fmd) {
  int index = THR_INDEX;
  for (int hard_index = index; hard_index < fmd.nHardClauses;
       hard_index += BLK_DIMEN) {
    HARD_WEIGHTS_AT(hard_index) = 1;
    weight_t sat = 0;
    for (unsigned i = fconst_d.hard_clauses[hard_index];
         i < fconst_d.hard_clauses[hard_index + 1]; ++i) {
      unsigned lit = fconst_d.hard_clauses_lits[i];
      if (IS_LIT_SAT(lit)) {
        sat = 1;
        break;
      }
    }
    HARD_SCORES_AT(hard_index) = sat;
  }
}

__device__ void update_soft_scores(MaxSATFormula_d &f_d, const MaxSATFormulaConstant_d &fconst_d, const FormulaMetadata &fmd) {
  int index = THR_INDEX;
  for (int soft_index = index; soft_index < fmd.nSoftClauses;
       soft_index += BLK_DIMEN) {
    SOFT_WEIGHTS_AT(soft_index) = fconst_d.soft_weights_orig[soft_index];
    weight_t sat = 0;
    for (unsigned i = fconst_d.soft_clauses[soft_index];
         i < fconst_d.soft_clauses[soft_index + 1]; ++i) {
      unsigned lit = fconst_d.soft_clauses_lits[i];
      if (IS_LIT_SAT(lit)) {
        sat = fconst_d.soft_weights_orig[soft_index];
        break;
      }
    }
    SOFT_SCORES_AT(soft_index) = sat;
  }
}

__device__ unsigned choose_var_hard(MaxSATFormula_d &f_d, const MaxSATFormulaConstant_d &fconst_d, const FormulaMetadata &fmd,
                                    curandState_t *state) {
  int hard_index = -1;
  int count = 0;
  for (auto i = 0; i < fmd.nHardClauses; i++) {
    if (HARD_SCORES_AT(i) == 0) count++;
  }
  int idx = curand_uniform(&state[TOT_INDEX]) * count;
  count = -1;
  // TODO: better to recompute or store?
  for (auto i = 0; i < fmd.nHardClauses; i++) {
    if (HARD_SCORES_AT(i) == 0) count++;
    if (count == idx) {
      hard_index = i;
      break;
    }
  }
  int lit_index =
      curand_uniform(&state[TOT_INDEX]) *
      (fconst_d.hard_clauses[hard_index + 1] - fconst_d.hard_clauses[hard_index]);
  unsigned lit = fconst_d.hard_clauses_lits[lit_index];
  return VAR_OF_LIT(lit);
}

__device__ unsigned choose_var_soft(MaxSATFormula_d &f_d, const MaxSATFormulaConstant_d &fconst_d, const FormulaMetadata &fmd,
                                    curandState_t *state) {
  int soft_index = -1;
  int count = 0;
  for (auto i = 0; i < fmd.nSoftClauses; i++) {
    if (SOFT_SCORES_AT(i) == 0) count++;
  }
  int idx = curand_uniform(&state[TOT_INDEX]) * count;
  count = -1;
  // TODO: better to recompute or store?
  for (auto i = 0; i < fmd.nSoftClauses; i++) {
    if (SOFT_SCORES_AT(i) == 0) count++;
    if (count == idx) {
      soft_index = i;
      break;
    }
  }
  int lit_index =
      curand_uniform(&state[TOT_INDEX]) *
      (fconst_d.soft_clauses[soft_index + 1] - fconst_d.soft_clauses[soft_index]);
  unsigned lit = fconst_d.soft_clauses_lits[lit_index];
  return VAR_OF_LIT(lit);
}

__device__ void find_max_var(MaxSATFormula_d &f_d, const FormulaMetadata &fmd, unsigned &res_max_v,
                             unsigned &res_chosen_v, float &prev_score,
                             curandState_t *state) {
  unsigned max_v = 1;
  unsigned chosen_v = 1;
  int index = THR_INDEX;
  for (int i = index + 1; i <= fmd.nVars; i += BLK_DIMEN) {
    float p = VAR_SCORES_AT(i) / (VAR_SCORES_AT(chosen_v) + VAR_SCORES_AT(i));
    p = (p<0.5)? (p/2): ((1+p)/2);
    float randf = curand_uniform(&state[TOT_INDEX]);
    if (VAR_SCORES_AT(i) > prev_score &&
         ((VAR_SCORES_AT(chosen_v) <= prev_score) ||
        (randf <= p))) {
      chosen_v = i;
    }
    if (VAR_SCORES_AT(max_v) < VAR_SCORES_AT(i)) {
      max_v = i;
    }
  }
  __shared__ int r[1024];
  __shared__ int cr[1024];
  r[index] = max_v;
  cr[index] = chosen_v;
  __syncthreads();
  for (int size = BLK_DIMEN >> 1; size > 0; size >>= 1) {
    if (index < size) {
      float p = VAR_SCORES_AT(cr[size + index]) /
                (VAR_SCORES_AT(cr[size + index]) + VAR_SCORES_AT(cr[index]));
      float randf = curand_uniform(&state[TOT_INDEX]);
      if ((VAR_SCORES_AT(cr[size + index]) > prev_score &&
           VAR_SCORES_AT(cr[index]) <= prev_score) ||
          (randf <= p)) {
        cr[index] = cr[index + size];
      }
      if (VAR_SCORES_AT(r[size + index]) > VAR_SCORES_AT(r[index])) {
        r[index] = r[index + size];
      }
    }
    __syncthreads();
  }
  if (index == 0) {
    res_max_v = r[0];
    res_chosen_v = cr[0];
  }
}

__device__ void compute_total_hard_score(MaxSATFormula_d &f_d, const FormulaMetadata &fmd,
                                         weight_t &curr_hard_sat,
                                         weight_t &hard_score) {
  weight_t score = 0;
  int index = THR_INDEX;
  for (int i = index; i < fmd.nHardClauses; i += BLK_DIMEN) {
    score += HARD_SCORES_AT(i);
  }
  __shared__ int r[1024];
  r[index] = score;
  __syncthreads();
  for (int size = BLK_DIMEN >> 1; size > 0; size >>= 1) {
    if (index < size) {
      r[index] += r[index + size];
    }
    __syncthreads();
  }
  if (index == 0) {
    curr_hard_sat = hard_score = r[0];
  }
}

__device__ void compute_total_soft_score(MaxSATFormula_d &f_d, const FormulaMetadata &fmd,
                                         weight_t &curr_soft_sat,
                                         weight_t &soft_score) {
  weight_t score = 0;
  int index = THR_INDEX;
  for (int i = index; i < fmd.nSoftClauses; i += BLK_DIMEN) {
    score += SOFT_SCORES_AT(i);
  }
  __shared__ int r[1024];
  r[index] = score;
  __syncthreads();
  for (int size = BLK_DIMEN >> 1; size > 0; size >>= 1) {
    if (index < size) {
      r[index] += r[index + size];
    }
    __syncthreads();
  }
  if (index == 0) {
    curr_soft_sat = soft_score = r[0];
  }
}

__global__ void search_alg(MaxSATFormula_d f_d, const MaxSATFormulaConstant_d fconst_d, const FormulaMetadata fmd, weight_t soft_max_delta,
                           float alpha, float beta, float sp,
                           float flip_hard_prob, int hard_inc, int n_iters,
                           int restart_threshold, int restart_flip,
                           curandState_t *state) {


  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  curand_init(9876543210l, idx, 0, &state[idx]);

  int index = THR_INDEX;

  bool first_thread = (index == 0);

  __shared__ weight_t best_hard_sat;
  __shared__ weight_t curr_hard_sat;
  __shared__ weight_t best_soft_sat;
  __shared__ weight_t curr_soft_sat;

  __shared__ weight_t total_hard_scores;
  __shared__ weight_t total_soft_scores;
  __shared__ float prev_score;
  __shared__ int not_improved;

  __shared__ bool smooth;
  __shared__ bool local_opt;

  __shared__ unsigned max_v;
  __shared__ unsigned chosen_v;
  __shared__ weight_t soft_max;

  if (first_thread) {
    best_hard_sat = curr_hard_sat = best_soft_sat = curr_soft_sat = 0;
    prev_score = 0.0;
    curr_hard_sat = 0;
    curr_soft_sat = 0;
    max_v = 1;
    chosen_v = 1;
    BEST_MODEL_FLAG = 0;
    soft_max = 1;
  }

//  if (BLK_INDEX != 0) get_random_model(f_d, fmd, state, true);

  __syncthreads();

  update_hard_scores(f_d, fconst_d, fmd);
  update_soft_scores(f_d, fconst_d, fmd);

  __syncthreads();

  compute_total_hard_score(f_d, fmd, curr_hard_sat, total_hard_scores);
  compute_total_soft_score(f_d, fmd, curr_soft_sat, total_soft_scores);

  if (first_thread) {
    prev_score = alpha * total_hard_scores + beta * total_soft_scores;
    best_hard_sat = curr_hard_sat;
    best_soft_sat = curr_soft_sat;
    not_improved = 0;

    printf("c Block %d: Initial Hard: %ld Soft: %ld Flag: %d\n", BLK_INDEX,
           best_hard_sat, best_soft_sat, BEST_MODEL_FLAG);
  }

  __syncthreads();

  for (unsigned iter = 0; iter < n_iters; iter++) {
    if (not_improved > restart_threshold) {
      not_improved = 0;
      if (first_thread)
        printf("c Block %d: Restarting at iter %d\n", BLK_INDEX, iter);
      get_random_model(f_d, fmd, state, false, restart_flip);
      __syncthreads();
    }

    for (unsigned v = index + 1; v <= fmd.nVars; v += BLK_DIMEN) {
      weight_t sat_hard_sum = 0;
      weight_t sat_soft_sum = 0;
      weight_t sat_hard_sum_orig = 0;
      weight_t sat_soft_sum_orig = 0;

      for (unsigned hard_index = 0; hard_index < fmd.nHardClauses;
           hard_index++) {
        for (unsigned i = fconst_d.hard_clauses[hard_index];
             i < fconst_d.hard_clauses[hard_index + 1]; ++i) {
          unsigned lit = fconst_d.hard_clauses_lits[i];
          // Simulate var flip
          if (VAR_OF_LIT(lit) == v) lit = NEG_LIT(lit);
          if (IS_LIT_SAT(lit)) {
            sat_hard_sum += HARD_WEIGHTS_AT(hard_index);
            sat_hard_sum_orig++;
            break;
          }
        }
      }

      for (unsigned soft_index = 0; soft_index < fmd.nSoftClauses;
           soft_index++) {
        for (unsigned i = fconst_d.soft_clauses[soft_index];
             i < fconst_d.soft_clauses[soft_index + 1]; ++i) {
          unsigned lit = fconst_d.soft_clauses_lits[i];
          // Simulate var flip
          if (VAR_OF_LIT(lit) == v) lit = NEG_LIT(lit);
          if (IS_LIT_SAT(lit)) {
            sat_soft_sum += SOFT_WEIGHTS_AT(soft_index);
            sat_soft_sum_orig += fconst_d.soft_weights_orig[soft_index];
            break;
          }
        }
      }

      float score = alpha * sat_hard_sum + beta * sat_soft_sum;
      VAR_SCORES_AT(v) = score;
      VAR_HARD_ORIG_AT(v) = sat_hard_sum_orig;
      VAR_SOFT_ORIG_AT(v) = sat_soft_sum_orig;
    }

    __syncthreads();

    find_max_var(f_d, fmd, max_v, chosen_v, prev_score, state);

    if (first_thread) {
      if (VAR_SCORES_AT(max_v) > prev_score) {
        MODEL_AT(chosen_v) ^= 1;
        local_opt = false;
      } else {
        local_opt = true;
        bool use_hard;
        float randf = curand_uniform(&state[TOT_INDEX]);
        use_hard = (randf <= flip_hard_prob);

        if (fmd.nHardClauses > 0 &&
            ((use_hard && curr_hard_sat != fmd.nHardClauses) ||
             curr_soft_sat == fmd.sum_soft_weights)) {
          chosen_v = choose_var_hard(f_d, fconst_d, fmd, state);
        } else if ((!use_hard && curr_soft_sat != fmd.sum_soft_weights) ||
                   curr_hard_sat == fmd.nHardClauses) {
          chosen_v = choose_var_soft(f_d, fconst_d, fmd, state);
        }
        MODEL_AT(chosen_v) ^= 1;

        randf = curand_uniform(&state[TOT_INDEX]);
        smooth = (randf <= sp);

        not_improved++;
      }

      prev_score = VAR_SCORES_AT(chosen_v);
      curr_hard_sat = VAR_HARD_ORIG_AT(chosen_v);
      curr_soft_sat = VAR_SOFT_ORIG_AT(chosen_v);
    }

    __syncthreads();

    if (local_opt) {
      for (unsigned i = index; i < fmd.nHardClauses; i += BLK_DIMEN) {
        if (smooth && HARD_WEIGHTS_AT(i) > 1 && HARD_SCORES_AT(i) > 0) {
          HARD_WEIGHTS_AT(i)--;
          prev_score -= alpha;
        }
        if (!smooth && HARD_SCORES_AT(i) == 0) {
          HARD_WEIGHTS_AT(i) += hard_inc;
        }
      }

      for (unsigned i = index; i < fmd.nSoftClauses; i += BLK_DIMEN) {
        if (smooth && SOFT_WEIGHTS_AT(i) > 1 && SOFT_SCORES_AT(i) > 0) {
          SOFT_WEIGHTS_AT(i)--;
          prev_score -= beta;
        }
        if (!smooth && SOFT_SCORES_AT(i) == 0 &&
            SOFT_WEIGHTS_AT(i) < soft_max) {
          SOFT_WEIGHTS_AT(i)++;
        }
      }
    }

    if (curr_hard_sat > best_hard_sat ||
        (curr_hard_sat == best_hard_sat && curr_soft_sat > best_soft_sat)) {
      if (BEST_MODEL_FLAG) {
        for (unsigned i = index + 1; i <= fmd.nVars; i += BLK_DIMEN) {
          BEST_MODEL_AT_0(i) = MODEL_AT(i);
        }
      } else {
        for (unsigned i = index + 1; i <= fmd.nVars; i += BLK_DIMEN) {
          BEST_MODEL_AT_1(i) = MODEL_AT(i);
        }
      }

      if (index == 0) {
        atomicXor(&BEST_MODEL_FLAG, 1);
        best_hard_sat = curr_hard_sat;
        best_soft_sat = curr_soft_sat;
        printf("c Block %d: Updated best model(%ld, %ld) at %d\n", BLK_INDEX,
               best_hard_sat, best_soft_sat, iter);
        soft_max += soft_max_delta;
      }
    }

    __syncthreads();
  }
}
