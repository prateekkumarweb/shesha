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

#define HARD_WEIGHTS_AT(index) f_d.hard_weights[index]
#define SOFT_WEIGHTS_AT(index) f_d.soft_weights[index]
#define HARD_SCORES_AT(index) f_d.hard_scores[index]
#define SOFT_SCORES_AT(index) f_d.soft_scores[index]
#define VAR_SCORES_AT(index) f_d.var_scores[index]
#define VAR_HARD_ORIG_AT(index) f_d.var_hard_orig[index]
#define VAR_SOFT_ORIG_AT(index) f_d.var_soft_orig[index]
#define MODEL_AT(index) f_d.model[index]
#define BEST_MODEL_AT(index) f_d.best_model[index]

__device__ void get_random_model(MaxSATFormula_d &f_d, curandState_t *state,
                                 bool update_best, int restart_flip = -1) {
  int index = threadIdx.x + blockDim.x * blockIdx.x;
  for (int var_index = index + 1; var_index <= f_d.nVars;
       var_index += blockDim.x) {
    float randf = curand_uniform(&state[index]);
    if (restart_flip == -1) {
      if (randf <= 0.5)
        MODEL_AT(var_index) = 1;
      else
        MODEL_AT(var_index) = 0;
      if (update_best) BEST_MODEL_AT(var_index) = MODEL_AT(var_index);
    } else {
      float p = float(restart_flip) / float(f_d.nVars);
      if (randf <= p) {
        MODEL_AT(var_index) = 1 - MODEL_AT(var_index);
      }
    }
  }
}

__device__ void update_hard_scores(MaxSATFormula_d &f_d) {
  int index = threadIdx.x + blockDim.x * blockIdx.x;
  for (int hard_index = index; hard_index < f_d.nHardClauses;
       hard_index += blockDim.x) {
    HARD_WEIGHTS_AT(hard_index) = 1;
    int sat = 0;
    for (unsigned i = f_d.hard_clauses[hard_index];
         i < f_d.hard_clauses[hard_index + 1]; ++i) {
      int lit = f_d.hard_clauses_lits[i];
      if (lit > 0) {
        if (MODEL_AT(lit) == 1) {
          sat = 1;
          break;
        }
      } else {
        if (MODEL_AT(-lit) == 0) {
          sat = 1;
          break;
        }
      }
    }
    HARD_SCORES_AT(hard_index) = sat;
  }
}

__device__ void update_soft_scores(MaxSATFormula_d &f_d) {
  int index = threadIdx.x + blockDim.x * blockIdx.x;
  for (int soft_index = index; soft_index < f_d.nSoftClauses;
       soft_index += blockDim.x) {
    SOFT_WEIGHTS_AT(soft_index) = f_d.soft_weights_orig[soft_index];
    int sat = 0;
    for (unsigned i = f_d.soft_clauses[soft_index];
         i < f_d.soft_clauses[soft_index + 1]; ++i) {
      int lit = f_d.soft_clauses_lits[i];
      if (lit > 0) {
        if (MODEL_AT(lit) == 1) {
          sat = f_d.soft_weights_orig[soft_index];
          break;
        }
      } else {
        if (MODEL_AT(-lit) == 0) {
          sat = f_d.soft_weights_orig[soft_index];
          break;
        }
      }
    }
    SOFT_SCORES_AT(soft_index) = sat;
  }
}

__device__ unsigned choose_var_hard(MaxSATFormula_d &f_d,
                                    curandState_t *state) {
  int index = threadIdx.x + blockDim.x * blockIdx.x;
  int hard_index = -1;
  int count = 0;
  for (auto i = 0; i < f_d.nHardClauses; i++) {
    if (HARD_SCORES_AT(i) == 0) count++;
  }
  int idx = curand_uniform(&state[index]) * count;
  count = -1;
  // TODO: better to recompute or store?
  for (auto i = 0; i < f_d.nHardClauses; i++) {
    if (HARD_SCORES_AT(i) == 0) count++;
    if (count == idx) {
      hard_index = i;
      break;
    }
  }
  int lit_index =
      curand_uniform(&state[index]) *
      (f_d.hard_clauses[hard_index + 1] - f_d.hard_clauses[hard_index]);
  int lit = f_d.hard_clauses_lits[lit_index];
  if (lit < 0) {
    return -lit;
  }
  return lit;
}

__device__ unsigned choose_var_soft(MaxSATFormula_d &f_d,
                                    curandState_t *state) {
  int index = threadIdx.x + blockDim.x * blockIdx.x;
  int soft_index = -1;
  int count = 0;
  for (auto i = 0; i < f_d.nSoftClauses; i++) {
    if (SOFT_SCORES_AT(i) == 0) count++;
  }
  int idx = curand_uniform(&state[index]) * count;
  count = -1;
  // TODO: better to recompute or store?
  for (auto i = 0; i < f_d.nSoftClauses; i++) {
    if (SOFT_SCORES_AT(i) == 0) count++;
    if (count == idx) {
      soft_index = i;
      break;
    }
  }
  int lit_index =
      curand_uniform(&state[index]) *
      (f_d.soft_clauses[soft_index + 1] - f_d.soft_clauses[soft_index]);
  int lit = f_d.soft_clauses_lits[lit_index];
  if (lit < 0) {
    return -lit;
  }
  return lit;
}

__device__ void find_max_var(MaxSATFormula_d &f_d, unsigned &res_max_v,
                             unsigned &res_chosen_v, float &prev_score,
                             curandState_t *state) {
  unsigned max_v = 1;
  unsigned chosen_v = 1;
  int index = threadIdx.x + blockDim.x * blockIdx.x;
  for (int i = index + 1; i <= f_d.nVars; i += blockDim.x) {
    float p = VAR_SCORES_AT(i) / (VAR_SCORES_AT(chosen_v) + VAR_SCORES_AT(i));
    float randf = curand_uniform(&state[index]);
    if ((VAR_SCORES_AT(i) > prev_score &&
         VAR_SCORES_AT(chosen_v) <= prev_score) ||
        (randf <= p)) {
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
  for (int size = blockDim.x >> 1; size > 0; size >>= 1) {
    if (index < size) {
      float p = VAR_SCORES_AT(cr[size + index]) /
                (VAR_SCORES_AT(cr[size + index]) + VAR_SCORES_AT(cr[index]));
      float randf = curand_uniform(&state[index]);
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

__device__ void compute_total_hard_score(MaxSATFormula_d &f_d,
                                         unsigned &curr_hard_sat,
                                         unsigned &hard_score) {
  unsigned score = 0;
  int index = threadIdx.x + blockDim.x * blockIdx.x;
  for (int i = index; i < f_d.nHardClauses; i += blockDim.x) {
    score += HARD_SCORES_AT(i);
  }
  __shared__ int r[1024];
  r[index] = score;
  __syncthreads();
  for (int size = blockDim.x >> 1; size > 0; size >>= 1) {
    if (index < size) {
      r[index] += r[index + size];
    }
    __syncthreads();
  }
  if (index == 0) {
    curr_hard_sat = hard_score = r[0];
  }
}

__device__ void compute_total_soft_score(MaxSATFormula_d &f_d,
                                         unsigned &curr_soft_sat,
                                         unsigned &soft_score) {
  unsigned score = 0;
  int index = threadIdx.x + blockDim.x * blockIdx.x;
  for (int i = index; i < f_d.nSoftClauses; i += blockDim.x) {
    score += SOFT_SCORES_AT(i);
  }
  __shared__ int r[1024];
  r[index] = score;
  __syncthreads();
  for (int size = blockDim.x >> 1; size > 0; size >>= 1) {
    if (index < size) {
      r[index] += r[index + size];
    }
    __syncthreads();
  }
  if (index == 0) {
    curr_soft_sat = soft_score = r[0];
  }
}

__device__ void print_final(int index, MaxSATFormula_d &f_d,
                            unsigned &curr_hard_sat, unsigned &curr_soft_sat,
                            unsigned &best_hard_sat, unsigned &best_soft_sat) {
  if (index == 0) {
    printf("c Last: %d %d\n", curr_hard_sat, curr_soft_sat);
    // printf("v");
    // for (int i = 1; i <= f_d.nVars; i++) {
    //   printf(" %d", MODEL_AT(i) > 0 ? i : -i);
    // }
    // printf("\n");
    printf("c Best: %d %d\n", best_hard_sat, best_soft_sat);
    // printf("b");
    // for (int i = 1; i <= f_d.nVars; i++) {
    //   printf(" %d", BEST_MODEL_AT(i) > 0 ? i : -i);
    // }
    // printf("\n");
  }
}

__global__ void search_alg(MaxSATFormula_d f_d, unsigned soft_max_delta,
                           float alpha, float beta, float sp,
                           float flip_hard_prob, int hard_inc, int n_iters,
                           int restart_threshold, int restart_flip,
                           curandState_t *state) {
  int index = threadIdx.x + blockDim.x * blockIdx.x;

  bool first_thread = (index == 0);

  __shared__ unsigned best_hard_sat;
  __shared__ unsigned curr_hard_sat;
  __shared__ unsigned best_soft_sat;
  __shared__ unsigned curr_soft_sat;

  __shared__ unsigned total_hard_scores;
  __shared__ unsigned total_soft_scores;
  __shared__ float prev_score;
  __shared__ int not_improved;

  __shared__ bool smooth;
  __shared__ bool local_opt;

  __shared__ unsigned max_v;
  __shared__ unsigned chosen_v;
  __shared__ unsigned soft_max;

  if (first_thread) {
    best_hard_sat = curr_hard_sat = best_soft_sat = curr_soft_sat = 0;
    prev_score = 0.0;
    curr_hard_sat = 0;
    curr_soft_sat = 0;
    max_v = 1;
    chosen_v = 1;
    MODEL_AT(0) = 0;
    soft_max = 1;
  }

  // if (f_d.nHardClauses == 0) get_random_model(f_d, state, true);

  __syncthreads();

  update_hard_scores(f_d);
  update_soft_scores(f_d);

  __syncthreads();

  compute_total_hard_score(f_d, curr_hard_sat, total_hard_scores);
  compute_total_soft_score(f_d, curr_soft_sat, total_soft_scores);

  if (first_thread) {
    prev_score = alpha * total_hard_scores + beta * total_soft_scores;
    best_hard_sat = curr_hard_sat;
    best_soft_sat = curr_soft_sat;
    not_improved = 0;
  }

  __syncthreads();

  for (int iter = 0; iter < n_iters; iter++) {
    if (not_improved > restart_threshold) {
      not_improved = 0;
      if (first_thread) printf("c Restarting at iter %d\n", iter);
      get_random_model(f_d, state, false, restart_flip);
      __syncthreads();
    }

    for (unsigned v = index + 1; v <= f_d.nVars; v += blockDim.x) {
      unsigned sat_hard_sum = 0;
      unsigned sat_soft_sum = 0;
      unsigned sat_hard_sum_orig = 0;
      unsigned sat_soft_sum_orig = 0;

      for (unsigned hard_index = 0; hard_index < f_d.nHardClauses;
           hard_index++) {
        for (unsigned i = f_d.hard_clauses[hard_index];
             i < f_d.hard_clauses[hard_index + 1]; ++i) {
          int lit = f_d.hard_clauses_lits[i];
          // Simulate var flip
          if (lit == v || -lit == v) lit = -lit;
          if (lit > 0) {
            if (MODEL_AT(lit) == 1) {
              sat_hard_sum += HARD_WEIGHTS_AT(hard_index);
              sat_hard_sum_orig++;
              break;
            }
          } else {
            if (MODEL_AT(-lit) == 0) {
              sat_hard_sum += HARD_WEIGHTS_AT(hard_index);
              sat_hard_sum_orig++;
              break;
            }
          }
        }
      }

      for (unsigned soft_index = 0; soft_index < f_d.nSoftClauses;
           soft_index++) {
        for (unsigned i = f_d.soft_clauses[soft_index];
             i < f_d.soft_clauses[soft_index + 1]; ++i) {
          int lit = f_d.soft_clauses_lits[i];
          // Simulate var flip
          if (lit == v || -lit == v) lit = -lit;
          if (lit > 0) {
            if (MODEL_AT(lit) == 1) {
              sat_soft_sum += SOFT_WEIGHTS_AT(soft_index);
              sat_soft_sum_orig += f_d.soft_weights_orig[soft_index];
              break;
            }
          } else {
            if (MODEL_AT(-lit) == 0) {
              sat_soft_sum += SOFT_WEIGHTS_AT(soft_index);
              sat_soft_sum_orig += f_d.soft_weights_orig[soft_index];
              break;
            }
          }
        }
      }

      float score = alpha * sat_hard_sum + beta * sat_soft_sum;
      VAR_SCORES_AT(v) = score;
      VAR_HARD_ORIG_AT(v) = sat_hard_sum_orig;
      VAR_SOFT_ORIG_AT(v) = sat_soft_sum_orig;
    }

    __syncthreads();

    find_max_var(f_d, max_v, chosen_v, prev_score, state);

    if (first_thread) {
      if (VAR_SCORES_AT(max_v) > prev_score) {
        MODEL_AT(chosen_v) = 1 - MODEL_AT(chosen_v);
        local_opt = false;
      } else {
        local_opt = true;
        bool use_hard;
        float randf = curand_uniform(&state[index]);
        use_hard = (randf <= flip_hard_prob);

        if (f_d.nHardClauses > 0 &&
            ((use_hard && curr_hard_sat != f_d.nHardClauses) ||
             curr_soft_sat == f_d.sum_soft_weights)) {
          chosen_v = choose_var_hard(f_d, state);
        } else if ((!use_hard && curr_soft_sat != f_d.sum_soft_weights) ||
                   curr_hard_sat == f_d.nHardClauses) {
          chosen_v = choose_var_soft(f_d, state);
        }
        MODEL_AT(chosen_v) = 1 - MODEL_AT(chosen_v);

        randf = curand_uniform(&state[index]);
        smooth = (randf <= sp);

        not_improved++;
      }

      prev_score = VAR_SCORES_AT(chosen_v);
      curr_hard_sat = VAR_HARD_ORIG_AT(chosen_v);
      curr_soft_sat = VAR_SOFT_ORIG_AT(chosen_v);
    }

    __syncthreads();

    if (local_opt) {
      for (unsigned i = index; i < f_d.nHardClauses; i += blockDim.x) {
        if (smooth && HARD_WEIGHTS_AT(i) > 1 && HARD_SCORES_AT(i) > 0) {
          HARD_WEIGHTS_AT(i)--;
          prev_score -= alpha;
        }
        if (!smooth && HARD_SCORES_AT(i) == 0) {
          HARD_WEIGHTS_AT(i) += hard_inc;
        }
      }

      for (unsigned i = index; i < f_d.nSoftClauses; i += blockDim.x) {
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
      if (index == 0) {
        BEST_MODEL_AT(0) = iter;
        best_hard_sat = curr_hard_sat;
        best_soft_sat = curr_soft_sat;
        printf("c Updated best model(%d, %d) at %d\n", best_hard_sat,
               best_soft_sat, iter);
        soft_max += soft_max_delta;
      }

      for (unsigned i = index + 1; i <= f_d.nVars; i += blockDim.x) {
        BEST_MODEL_AT(i) = MODEL_AT(i);
      }
    }

    __syncthreads();
  }

  __syncthreads();
  print_final(index, f_d, curr_hard_sat, curr_soft_sat, best_hard_sat,
              best_soft_sat);
}
