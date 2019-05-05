#include <curand.h>
#include <cstdio>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include "maxsat.h"
#include "search_algs.h"

__device__ void get_random_model(MaxSATFormula_d &f_d, curandState_t *state,
                                 bool update_best = true) {
  int index = threadIdx.x + blockDim.x * blockIdx.x;
  for (int var_index = index + 1; var_index <= f_d.nVars;
       var_index += blockDim.x) {
    float randf = curand_uniform(&state[index]);
    if (randf <= 0.5)
      f_d.model[var_index] = 1;
    else
      f_d.model[var_index] = 0;
    if (update_best) f_d.best_model[var_index] = f_d.model[var_index];
  }
}

__device__ void update_hard_scores(MaxSATFormula_d &f_d) {
  int index = threadIdx.x + blockDim.x * blockIdx.x;
  for (int hard_index = index; hard_index < f_d.nHardClauses;
       hard_index += blockDim.x) {
    f_d.hard_weights[hard_index] = 1;
    int sat = 0;
    for (unsigned i = f_d.hard_clauses[hard_index];
         i < f_d.hard_clauses[hard_index + 1]; ++i) {
      int lit = f_d.hard_clauses_lits[i];
      if (lit > 0) {
        if (f_d.model[lit] == 1) {
          sat = f_d.hard_weights[hard_index];
          break;
        }
      } else {
        if (f_d.model[-lit] == 0) {
          sat = f_d.hard_weights[hard_index];
          break;
        }
      }
    }
    f_d.hard_scores[hard_index] = sat;
  }
}

__device__ void update_soft_scores(MaxSATFormula_d &f_d) {
  int index = threadIdx.x + blockDim.x * blockIdx.x;
  for (int soft_index = index; soft_index < f_d.nSoftClauses;
       soft_index += blockDim.x) {
    int sat = 0;
    for (unsigned i = f_d.soft_clauses[soft_index];
         i < f_d.soft_clauses[soft_index + 1]; ++i) {
      int lit = f_d.soft_clauses_lits[i];
      if (lit > 0) {
        if (f_d.model[lit] == 1) {
          sat = f_d.soft_weights_orig[soft_index];
          f_d.soft_temp[soft_index] = f_d.soft_weights[soft_index];
          break;
        }
      } else {
        if (f_d.model[-lit] == 0) {
          sat = f_d.soft_weights_orig[soft_index];
          f_d.soft_temp[soft_index] = f_d.soft_weights[soft_index];
          break;
        }
      }
    }
    f_d.soft_scores[soft_index] = sat;
  }
}

__device__ unsigned choose_var_hard(MaxSATFormula_d &f_d,
                                    curandState_t *state) {
  int index = threadIdx.x + blockDim.x * blockIdx.x;
  int hard_index = -1;
  int count = 0;
  for (auto i = 0; i < f_d.nHardClauses; i++) {
    if (f_d.hard_scores[i] == 0) count++;
  }
  int idx = curand_uniform(&state[index]) * count;
  count = -1;
  // TODO: better to recompute or store?
  for (auto i = 0; i < f_d.nHardClauses; i++) {
    if (f_d.hard_scores[i] == 0) count++;
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
    if (f_d.soft_scores[i] == 0) count++;
  }
  int idx = curand_uniform(&state[index]) * count;
  count = -1;
  // TODO: better to recompute or store?
  for (auto i = 0; i < f_d.nSoftClauses; i++) {
    if (f_d.soft_scores[i] == 0) count++;
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
    float p =
        f_d.var_scores[i] / (f_d.var_scores[chosen_v] + f_d.var_scores[i]);
    float randf = curand_uniform(&state[index]);
    if (randf <= p) {
      chosen_v = i;
    }
    if (f_d.var_scores[max_v] < f_d.var_scores[i]) {
      max_v = i;
    }
  }
  __shared__ int r[1024];
  __shared__ int cr[1024];
  r[index] = max_v;
  cr[index] = chosen_v;
  __syncthreads();
  for (int size = blockDim.x / 2; size > 0; size >>= 1) {
    if (index < size) {
      float p = f_d.var_scores[cr[size + index]] /
                (f_d.var_scores[cr[size + index]] + f_d.var_scores[cr[index]]);
      float randf = curand_uniform(&state[index]);
      if (randf <= p) {
        cr[index] = cr[score + index];
      }
      if (f_d.var_scores[r[size + index]] > f_d.var_scores[r[index]]) {
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
    score += f_d.hard_scores[i];
  }
  __shared__ int r[1024];
  r[index] = score;
  __syncthreads();
  for (int size = blockDim.x / 2; size > 0; size >>= 1) {
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
    score += f_d.soft_scores[i];
  }
  __shared__ int r[1024];
  r[index] = score;
  __syncthreads();
  for (int size = blockDim.x / 2; size > 0; size >>= 1) {
    if (index < size) {
      r[index] += r[index + size];
    }
    __syncthreads();
  }
  if (index == 0) {
    curr_soft_sat = r[0];
  }

  score = 0;
  for (int i = index; i < f_d.nSoftClauses; i += blockDim.x) {
    score += f_d.soft_temp[i];
  }
  r[index] = score;
  __syncthreads();
  for (int size = blockDim.x / 2; size > 0; size >>= 1) {
    if (index < size) {
      r[index] += r[index + size];
    }
    __syncthreads();
  }
  if (index == 0) {
    soft_score = r[0];
  }
}

__device__ void print_final(int index, MaxSATFormula_d &f_d,
                            unsigned &curr_hard_sat, unsigned &curr_soft_sat,
                            unsigned &best_hard_sat, unsigned &best_soft_sat) {
  if (index == 0) {
    printf("c Last: %d %d\n", curr_hard_sat, curr_soft_sat);
    // printf("v");
    // for (int i = 1; i <= f_d.nVars; i++) {
    //   printf(" %d", f_d.model[i] > 0 ? i : -i);
    // }
    // printf("\n");
    printf("c Best: %d %d\n", best_hard_sat, best_soft_sat);
    // printf("b");
    // for (int i = 1; i <= f_d.nVars; i++) {
    //   printf(" %d", f_d.best_model[i] > 0 ? i : -i);
    // }
    // printf("\n");
  }
}

__global__ void search_alg1(MaxSATFormula_d f_d, unsigned hard_weight,
                            float alpha, float beta, float sp,
                            float flip_hard_prob, int hard_inc, int n_iters,
                            int restart_threshold, curandState_t *state) {
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

  if (first_thread) {
    best_hard_sat = curr_hard_sat = best_soft_sat = curr_soft_sat = 0;
    prev_score = 0.0;
    curr_hard_sat = 0;
    curr_soft_sat = 0;
    max_v = 1;
    chosen_v = 1;
  }

  // get_random_model(f_d, state);

  __syncthreads();

  update_hard_scores(f_d);
  update_soft_scores(f_d);

  __syncthreads();

  compute_total_hard_score(f_d, curr_hard_sat, total_hard_scores);
  compute_total_soft_score(f_d, curr_soft_sat, total_soft_scores);

  if (first_thread) {
    prev_score =
        alpha * total_hard_scores * hard_weight + beta * total_soft_scores;
    best_hard_sat = curr_hard_sat;
    best_soft_sat = curr_soft_sat;
    not_improved = 0;
  }

  __syncthreads();

  for (int iter = 0; iter < n_iters; iter++) {
    if (not_improved > restart_threshold) {
      not_improved = 0;
      get_random_model(f_d, state, false);
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
            if (f_d.model[lit] == 1) {
              sat_hard_sum += f_d.hard_weights[hard_index];
              sat_hard_sum_orig++;
              break;
            }
          } else {
            if (f_d.model[-lit] == 0) {
              sat_hard_sum += f_d.hard_weights[hard_index];
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
            if (f_d.model[lit] == 1) {
              sat_soft_sum += f_d.soft_weights[soft_index];
              sat_soft_sum_orig += f_d.soft_weights_orig[soft_index];
              break;
            }
          } else {
            if (f_d.model[-lit] == 0) {
              sat_soft_sum += f_d.soft_weights[soft_index];
              sat_soft_sum_orig += f_d.soft_weights_orig[soft_index];
              break;
            }
          }
        }
      }

      float score = alpha * sat_hard_sum * hard_weight + beta * sat_soft_sum;
      f_d.var_scores[v] = score;
      f_d.var_hard_orig[v] = sat_hard_sum_orig;
      f_d.var_soft_orig[v] = sat_soft_sum_orig;
    }

    __syncthreads();

    find_max_var(f_d, max_v, chosen_v, prev_score, state);

    if (first_thread) {
      if (f_d.var_scores[max_v] >= prev_score) {
        f_d.model[chosen_v] = 1 - f_d.model[chosen_v];
        local_opt = false;
      } else {
        local_opt = true;
        bool use_hard;
        float randf = curand_uniform(&state[index]);
        if (randf <= flip_hard_prob)
          use_hard = true;
        else
          use_hard = false;

        // TODO: check if this is strong enough
        if ((use_hard && curr_hard_sat != f_d.nHardClauses) ||
            curr_soft_sat == f_d.sum_soft_weights) {
          chosen_v = choose_var_hard(f_d, state);
        } else if ((!use_hard && curr_soft_sat != f_d.sum_soft_weights) ||
                   curr_hard_sat == f_d.nHardClauses) {
          chosen_v = choose_var_soft(f_d, state);
        }
        f_d.model[chosen_v] = 1 - f_d.model[chosen_v];

        randf = curand_uniform(&state[index]);
        smooth = (randf <= sp);

        not_improved++;
      }

      prev_score = f_d.var_scores[chosen_v];
      curr_hard_sat = f_d.var_hard_orig[chosen_v];
      curr_soft_sat = f_d.var_soft_orig[chosen_v];
    }

    __syncthreads();

    if (local_opt) {
      for (unsigned i = index; i < f_d.nHardClauses; i += blockDim.x) {
        if (smooth && f_d.hard_weights[i] > 1 && f_d.hard_scores[i] > 0) {
          f_d.hard_weights[i]--;
          prev_score -= alpha * hard_weight;
        }
        if (!smooth && f_d.hard_scores[i] == 0) {
          f_d.hard_weights[i] += hard_inc;
        }
      }

      for (unsigned i = index; i < f_d.nSoftClauses; i += blockDim.x) {
        if (smooth && f_d.soft_weights[i] > 1 && f_d.soft_scores[i] > 0) {
          f_d.soft_weights[i]--;
          prev_score -= beta;
        }
        if (!smooth && f_d.soft_scores[i] == 0 &&
            f_d.soft_weights[i] < hard_weight) {
          f_d.soft_weights[i]++;
        }
      }
    }

    if (curr_hard_sat > best_hard_sat ||
        (curr_hard_sat == best_hard_sat && curr_soft_sat > best_soft_sat)) {
      if (index == 0) {
        best_hard_sat = curr_hard_sat;
        best_soft_sat = curr_soft_sat;
      }

      for (unsigned i = index + 1; i <= f_d.nVars; i += blockDim.x) {
        f_d.best_model[i] = f_d.model[i];
      }
    }

    __syncthreads();
  }

  __syncthreads();
  print_final(index, f_d, curr_hard_sat, curr_soft_sat, best_hard_sat,
              best_soft_sat);
}
