#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <cstdio>
#include <curand.h>
#include "maxsat.h"
#include "search_algs.h"

__device__ void get_random_model(MaxSATFormula_d &f_d, curandState_t *state, bool update_best = true) {
  int index = threadIdx.x + blockDim.x * blockIdx.x;
  int var_index = index;
  bool var_index_is_valid = (var_index >= 1 && var_index <= f_d.nVars);

  if (var_index_is_valid) {
    float randf = curand_uniform(&state[index]);
    if (randf <= 0.5) f_d.model[var_index] = 1;
    else f_d.model[var_index] = 0;
    if (update_best) f_d.best_model[var_index] = f_d.model[var_index];
  }
}

__device__ void update_hard_scores(MaxSATFormula_d &f_d) {
  int index = threadIdx.x + blockDim.x * blockIdx.x;
  int hard_index = index;
  bool hard_index_is_valid = (hard_index >= 0 && hard_index < f_d.nHardClauses);

  if (hard_index_is_valid) {
    f_d.hard_weights[hard_index] = 1;
    int sat = 0;    
    for (unsigned i = f_d.hard_clauses[hard_index]; i < f_d.hard_clauses[hard_index+1]; ++i) {
      int lit = f_d.hard_clauses_lits[i];
      if (lit > 0) {
        if (f_d.model[lit] == 1) {
          sat = 1;
          break;
        }
      } else {
        if (f_d.model[-lit] == 0) {
          sat = 1;
          break;
        }
      }
    }
    f_d.hard_scores[hard_index] = sat;
  }
}

__device__ void update_soft_scores(MaxSATFormula_d &f_d) {
  int index = threadIdx.x + blockDim.x * blockIdx.x;
  int soft_index = index - int(f_d.nHardClauses);
  bool soft_index_is_valid = (soft_index >= 0 && soft_index < f_d.nSoftClauses);

  if (soft_index_is_valid) {
    int sat = 0;    
    for (unsigned i = f_d.soft_clauses[soft_index]; i < f_d.soft_clauses[soft_index+1]; ++i) {
      int lit = f_d.soft_clauses_lits[i];
      if (lit > 0) {
        if (f_d.model[lit] == 1) {
          sat = 1;
          break;
        }
      } else {
        if (f_d.model[-lit] == 0) {
          sat = 1;
          break;
        }
      }
    }
    f_d.soft_scores[soft_index] = sat;
  }
}

__device__ void reset_var_scores(MaxSATFormula_d &f_d) {
  int index = threadIdx.x + blockDim.x * blockIdx.x;
  int var_index = index;
  bool var_index_is_valid = (var_index >= 1 && var_index <= f_d.nVars);

  if (var_index_is_valid) {
    f_d.var_scores[var_index] = 0.0;
  }
}

__global__ void search_alg1(MaxSATFormula_d f_d, unsigned hard_weight, float alpha, float beta, float sp, int n_iters, curandState_t *state) {
  int index = threadIdx.x + blockDim.x * blockIdx.x;

  bool first_thread = (index == 0);

  __shared__ unsigned best_hard_sat;
  __shared__ unsigned curr_hard_sat;
  __shared__ unsigned best_soft_sat;
  __shared__ unsigned curr_soft_sat;
  __shared__ float best_score;

  if (first_thread) {
    best_hard_sat = curr_hard_sat = best_soft_sat = curr_soft_sat = 0;
  }

  __syncthreads();

  get_random_model(f_d, state);

  __syncthreads();

  update_hard_scores(f_d);
  update_soft_scores(f_d);

  __syncthreads();

  if (first_thread) {
    best_score = 0.0;
    curr_hard_sat = 0;
    curr_soft_sat = 0;
    for (unsigned i = 0; i<f_d.nHardClauses; ++i) {
      curr_hard_sat += f_d.hard_scores[i];
      best_score += alpha * f_d.hard_scores[i] * f_d.hard_weights[i];
    }
    best_score *= hard_weight;
    for (unsigned i = 0; i<f_d.nSoftClauses; ++i) {
      curr_soft_sat += f_d.soft_scores[i] * f_d.soft_weights[i];
      best_score += beta * f_d.soft_scores[i] * f_d.soft_weights[i];
    }

    best_hard_sat = curr_hard_sat;
    best_soft_sat = curr_soft_sat;
  }

  __syncthreads();

  __shared__ int not_improved;

  if (first_thread) {
    printf("r");
    for (int i=1; i <= f_d.nVars; i++) {
      printf(" %d", f_d.model[i] > 0 ? i : -i);
    }
    printf("\n");
    not_improved = 0;
  }

  __syncthreads();

  for (int iter = 0; iter < n_iters; iter++) {

    if (best_hard_sat == f_d.nHardClauses && best_soft_sat == f_d.sum_soft_weights) {
      break;
    }

    if (not_improved > 25) {
      // printf("Restarting ...\n");
      not_improved = 0;
      // iter = 0;

      get_random_model(f_d, state, false);
    }

    reset_var_scores(f_d);

    __syncthreads();

    for (int v = 1; v <= f_d.nVars; ++v) {
      if (first_thread) {
        f_d.model[v] = 1 - f_d.model[v];
      }

      __syncthreads();

      update_hard_scores(f_d);
      update_soft_scores(f_d);

      __syncthreads();

      if (first_thread) {
        float score = 0.0;
        curr_hard_sat = 0;
        curr_soft_sat = 0;
        for (unsigned i = 0; i<f_d.nHardClauses; ++i) {
          curr_hard_sat += f_d.hard_scores[i];
          score += alpha * f_d.hard_scores[i] * f_d.hard_weights[i];
        }
        score *= hard_weight;
        for (unsigned i = 0; i<f_d.nSoftClauses; ++i) {
          curr_soft_sat += f_d.soft_scores[i] * f_d.soft_weights[i];
          score += beta * f_d.soft_scores[i] * f_d.soft_weights[i];
        }

        f_d.var_scores[v] = score;
        f_d.model[v] = 1 - f_d.model[v];
      }

      __syncthreads();

    }

    __syncthreads();

    if (first_thread) {
      unsigned max_v = 1;
      for (unsigned v = 2; v <= f_d.nVars; ++v) {
        if (f_d.var_scores[v] >= f_d.var_scores[max_v]) {
          max_v = v;
        }
      }

      f_d.model[max_v] = 1 - f_d.model[max_v];

      if (f_d.var_scores[max_v] < best_score) {
        bool smooth;
        float randf = curand_uniform(&state[index]);
        smooth = (randf <= sp);

        for (unsigned i = 0; i < f_d.nHardClauses; i++) {
          if (smooth && f_d.hard_weights[i] > 1 && f_d.hard_scores[i] > 0) {
            f_d.hard_weights[i]--;
          }
          if (!smooth && f_d.hard_scores[i] == 0) {
            f_d.hard_weights[i]++;
          }
        }
      }
    }

    __syncthreads();

    update_hard_scores(f_d);
    update_soft_scores(f_d);

    __syncthreads();

    if (first_thread) {
      float score = 0.0;
      curr_hard_sat = 0;
      curr_soft_sat = 0;
      for (unsigned i = 0; i<f_d.nHardClauses; ++i) {
        curr_hard_sat += f_d.hard_scores[i];
        score += alpha * f_d.hard_scores[i] * f_d.hard_weights[i];
      }
      score *= hard_weight;
      for (unsigned i = 0; i<f_d.nSoftClauses; ++i) {
        curr_soft_sat += f_d.soft_scores[i] * f_d.soft_weights[i];
        score += beta * f_d.soft_scores[i] * f_d.soft_weights[i];
      }
      best_score = score;

      if (curr_hard_sat > best_hard_sat || (curr_hard_sat == best_hard_sat && curr_soft_sat >= best_soft_sat)) {
        best_hard_sat = curr_hard_sat;
        best_soft_sat = curr_soft_sat;

        for (unsigned i = 1; i <= f_d.nVars; ++i) {
          f_d.best_model[i] = f_d.model[i];
        }
      } else {
        not_improved++;
      }
    }

    __syncthreads();

    // printf("%d %d %d %d\n", curr_hard_sat, best_hard_sat, curr_soft_sat, best_soft_sat);

  }

  __syncthreads();

  if (first_thread) {
    printf("c Last: %d %d\n", curr_hard_sat, curr_soft_sat);
    printf("v");
    for (int i=1; i <= f_d.nVars; i++) {
      printf(" %d", f_d.model[i] > 0 ? i : -i);
    }
    printf("\n");
    printf("c Best: %d %d\n", best_hard_sat, best_soft_sat);
    printf("b");
    for (int i=1; i <= f_d.nVars; i++) {
      printf(" %d", f_d.best_model[i] > 0 ? i : -i);
    }
    printf("\n");
  }
  __syncthreads();
}
