#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <cstdio>
#include <curand.h>
#include "maxsat.h"
#include "search_algs.h"

__global__ void search_alg1(MaxSATFormula_d f_d, float alpha, float beta, float sp, curandState_t *state) {
  int index = threadIdx.x + blockDim.x * blockIdx.x;

  __shared__ unsigned best_hard_sat;
  __shared__ unsigned curr_hard_sat;
  __shared__ unsigned best_soft_sat;
  __shared__ unsigned curr_soft_sat;
  __shared__ float best_score;
  __shared__ unsigned hard_weight;

  if (index == 0) {
    best_hard_sat = curr_hard_sat = best_soft_sat = curr_soft_sat = 0;
    hard_weight = 0;
  }

  __syncthreads();

  int var_index = index;
  int hard_index = index;
  int soft_index = index - f_d.nHardClauses;

  bool var_index_is_valid = (var_index >= 1 && var_index <= f_d.nVars);
  bool hard_index_is_valid = (hard_index >= 0 && hard_index < f_d.nHardClauses);
  bool soft_index_is_valid = (soft_index >= 0 && soft_index < f_d.nSoftClauses);
  bool first_thread = (index == 0);

  if (var_index_is_valid) {
    float randf = curand_uniform(&state[index]);
    if (randf <= 0.5) f_d.model[var_index] = 1;
    else f_d.model[var_index] = 0;
    f_d.best_model[var_index] = f_d.model[var_index];
  }

  __syncthreads();

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

  __syncthreads();

  if (first_thread) {
    best_score = 0.0;
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

  for (int iter = 0; iter < 1000; iter++) {

    if (var_index_is_valid) {
      f_d.var_scores[var_index] = 0.0;
    }

    __syncthreads();

    for (int v = 1; v <= f_d.nVars; ++v) {
      if (first_thread) {
        f_d.model[v] = 1 - f_d.model[v];
      }

      __syncthreads();

      if (hard_index_is_valid) {
        int s = 0;
        for (unsigned i = f_d.hard_clauses[hard_index]; i < f_d.hard_clauses[hard_index+1]; ++i) {
          int lit = f_d.hard_clauses_lits[i];
          if (lit > 0) {
            if (f_d.model[lit] == 1) {
              s = 1;
              break;
            }
          } else {
            if (f_d.model[-lit] == 0) {
              s = 1;
              break;
            }
          }
        }
        f_d.hard_scores[hard_index] = s;
      }

      if (soft_index_is_valid) {
        int s = 0;
        for (unsigned i = f_d.soft_clauses[soft_index]; i < f_d.soft_clauses[soft_index+1]; ++i) {
          int lit = f_d.soft_clauses_lits[i];
          if (lit > 0) {
            if (f_d.model[lit] == 1) {
              s = 1;
              break;
            }
          } else {
            if (f_d.model[-lit] == 0) {
              s = 1;
              break;
            }
          }
        }
        f_d.soft_scores[soft_index] = s;
      }

      __syncthreads();

      if (first_thread) {
        float score = 0.0;
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

      if (f_d.var_scores[index] < best_score) {
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

      if (curr_hard_sat > best_hard_sat || (curr_hard_sat == best_hard_sat && curr_soft_sat > best_soft_sat)) {
        best_hard_sat = curr_hard_sat;
        best_soft_sat = curr_soft_sat;

        for (unsigned i = 1; i <= f_d.nVars; ++i) {
          f_d.best_model[i] = f_d.model[i];
        }
      }
    }

    __syncthreads();
  }
}
