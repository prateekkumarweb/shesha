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
    for (; var_index <= f_d.nVars; var_index += f_d.nThreads) {
      float randf = curand_uniform(&state[index]);
      if (randf <= 0.5) f_d.model[var_index] = 1;
      else f_d.model[var_index] = 0;
      if (update_best) f_d.best_model[var_index] = f_d.model[var_index];
    }
  }
}

__device__ void update_hard_scores(MaxSATFormula_d &f_d) {
  int index = threadIdx.x + blockDim.x * blockIdx.x;
  int hard_index = index;
  bool hard_index_is_valid = (hard_index >= 0 && hard_index < f_d.nHardClauses);

  if (hard_index_is_valid) {
    for (; hard_index < f_d.nHardClauses; hard_index += f_d.nThreads) {
      f_d.hard_weights[hard_index] = 1;
      int sat = 0;    
      for (unsigned i = f_d.hard_clauses[hard_index]; i < f_d.hard_clauses[hard_index+1]; ++i) {
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
}

__device__ void update_soft_scores(MaxSATFormula_d &f_d) {
  int index = threadIdx.x + blockDim.x * blockIdx.x;
  int soft_index = index - int(f_d.nHardClauses);
  bool soft_index_is_valid = (soft_index >= 0 && soft_index < f_d.nSoftClauses);

  if (soft_index_is_valid) {
    for (; soft_index < f_d.nSoftClauses; soft_index += f_d.nThreads) {
      int sat = 0;
      for (unsigned i = f_d.soft_clauses[soft_index]; i < f_d.soft_clauses[soft_index+1]; ++i) {
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
}

__device__ unsigned choose_var_hard(MaxSATFormula_d &f_d, curandState_t *state) {
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
  int lit_index = curand_uniform(&state[index]) * (f_d.hard_clauses[hard_index+1]-f_d.hard_clauses[hard_index]);
  int lit = f_d.hard_clauses_lits[lit_index];
  if (lit < 0) {
    return -lit;
  }
  return lit;
}

__device__ unsigned choose_var_soft(MaxSATFormula_d &f_d, curandState_t *state) {
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
  int lit_index = curand_uniform(&state[index]) * (f_d.soft_clauses[soft_index+1]-f_d.soft_clauses[soft_index]);
  int lit = f_d.soft_clauses_lits[lit_index];
  if (lit < 0) {
    return -lit;
  }
  return lit;
}

__device__ void reset_var_scores(MaxSATFormula_d &f_d) {
  int index = threadIdx.x + blockDim.x * blockIdx.x;
  int var_index = index;
  bool var_index_is_valid = (var_index >= 1 && var_index <= f_d.nVars);

  if (var_index_is_valid) {
    for (; var_index <= f_d.nVars; var_index += f_d.nThreads)
      f_d.var_scores[var_index] = 0.0;
  }
}

__device__ void compute_total_hard_score(MaxSATFormula_d &f_d, unsigned &curr_hard_sat, unsigned &hard_score) {
  unsigned score = 0;
  int index = threadIdx.x + blockDim.x * blockIdx.x;
  for (int i = index; i<f_d.nHardClauses; i+=f_d.nThreads) {
    score += f_d.hard_scores[i];
  }
  __shared__ int r[1024];
  r[index] = score;
  __syncthreads();
  for (int size = blockDim.x/2; size > 0; size >>= 1) {
    if (index < size) {
      r[index] += r[index+size];
    }
    __syncthreads();
  }
  if (index == 0) {
    curr_hard_sat = hard_score = r[0];
  }
}

__device__ void compute_total_soft_score(MaxSATFormula_d &f_d, unsigned &curr_soft_sat, unsigned &soft_score) {
  unsigned score = 0;
  int index = threadIdx.x + blockDim.x * blockIdx.x;

  {
    for (int i = index; i<f_d.nSoftClauses; i+=f_d.nThreads) {
      score += f_d.soft_scores[i];
    }
    __shared__ int r[1024];
    r[index] = score;
    __syncthreads();
    for (int size = blockDim.x/2; size > 0; size >>= 1) {
      if (index < size) {
        r[index] += r[index+size];
      }
      __syncthreads();
    }
    if (index == 0) {
      curr_soft_sat = r[0];
    }
  }
  
  {
    score = 0;
    for (int i = index; i<f_d.nSoftClauses; i+=f_d.nThreads) {
      score += f_d.soft_temp[i];
    }
    __shared__ int r[1024];
    r[index] = score;
    __syncthreads();
    for (int size = blockDim.x/2; size > 0; size >>= 1) {
      if (index < size) {
        r[index] += r[index+size];
      }
      __syncthreads();
    }
    if (index == 0) {
      soft_score = r[0];
    }
  }
  
}

__device__ void print_final(int index, MaxSATFormula_d &f_d, unsigned &curr_hard_sat, unsigned &curr_soft_sat, unsigned &best_hard_sat, unsigned &best_soft_sat) {
  if (index == 0) {
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
}

__global__ void search_alg1(MaxSATFormula_d f_d, unsigned hard_weight, float alpha, float beta, float sp, float flip_hard_prob, int n_iters, int restart_threshold, curandState_t *state) {
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
  unsigned max_v = 1;
  float score;

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
    prev_score = 0.0;
    curr_hard_sat = 0;
    curr_soft_sat = 0;
  }

  __syncthreads();

  compute_total_hard_score(f_d, curr_hard_sat, total_hard_scores);
  compute_total_soft_score(f_d, curr_soft_sat, total_soft_scores);

  if (first_thread) {
    prev_score += alpha * total_hard_scores * hard_weight;
    prev_score += beta * total_soft_scores;
    best_hard_sat = curr_hard_sat;
    best_soft_sat = curr_soft_sat;
    // printf("r");
    // for (int i=1; i <= f_d.nVars; i++) {
    //   printf(" %d", f_d.model[i] > 0 ? i : -i);
    // }
    // printf("\n");
    not_improved = 0;
  }

  __syncthreads();

  for (int iter = 0; iter < n_iters; iter++) {

    if (best_hard_sat == f_d.nHardClauses && best_soft_sat == f_d.sum_soft_weights) {
      break;
    }

    if (not_improved > restart_threshold) {
      not_improved = 0;
      get_random_model(f_d, state, false);
    }

    __syncthreads();

    if (first_thread) max_v = 1;

    for (int v = 1; v <= f_d.nVars; ++v) {
      if (first_thread) {
        f_d.model[v] = 1 - f_d.model[v];
      }

      __syncthreads();

      update_hard_scores(f_d);
      update_soft_scores(f_d);

      __syncthreads();

      if (first_thread) {
        score = 0.0;
        curr_hard_sat = 0;
        curr_soft_sat = 0;
      }

      __syncthreads();

      compute_total_hard_score(f_d, curr_hard_sat, total_hard_scores);
      compute_total_soft_score(f_d, curr_soft_sat, total_soft_scores);

      if (first_thread) {
        score += alpha * total_hard_scores * hard_weight;
        score += beta * total_soft_scores;

        f_d.var_scores[v] = score;
        if (f_d.var_scores[v] >= f_d.var_scores[max_v]) {
          max_v = v;
        }
        f_d.model[v] = 1 - f_d.model[v];
      }

    }

    __syncthreads();

    if (first_thread) {
 
      if (f_d.var_scores[max_v] >= prev_score) {
        f_d.model[max_v] = 1 - f_d.model[max_v];
      } else {
        bool use_hard;
        unsigned flip_v;
        float randf = curand_uniform(&state[index]);
        if (randf <= flip_hard_prob) use_hard = true;
        else use_hard = false;

        // TODO: check if this is strong enough
        if ((use_hard && curr_hard_sat != f_d.nHardClauses) || curr_soft_sat == f_d.nSoftClauses) {
          flip_v = choose_var_hard(f_d, state);
        } else if ((!use_hard && curr_soft_sat != f_d.nSoftClauses) || curr_hard_sat == f_d.nHardClauses) {
          flip_v = choose_var_soft(f_d, state);
        }
        f_d.model[flip_v] = 1 - f_d.model[flip_v];

        bool smooth;
        randf = curand_uniform(&state[index]);
        smooth = (randf <= sp);

        for (unsigned i = 0; i < f_d.nHardClauses; i++) {
          if (smooth && f_d.hard_weights[i] > 1 && f_d.hard_scores[i] > 0) {
            f_d.hard_weights[i]--;
          }
          if (!smooth && f_d.hard_scores[i] == 0) {
            f_d.hard_weights[i]++;
          }
        }

        for (unsigned i = 0; i < f_d.nSoftClauses; i++) {
          if (smooth && f_d.soft_weights[i] > 1 && f_d.soft_scores[i] > 0) {
            f_d.soft_weights[i]--;
          }
          if (!smooth && f_d.soft_scores[i] == 0) {
            f_d.soft_weights[i]++;
          }
        }
        not_improved++;
      }
    }

    __syncthreads();

    update_hard_scores(f_d);
    update_soft_scores(f_d);

    __syncthreads();

    if (first_thread) {
      score = 0.0;
      curr_hard_sat = 0;
      curr_soft_sat = 0;
    }

    __syncthreads();

    compute_total_hard_score(f_d, curr_hard_sat, total_hard_scores);
    compute_total_soft_score(f_d, curr_soft_sat, total_soft_scores);

    if (first_thread) {
      score += alpha * total_hard_scores * hard_weight;
      score += beta * total_soft_scores;
      prev_score = score;

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

  __syncthreads();
  print_final(index, f_d, curr_hard_sat, curr_soft_sat, best_hard_sat, best_soft_sat);
}

// __global__ void search_alg2(MaxSATFormula_d f_d, unsigned hard_weight, float hard_prob, float sp, int n_iters, int restart_threshold, curandState_t *state) {
//   int index = threadIdx.x + blockDim.x * blockIdx.x;

//   __shared__ unsigned best_hard_sat;
//   __shared__ unsigned curr_hard_sat;
//   __shared__ unsigned best_soft_sat;
//   __shared__ unsigned curr_soft_sat;
//   __shared__ float best_hard_score;
//   __shared__ float best_soft_score;
//   unsigned max_v = 1;

//   if (index == 0) {
//     best_hard_sat = curr_hard_sat = best_soft_sat = curr_soft_sat = 0;
//   }

//   __syncthreads();

//   bool first_thread = (index == 0);

//   get_random_model(f_d, state);

//   __syncthreads();

//   update_hard_scores(f_d);
//   update_soft_scores(f_d);

//   __syncthreads();

//   compute_total_hard_score(f_d, curr_hard_sat, best_hard_score);
//   compute_total_soft_score(f_d, curr_soft_sat, best_soft_score);

//   if (first_thread) {
//     best_hard_sat = curr_hard_sat;
//     best_soft_sat = curr_soft_sat;
//   }

//   __syncthreads();

//   __shared__ int not_improved;

//   if (first_thread) {
//     // printf("r");
//     // for (int i=1; i <= f_d.nVars; i++) {
//     //   printf(" %d", f_d.model[i] > 0 ? i : -i);
//     // }
//     // printf("\n");
//     not_improved = 0;
//   }

//   __syncthreads();

//   for (int iter = 0; iter < n_iters; iter++) {

//     if (best_hard_sat == f_d.nHardClauses && best_soft_sat == f_d.sum_soft_weights) {
//       break;
//     }

//     if (not_improved > restart_threshold) {
//       not_improved = 0;
//       get_random_model(f_d, state, false);
//     }

//     bool use_hard = false;

//     if (first_thread) {
//       float randf = curand_uniform(&state[index]);
//       if (randf <= hard_prob) use_hard = true;
//       else use_hard = false;
//     }
//     __syncthreads();

//     if (first_thread) max_v = 1;

//     for (int v = 1; v <= f_d.nVars; ++v) {
//       if (first_thread) {
//         f_d.model[v] = 1 - f_d.model[v];
//       }

//       __syncthreads();

//       update_hard_scores(f_d);
//       update_soft_scores(f_d);

//       __syncthreads();

//       if (first_thread) {
//         float score = 0.0;
//         if (use_hard) {
//           score += compute_total_hard_score(f_d, curr_hard_sat);
//         } else {
//           score += compute_total_soft_score(f_d, curr_soft_sat);
//         }
//         f_d.var_scores[v] = score;
//         if (f_d.var_scores[v] >= f_d.var_scores[max_v]) {
//           max_v = v;
//         }
//         f_d.model[v] = 1 - f_d.model[v];
//       }

//     }

//     __syncthreads();

//     if (first_thread) {

//       f_d.model[max_v] = 1 - f_d.model[max_v];

//       if (use_hard && f_d.var_scores[max_v] < best_hard_score) {
//         bool smooth;
//         float randf = curand_uniform(&state[index]);
//         smooth = (randf <= sp);

//         for (unsigned i = 0; i < f_d.nHardClauses; i++) {
//           if (smooth && f_d.hard_weights[i] > 1 && f_d.hard_scores[i] > 0) {
//             f_d.hard_weights[i]--;
//           }
//           if (!smooth && f_d.hard_scores[i] == 0) {
//             f_d.hard_weights[i]++;
//           }
//         }
//         not_improved++;
//       }

//       if (!use_hard && f_d.var_scores[max_v] < best_soft_score) {
//         bool smooth;
//         float randf = curand_uniform(&state[index]);
//         smooth = (randf <= sp);

//         for (unsigned i = 0; i < f_d.nSoftClauses; i++) {
//           if (smooth && f_d.soft_weights[i] > 1 && f_d.soft_scores[i] > 0) {
//             f_d.soft_weights[i]--;
//           }
//           if (!smooth && f_d.soft_scores[i] == 0) {
//             f_d.soft_weights[i]++;
//           }
//         }
//         not_improved++;
//       }
//     }

//     __syncthreads();

//     update_hard_scores(f_d);
//     update_soft_scores(f_d);
    
//     __syncthreads();

//     if (first_thread) {
//       curr_hard_sat = 0;
//       curr_soft_sat = 0;
//       best_hard_score = compute_total_hard_score(f_d, curr_hard_sat);
//       best_soft_score = compute_total_soft_score(f_d, curr_soft_sat);

//       if (curr_hard_sat > best_hard_sat || (curr_hard_sat == best_hard_sat && curr_soft_sat > best_soft_sat)) {
//         best_hard_sat = curr_hard_sat;
//         best_soft_sat = curr_soft_sat;

//         for (unsigned i = 1; i <= f_d.nVars; ++i) {
//           f_d.best_model[i] = f_d.model[i];
//         }
//       }
//     }

//     __syncthreads();

//   }

//   __syncthreads();
//   print_final(index, f_d, curr_hard_sat, curr_soft_sat, best_hard_sat, best_soft_sat);
// }
