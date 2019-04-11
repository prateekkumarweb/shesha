#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <cstdio>
#include "search_algs.h"

__global__ void search_alg1(int nVars, int nClauses, int hardWeight,
                           int *input_arr, int *clauses, int *weights,
                           int *model, int *best_model, float *scores,
                           int *d_sat, float alpha, float beta) {
  int index = threadIdx.x + blockDim.x * blockIdx.x;

  __shared__ int best_cost;
  __shared__ int current_cost;
  __shared__ int hard_sat;
  __shared__ int best_hard_sat;
  __shared__ int total_cost;

  if (index == 0) {
    total_cost = best_cost = current_cost = hard_sat = best_hard_sat = 0;
  }
  __syncthreads();

  if (index < nClauses) {
    bool s = false;
    for (int i = clauses[index]; i < clauses[index + 1]; i++) {
      int lit = input_arr[i];
      if (lit > 0) {
        if (model[lit - 1] == 1) {
          s = true;
        }
      } else {
        if (model[-lit - 1] == 0) {
          s = true;
        }
      }
      if (s)
        break;
    }
    d_sat[index] = s;
  }
  __syncthreads();

  if (index == 0) {
    int c = 0;
    for (int i = 0; i < nClauses; i++) {
      if (d_sat[i]) {
        if (weights[i] == hardWeight) {
          hard_sat++;
          best_hard_sat++;
        } else {
          c += weights[i];
        }
      }
      if (weights[i] != hardWeight) {
        total_cost += weights[i];
      }
    }
    best_cost = current_cost = c;
  }
  __syncthreads();

  for (int k = 0; k < 100000; k++) {
    if (index < nVars) {
      scores[index] = 0.0;
    }

    __syncthreads();

    for (int v = 0; v < nVars; v++) {
      if (index == 0) {
        model[v] = 1 - model[v];
      }
      __syncthreads();

      if (index < nClauses) {
        bool s = false;
        for (int i = clauses[index]; i < clauses[index + 1]; i++) {
          if (input_arr[i] > 0) {
            if (model[input_arr[i] - 1] == 1) {
              s = true;
            }
          } else {
            if (model[-input_arr[i] - 1] == 0) {
              s = true;
            }
          }
          if (s)
            break;
        }
        d_sat[index] = s;
      }
      __syncthreads();

      if (index == 0) {
        float c = 0.0;
        for (int j = 0; j < nClauses; j++) {
          if (d_sat[j]) {
            c += weights[j] * (weights[j] == hardWeight ? alpha : beta);
          }
        }
        scores[v] = c;
        model[v] = 1 - model[v];
      }
    }
    __syncthreads();

    if (index == 0) {
      int max_i = 0;
      for (int i = 1; i < nVars; i++) {
        if (scores[i] >= scores[max_i]) {
          max_i = i;
        }
      }

      model[max_i] = 1 - model[max_i];
    }

    __syncthreads();
    if (index < nClauses) {
      bool s = false;
      for (int i = clauses[index]; i < clauses[index + 1]; i++) {
        if (input_arr[i] > 0) {
          if (model[input_arr[i] - 1] == 1) {
            s = true;
          }
        } else {
          if (model[-input_arr[i] - 1] == 0) {
            s = true;
          }
        }
        if (s)
          break;
      }
      d_sat[index] = s;
    }
    __syncthreads();

    if (index == 0) {
      hard_sat = 0;
      current_cost = 0;
      int c = 0;
      for (int i = 0; i < nClauses; i++) {
        if (d_sat[i]) {
          if (weights[i] == hardWeight) {
            hard_sat++;
          } else {
            c += weights[i];
          }
        }
      }
      current_cost = c;

      if (hard_sat > best_hard_sat ||
          (hard_sat == best_hard_sat && current_cost > best_cost)) {
        best_cost = current_cost;
        best_hard_sat = hard_sat;

        for (int i = 0; i < nVars; i++) {
          best_model[i] = model[i];
        }
      }
    }
    __syncthreads();
  }

  if (index == 0)
    printf("Total cost: %d\nBest Cost: %d\nHard Sat: %d\n", total_cost,
           best_cost, best_hard_sat);
}
