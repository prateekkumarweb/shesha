#include <cuda_runtime.h>

#include <iostream>
#include <sstream>
#include <string>

#include "maxsat.h"

bool parse_formula(std::istream &in, MaxSATFormula &f) {
  std::string line;
  bool first_line = true;
  bool wcnf = true;

  f.soft_clauses.push_back(0);
  f.hard_clauses.push_back(0);

  f.sum_soft_weights = 0;

  while (std::getline(in, line)) {
    if (line == "") continue;
    if (line[0] == 'c') continue;

    weight_t hardWt;

    std::stringstream ss(line);
    std::string s;

    if (first_line) {
      ss >> s;
      if (s != "p") {
        return false;
      }
      ss >> s;
      if (s == "wcnf") {
        wcnf = true;
      } else if (s == "cnf") {
        wcnf = false;
      } else {
        return false;
      }
      ss >> f.nVars;
      ss >> f.nClauses;
      if (wcnf)
        ss >> hardWt;
      else
        hardWt = 10;
      f.hard_weight = hardWt;
      first_line = false;
    } else {
      weight_t tmpWt;
      int lit;

      if (wcnf)
        ss >> tmpWt;
      else
        tmpWt = 1;
      if (tmpWt == hardWt) {
        ss >> lit;
        while (lit != 0) {
          weight_t l = (weight_t)(lit > 0 ? (2 * lit) : (-2 * lit + 1));
          f.hard_clauses_lits.push_back(l);
          ss >> lit;
        }
        f.hard_clauses.push_back(f.hard_clauses_lits.size());
      } else {
        f.soft_weights.push_back(tmpWt);
        f.sum_soft_weights += tmpWt;
        ss >> lit;
        while (lit != 0) {
          weight_t l = (weight_t)(lit > 0 ? (2 * lit) : (-2 * lit + 1));
          f.soft_clauses_lits.push_back(l);
          ss >> lit;
        }
        f.soft_clauses.push_back(f.soft_clauses_lits.size());
      }
    }
  }

  return true;
}

void alloc_formula_d(MaxSATFormula &f, MaxSATFormula_d &f_d, unsigned nThreads,
                     unsigned nBlocks, std::vector<unsigned> &model) {
  f_d.nVars = f.nVars;
  f_d.nClauses = f.nClauses;
  f_d.nHardLits = f.hard_clauses_lits.size();
  f_d.nSoftLits = f.soft_clauses_lits.size();
  f_d.nHardClauses = f.hard_clauses.size() - 1;
  f_d.nSoftClauses = f.soft_clauses.size() - 1;
  f_d.sum_soft_weights = f.sum_soft_weights;

  f_d.nThreads = nThreads;
  f_d.nBlocks = nBlocks;

  cudaMalloc(&f_d.hard_clauses_lits, f_d.nHardLits * sizeof(unsigned));
  cudaMalloc(&f_d.soft_clauses_lits, f_d.nSoftLits * sizeof(unsigned));
  cudaMalloc(&f_d.hard_clauses, (f_d.nHardClauses + 1) * sizeof(unsigned));
  cudaMalloc(&f_d.soft_clauses, (f_d.nSoftClauses + 1) * sizeof(unsigned));
  cudaMalloc(&f_d.soft_weights_orig, f_d.nSoftClauses * sizeof(weight_t));

  cudaMemcpy(f_d.hard_clauses_lits, f.hard_clauses_lits.data(),
             f_d.nHardLits * sizeof(unsigned), cudaMemcpyHostToDevice);
  cudaMemcpy(f_d.soft_clauses_lits, f.soft_clauses_lits.data(),
             f_d.nSoftLits * sizeof(unsigned), cudaMemcpyHostToDevice);
  cudaMemcpy(f_d.hard_clauses, f.hard_clauses.data(),
             (f_d.nHardClauses + 1) * sizeof(unsigned), cudaMemcpyHostToDevice);
  cudaMemcpy(f_d.soft_clauses, f.soft_clauses.data(),
             (f_d.nSoftClauses + 1) * sizeof(unsigned), cudaMemcpyHostToDevice);
  cudaMemcpy(f_d.soft_weights_orig, f.soft_weights.data(),
             f_d.nSoftClauses * sizeof(weight_t), cudaMemcpyHostToDevice);

  cudaMalloc(&f_d.hard_weights, f_d.nHardClauses * sizeof(weight_t) * nBlocks);
  cudaMalloc(&f_d.soft_weights, f_d.nSoftClauses * sizeof(weight_t) * nBlocks);

  cudaMalloc(&f_d.model, (f_d.nVars + 1) * sizeof(unsigned) * nBlocks);
  cudaMalloc(&f_d.hard_scores, f_d.nHardClauses * sizeof(weight_t) * nBlocks);
  cudaMalloc(&f_d.soft_scores, f_d.nSoftClauses * sizeof(weight_t) * nBlocks);

  cudaMalloc(&f_d.var_scores, (f_d.nVars + 1) * sizeof(float) * nBlocks);
  cudaMalloc(&f_d.var_hard_orig, (f_d.nVars + 1) * sizeof(weight_t) * nBlocks);
  cudaMalloc(&f_d.var_soft_orig, (f_d.nVars + 1) * sizeof(weight_t) * nBlocks);

  cudaMalloc(&f_d.best_model, (2 * f_d.nVars + 1) * sizeof(unsigned) * nBlocks);

  cudaMemcpy(f_d.model, model.data(), (f_d.nVars + 1) * sizeof(unsigned) * nBlocks,
             cudaMemcpyHostToDevice);
  cudaMemcpy(f_d.best_model, model.data(), (f_d.nVars + 1) * sizeof(unsigned),
             cudaMemcpyHostToDevice);
}

void free_formula_d(MaxSATFormula &f, MaxSATFormula_d &f_d) {
  delete[] f.best_model;

  cudaFree(f_d.hard_clauses_lits);
  cudaFree(f_d.soft_clauses_lits);
  cudaFree(f_d.hard_clauses);
  cudaFree(f_d.soft_clauses);
  cudaFree(f_d.soft_weights_orig);

  cudaFree(f_d.hard_weights);
  cudaFree(f_d.soft_weights);

  cudaFree(f_d.model);
  cudaFree(f_d.hard_scores);
  cudaFree(f_d.soft_scores);
  cudaFree(f_d.var_scores);
  cudaFree(f_d.var_hard_orig);
  cudaFree(f_d.var_soft_orig);

  cudaFree(f_d.best_model);
}

void print_model(MaxSATFormula &f, unsigned nBlocks) {
  printf("c Original Hard: %ld Soft: %ld\n", f.hard_clauses.size() - 1,
         f.sum_soft_weights);

  weight_t *hard_ss = new weight_t[nBlocks];
  weight_t *soft_ss = new weight_t[nBlocks];
  weight_t best_hard_ss = 0;
  weight_t best_soft_ss = 0;
  int best_block = -1;
  int best_offset = 0;

#define MODEL(b, i) f.best_model[i + (2 * f.nVars + 1) * b]

  for (unsigned b = 0; b < nBlocks; b++) {
    weight_t ss = 0;
    unsigned offset = MODEL(b, 0) ? f.nVars : 0;
    for (unsigned i = 0; i < f.hard_clauses.size() - 1; i++) {
      unsigned s = 0;
      for (unsigned j = f.hard_clauses[i]; j < f.hard_clauses[i + 1]; j++) {
        unsigned l = f.hard_clauses_lits[j];
        if ((MODEL(b, (l >> 1) + offset) ^ 1) == (l & 1)) {
          s = 1;
          break;
        }
      }
      ss += s;
    }
    hard_ss[b] = ss;

    ss = 0;
    for (unsigned i = 0; i < f.soft_clauses.size() - 1; i++) {
      unsigned s = 0;
      for (unsigned j = f.soft_clauses[i]; j < f.soft_clauses[i + 1]; j++) {
        unsigned l = f.soft_clauses_lits[j];
        if ((MODEL(b, (l >> 1) + offset) ^ 1) == (l & 1)) {
          s = 1;
          break;
        }
      }
      ss += s * f.soft_weights[i];
    }
    soft_ss[b] = ss;

    printf("c Block: %d Last Updated Flag %d\n", b, MODEL(b, 0));
    printf("c Block: %d Hard: %ld Soft: %ld Cost: %ld\n", b, hard_ss[b],
           soft_ss[b], f.sum_soft_weights - soft_ss[b]);

    if (best_block == -1 || hard_ss[b] > best_hard_ss ||
        (hard_ss[b] == best_hard_ss && soft_ss[b] > best_soft_ss)) {
      if (hard_ss[b] + 1 == f.hard_clauses.size()) {
        printf("o %ld\n", f.sum_soft_weights - soft_ss[b]);
      }
      best_block = b;
      best_offset = offset;
      best_hard_ss = hard_ss[b];
      best_soft_ss = soft_ss[b];
    }
  }

  if (best_block != -1) {
    printf("c Best Block: %d Hard: %ld Soft: %ld Cost: %ld\n", best_block,
           best_hard_ss, best_soft_ss, f.sum_soft_weights - best_soft_ss);

    if (best_hard_ss + 1 == f.hard_clauses.size())
      printf("o %ld\n", f.sum_soft_weights - best_soft_ss);

    printf("v");
    for (unsigned i = 1; i <= f.nVars; ++i) {
      printf("%s%d", MODEL(best_block, i + best_offset) == 1 ? " " : " -", i);
    }
    printf("\n");
  }

#undef MODEL

  delete[] hard_ss;
  delete[] soft_ss;
}
