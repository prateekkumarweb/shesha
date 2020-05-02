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

    unsigned hardWt;

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
      unsigned tmpWt;
      int lit;

      if (wcnf)
        ss >> tmpWt;
      else
        tmpWt = 1;
      if (tmpWt == hardWt) {
        ss >> lit;
        while (lit != 0) {
          int l = lit > 0 ? (2 * lit) : (-2 * lit + 1);
          f.hard_clauses_lits.push_back(l);
          ss >> lit;
        }
        f.hard_clauses.push_back(f.hard_clauses_lits.size());
      } else {
        f.soft_weights.push_back(tmpWt);
        f.sum_soft_weights += tmpWt;
        ss >> lit;
        while (lit != 0) {
          int l = lit > 0 ? (2 * lit) : (-2 * lit + 1);
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
                     unsigned nBlocks) {
  f_d.nVars = f.nVars;
  f_d.nClauses = f.nClauses;
  f_d.nHardLits = f.hard_clauses_lits.size();
  f_d.nSoftLits = f.soft_clauses_lits.size();
  f_d.nHardClauses = f.hard_clauses.size() - 1;
  f_d.nSoftClauses = f.soft_clauses.size() - 1;
  f_d.sum_soft_weights = f.sum_soft_weights;

  f_d.nThreads = nThreads;
  f_d.nBlocks = nBlocks;

  cudaMalloc(&f_d.hard_clauses_lits, f_d.nHardLits * sizeof(int));
  cudaMalloc(&f_d.soft_clauses_lits, f_d.nSoftLits * sizeof(int));
  cudaMalloc(&f_d.hard_clauses, (f_d.nHardClauses + 1) * sizeof(unsigned));
  cudaMalloc(&f_d.soft_clauses, (f_d.nSoftClauses + 1) * sizeof(unsigned));
  cudaMalloc(&f_d.soft_weights_orig, f_d.nSoftClauses * sizeof(unsigned));

  cudaMemcpy(f_d.hard_clauses_lits, f.hard_clauses_lits.data(),
             f_d.nHardLits * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(f_d.soft_clauses_lits, f.soft_clauses_lits.data(),
             f_d.nSoftLits * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(f_d.hard_clauses, f.hard_clauses.data(),
             (f_d.nHardClauses + 1) * sizeof(unsigned), cudaMemcpyHostToDevice);
  cudaMemcpy(f_d.soft_clauses, f.soft_clauses.data(),
             (f_d.nSoftClauses + 1) * sizeof(unsigned), cudaMemcpyHostToDevice);
  cudaMemcpy(f_d.soft_weights_orig, f.soft_weights.data(),
             f_d.nSoftClauses * sizeof(unsigned), cudaMemcpyHostToDevice);

  cudaMalloc(&f_d.hard_weights, f_d.nHardClauses * sizeof(unsigned) * nBlocks);
  cudaMalloc(&f_d.soft_weights, f_d.nSoftClauses * sizeof(unsigned) * nBlocks);

  cudaMalloc(&f_d.model, (f_d.nVars + 1) * sizeof(int) * nBlocks);
  cudaMalloc(&f_d.hard_scores, f_d.nHardClauses * sizeof(int) * nBlocks);
  cudaMalloc(&f_d.soft_scores, f_d.nSoftClauses * sizeof(int) * nBlocks);

  cudaMalloc(&f_d.var_scores, (f_d.nVars + 1) * sizeof(float) * nBlocks);
  cudaMalloc(&f_d.var_hard_orig, (f_d.nVars + 1) * sizeof(unsigned) * nBlocks);
  cudaMalloc(&f_d.var_soft_orig, (f_d.nVars + 1) * sizeof(unsigned) * nBlocks);

  cudaMalloc(&f_d.best_model, (f_d.nVars + 1) * sizeof(int) * nBlocks);

  cudaMemcpy(f_d.model, f.model, (f_d.nVars + 1) * sizeof(int) * nBlocks,
             cudaMemcpyHostToDevice);
  cudaMemcpy(f_d.best_model, f.model, (f_d.nVars + 1) * sizeof(int) * nBlocks,
             cudaMemcpyHostToDevice);
}

void free_formula_d(MaxSATFormula &f, MaxSATFormula_d &f_d) {
  delete[] f.model;

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

void print_model(std::ostream &out, MaxSATFormula &f, unsigned nBlocks) {
  unsigned tot = 0;
  for (auto i : f.soft_weights) {
    tot += i;
  }
  std::cout << "c Original Hard: " << f.hard_clauses.size() - 1
            << " Soft: " << tot << std::endl;

  unsigned *hard_ss = new unsigned[nBlocks];
  unsigned *soft_ss = new unsigned[nBlocks];
  unsigned best_hard_ss = 0;
  unsigned best_soft_ss = 0;
  int best_block = -1;

#define MODEL(b, i) f.model[i + (f.nVars + 1) * b]

  for (int b = 0; b < nBlocks; b++) {
    unsigned ss = 0;
    for (int i = 0; i < f.hard_clauses.size() - 1; i++) {
      int s = 0;
      for (int j = f.hard_clauses[i]; j < f.hard_clauses[i + 1]; j++) {
        int l = f.hard_clauses_lits[j];
        if ((MODEL(b, (l >> 1)) ^ 1) == (l & 1)) {
          s = 1;
          break;
        }
        // if (l > 0) {
        //   if (MODEL(b, l) == 1) s = 1;
        // } else {
        //   if (MODEL(b, -l) == 0) s = 1;
        // }
        // if (s) break;
      }
      ss += s;
    }
    hard_ss[b] = ss;

    ss = 0;
    for (int i = 0; i < f.soft_clauses.size() - 1; i++) {
      int s = 0;
      for (int j = f.soft_clauses[i]; j < f.soft_clauses[i + 1]; j++) {
        int l = f.soft_clauses_lits[j];
        if ((MODEL(b, (l >> 1)) ^ 1) == (l & 1)) {
          s = 1;
          break;
        }
        // if (l > 0) {
        //   if (MODEL(b, l) == 1) s = 1;
        // } else {
        //   if (MODEL(b, -l) == 0) s = 1;
        // }
        // if (s) break;
      }
      ss += s * f.soft_weights[i];
    }
    soft_ss[b] = ss;

    std::cout << "c Block: " << b << " Last Updated Iter " << MODEL(b, 0)
              << std::endl;
    std::cout << "c Block: " << b << " Hard: " << hard_ss[b]
              << " Soft: " << soft_ss[b] << " Cost: " << tot - soft_ss[b]
              << std::endl;

    if (best_block == -1 || hard_ss[b] > best_hard_ss ||
        (hard_ss[b] == best_hard_ss && soft_ss[b] > best_soft_ss)) {
      best_block = b;
      best_hard_ss = hard_ss[b];
      best_soft_ss = soft_ss[b];
    }
  }

  if (best_block != -1) {
    std::cout << "c Best Block: " << best_block << " Hard: " << best_hard_ss
              << " Soft: " << best_soft_ss << " Cost: " << tot - best_soft_ss
              << std::endl;
    if (best_hard_ss + 1 == f.hard_clauses.size())
      std::cout << "o " << tot - best_soft_ss << std::endl;

    out << 'v';
    for (int i = 1; i <= f.nVars; ++i) {
      out << ' ' << (MODEL(best_block, i) == 1 ? i : -i);
    }
    out << std::endl;
  }

#undef MODEL

  delete[] hard_ss;
  delete[] soft_ss;
}
