#include <iostream>
#include <sstream>
#include <string>

#include "maxsat.h"

bool parse_formula(std::istream &in, MaxSATFormula &f) {
  std::string line;
  bool first_line = true;

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
      if (s != "wcnf") {
        return false;
      }
      ss >> f.nVars;
      ss >> f.nClauses;
      ss >> hardWt;
      f.hard_weight = hardWt;
      first_line = false;
    } else {
      unsigned tmpWt;
      int lit;

      ss >> tmpWt;
      if (tmpWt == hardWt) {
        ss >> lit;
        while (lit != 0) {
          f.hard_clauses_lits.push_back(lit);
          ss >> lit;
        }
        f.hard_clauses.push_back(f.hard_clauses_lits.size());
      } else {
        f.soft_weights.push_back(tmpWt);
        f.sum_soft_weights += tmpWt;
        ss >> lit;
        while (lit != 0) {
          f.soft_clauses_lits.push_back(lit);
          ss >> lit;
        }
        f.soft_clauses.push_back(f.soft_clauses_lits.size());
      }
    }
  }

  f.model = new int[f.nVars + 1];

  return true;
}

void alloc_formuala_d(MaxSATFormula &f, MaxSATFormula_d &f_d,
                      unsigned nThreads) {
  f_d.nVars = f.nVars;
  f_d.nClauses = f.nClauses;
  f_d.nHardLits = f.hard_clauses_lits.size();
  f_d.nSoftLits = f.soft_clauses_lits.size();
  f_d.nHardClauses = f.hard_clauses.size() - 1;
  f_d.nSoftClauses = f.soft_clauses.size() - 1;
  f_d.sum_soft_weights = f.sum_soft_weights;
  f_d.nThreads = nThreads;

  cudaMalloc(&f_d.hard_clauses_lits, f_d.nHardLits * sizeof(int));
  cudaMalloc(&f_d.soft_clauses_lits, f_d.nSoftLits * sizeof(int));
  cudaMalloc(&f_d.hard_clauses, (f_d.nHardClauses + 1) * sizeof(unsigned));
  cudaMalloc(&f_d.soft_clauses, (f_d.nSoftClauses + 1) * sizeof(unsigned));
  cudaMalloc(&f_d.soft_weights, f_d.nSoftClauses * sizeof(unsigned));
  cudaMalloc(&f_d.soft_weights_orig, f_d.nSoftClauses * sizeof(unsigned));
  cudaMalloc(&f_d.model, (f_d.nVars + 1) * sizeof(int));
  cudaMalloc(&f_d.best_model, (f_d.nVars + 1) * sizeof(int));
  cudaMalloc(&f_d.hard_scores, f_d.nHardClauses * sizeof(int));
  cudaMalloc(&f_d.soft_scores, f_d.nSoftClauses * sizeof(int));
  cudaMalloc(&f_d.hard_temp, f_d.nHardClauses * sizeof(int));
  cudaMalloc(&f_d.soft_temp, f_d.nSoftClauses * sizeof(int));
  cudaMalloc(&f_d.hard_weights, f_d.nHardClauses * sizeof(unsigned));
  cudaMalloc(&f_d.var_scores, (f_d.nVars + 1) * sizeof(float));
  cudaMalloc(&f_d.var_hard_orig, (f_d.nVars + 1) * sizeof(unsigned));
  cudaMalloc(&f_d.var_soft_orig, (f_d.nVars + 1) * sizeof(unsigned));

  cudaMemcpy(f_d.hard_clauses_lits, f.hard_clauses_lits.data(),
             f_d.nHardLits * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(f_d.soft_clauses_lits, f.soft_clauses_lits.data(),
             f_d.nSoftLits * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(f_d.hard_clauses, f.hard_clauses.data(),
             (f_d.nHardClauses + 1) * sizeof(unsigned), cudaMemcpyHostToDevice);
  cudaMemcpy(f_d.soft_clauses, f.soft_clauses.data(),
             (f_d.nSoftClauses + 1) * sizeof(unsigned), cudaMemcpyHostToDevice);
  cudaMemcpy(f_d.soft_weights, f.soft_weights.data(),
             f_d.nSoftClauses * sizeof(unsigned), cudaMemcpyHostToDevice);
  cudaMemcpy(f_d.soft_weights_orig, f.soft_weights.data(),
             f_d.nSoftClauses * sizeof(unsigned), cudaMemcpyHostToDevice);
  cudaMemcpy(f_d.model, f.model, (f_d.nVars + 1) * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(f_d.best_model, f.model, (f_d.nVars + 1) * sizeof(int),
             cudaMemcpyHostToDevice);
}

void free_formula_d(MaxSATFormula_d &f_d) {
  cudaFree(f_d.hard_clauses_lits);
  cudaFree(f_d.soft_clauses_lits);
  cudaFree(f_d.hard_clauses);
  cudaFree(f_d.soft_clauses);
  cudaFree(f_d.soft_weights);
  cudaFree(f_d.soft_weights_orig);
  cudaFree(f_d.model);
  cudaFree(f_d.best_model);
  cudaFree(f_d.hard_scores);
  cudaFree(f_d.soft_scores);
  cudaFree(f_d.hard_temp);
  cudaFree(f_d.soft_temp);
  cudaFree(f_d.hard_weights);
  cudaFree(f_d.var_scores);
  cudaFree(f_d.var_hard_orig);
  cudaFree(f_d.var_soft_orig);
}

void free_formula(MaxSATFormula &f) { delete[] f.model; }

void print_model(std::ostream &out, MaxSATFormula &f) {
  unsigned ss = 0;
  for (auto i : f.soft_weights) {
    ss += i;
  }
  std::cout << "c Original Hard: " << f.hard_clauses.size() - 1
            << " Soft: " << ss << std::endl;
  unsigned tot = ss;

  ss = 0;
  for (int i = 0; i < f.hard_clauses.size() - 1; i++) {
    int s = 0;
    for (int j = f.hard_clauses[i]; j < f.hard_clauses[i + 1]; j++) {
      int l = f.hard_clauses_lits[j];
      if (l > 0) {
        if (f.model[l] == 1) s = 1;
      } else {
        if (f.model[-l] == 0) s = 1;
      }
      if (s) break;
    }
    ss += s;
  }
  std::cout << "c Hard: " << ss << " ";

  ss = 0;
  for (int i = 0; i < f.soft_clauses.size() - 1; i++) {
    int s = 0;
    for (int j = f.soft_clauses[i]; j < f.soft_clauses[i + 1]; j++) {
      int l = f.soft_clauses_lits[j];
      if (l > 0) {
        if (f.model[l] == 1) s = 1;
      } else {
        if (f.model[-l] == 0) s = 1;
      }
      if (s) break;
    }
    ss += s * f.soft_weights[i];
  }
  std::cout << "Soft: " << ss << " Cost: " << tot - ss << std::endl;
  std::cout << "o " << tot - ss << std::endl;

  // out << 'v';
  // for (int i = 1; i <= f.nVars; ++i) {
  //   out << ' ' << (f.model[i] == 1 ? i : -i);
  // }
  // out << std::endl;
}
