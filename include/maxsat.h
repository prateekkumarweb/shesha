#if !defined(MAXSAT_H)
#define MAXSAT_H

#include <vector>

struct MaxSATFormula {
  unsigned nVars;
  unsigned nClauses;
  unsigned hard_weight;
  unsigned sum_soft_weights;

  std::vector<int> soft_clauses_lits;
  std::vector<unsigned> soft_clauses;
  std::vector<unsigned> soft_weights;
  std::vector<int> hard_clauses_lits;
  std::vector<unsigned> hard_clauses;

  int* model;
};

struct MaxSATFormula_d {
  unsigned nVars;
  unsigned nClauses;
  unsigned nHardLits;
  unsigned nSoftLits;
  unsigned nHardClauses;
  unsigned nSoftClauses;
  unsigned sum_soft_weights;

  int* soft_clauses_lits; // nSoftLits
  unsigned* soft_clauses; // nSoftClauses + 1
  unsigned* soft_weights_orig; // nSoftClauses
  unsigned* soft_weights; // nSoftClauses
  int* hard_clauses_lits; // nHardLits
  unsigned* hard_clauses; // nHardClauses + 1
  unsigned* hard_weights; // nHardClauses

  int* model; // nVars + 1
  int* best_model; // nVars + 1
  int* soft_scores; // nSoftClauses
  int* hard_scores; // nHardClauses
  int* hard_temp; // nHardClauses
  int* soft_temp; // nSoftClauses
  float* var_scores; // nVars + 1
  unsigned* var_hard_orig; // nVars + 1
  unsigned* var_soft_orig; // nVars + 1

  unsigned nThreads;
};

bool parse_formula(std::istream& in, MaxSATFormula& f);
void alloc_formuala_d(MaxSATFormula& f, MaxSATFormula_d& f_d,
                      unsigned nThreads);
void free_formula(MaxSATFormula& f);
void free_formula_d(MaxSATFormula_d& f_d);
void print_model(std::ostream& out, MaxSATFormula& f);

#endif  // MAXSAT_H
