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

  unsigned nThreads;
  unsigned nBlocks;

  // Doesn't change
  int* hard_clauses_lits;       // nHardLits
  int* soft_clauses_lits;       // nSoftLits
  unsigned* hard_clauses;       // nHardClauses + 1
  unsigned* soft_clauses;       // nSoftClauses + 1
  unsigned* soft_weights_orig;  // nSoftClauses

  // Changes
  unsigned* hard_weights;  // nHardClauses
  unsigned* soft_weights;  // nSoftClauses

  int* model;        // nVars + 1
  int* hard_scores;  // nHardClauses
  int* soft_scores;  // nSoftClauses

  float* var_scores;        // nVars + 1
  unsigned* var_hard_orig;  // nVars + 1
  unsigned* var_soft_orig;  // nVars + 1

  int* best_model;  // nVars + 1
};

bool parse_formula(std::istream& in, MaxSATFormula& f);
void alloc_formula_d(MaxSATFormula& f, MaxSATFormula_d& f_d, unsigned nThreads,
                     unsigned nBlocks);
void free_formula_d(MaxSATFormula& f, MaxSATFormula_d& f_d);
void print_model(std::ostream& out, MaxSATFormula& f, unsigned nBlocks);

#endif  // MAXSAT_H
