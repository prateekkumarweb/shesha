#if !defined(MAXSAT_H)
#define MAXSAT_H

#include <cstdint>
#define weight_t uint64_t

#include <vector>

struct MaxSATFormula {
  unsigned nVars;
  unsigned nClauses;
  weight_t hard_weight;
  weight_t sum_soft_weights;

  std::vector<unsigned> soft_clauses_lits;
  std::vector<unsigned> soft_clauses;
  std::vector<weight_t> soft_weights;
  std::vector<unsigned> hard_clauses_lits;
  std::vector<unsigned> hard_clauses;

  unsigned* best_model;
};

struct MaxSATFormula_d {
  unsigned nVars;
  unsigned nClauses;
  unsigned nHardLits;
  unsigned nSoftLits;
  unsigned nHardClauses;
  unsigned nSoftClauses;
  weight_t sum_soft_weights;

  unsigned nThreads;
  unsigned nBlocks;

  // Doesn't change
  unsigned* hard_clauses_lits;  // nHardLits
  unsigned* soft_clauses_lits;  // nSoftLits
  unsigned* hard_clauses;       // nHardClauses + 1
  unsigned* soft_clauses;       // nSoftClauses + 1
  weight_t* soft_weights_orig;  // nSoftClauses

  // Changes
  weight_t* hard_weights;  // nHardClauses
  weight_t* soft_weights;  // nSoftClauses

  unsigned* model;        // nVars + 1
  weight_t* hard_scores;  // nHardClauses
  weight_t* soft_scores;  // nSoftClauses

  float* var_scores;        // nVars + 1
  weight_t* var_hard_orig;  // nVars + 1
  weight_t* var_soft_orig;  // nVars + 1

  unsigned* best_model;  // 2 * nVars + 1
};

bool parse_formula(std::istream& in, MaxSATFormula& f);
void alloc_formula_d(MaxSATFormula& f, MaxSATFormula_d& f_d, unsigned nThreads,
                     unsigned nBlocks, std::vector<unsigned>& model);
void free_formula_d(MaxSATFormula& f, MaxSATFormula_d& f_d);
void print_model(MaxSATFormula& f, unsigned nBlocks);

#endif  // MAXSAT_H
