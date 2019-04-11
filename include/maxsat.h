#if !defined(MAXSAT_H)
#define MAXSAT_H

#include <vector>

struct MaxSATFormula {
  unsigned nVars;
  unsigned nClauses;
  unsigned hard_weight;

  std::vector<int> soft_clauses_lits;
  std::vector<unsigned> soft_clauses;
  std::vector<unsigned> soft_weights;
  std::vector<int> hard_clauses_lits;
  std::vector<unsigned> hard_clauses;

  int *model;
};

struct MaxSATFormula_d {
  unsigned nVars;
  unsigned nClauses;
  unsigned nHardLits;
  unsigned nSoftLits;
  unsigned nHardClauses;
  unsigned nSoftClauses;

  int* soft_clauses_lits;
  unsigned* soft_clauses;
  unsigned* soft_weights;
  int* hard_clauses_lits;
  unsigned* hard_clauses;

  int* model;
  int* best_model;
  int* soft_scores;
  int* hard_scores;
  unsigned *hard_weights;
  float* var_scores;
};

bool parse_formula(std::istream &in, MaxSATFormula &f);
void alloc_formuala_d(MaxSATFormula &f, MaxSATFormula_d &f_d);
void free_formula(MaxSATFormula &f);
void free_formula_d(MaxSATFormula_d &f_d);
void print_model(std::ostream &out, MaxSATFormula &f);

#endif // MAXSAT_H
