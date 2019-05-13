#include <curand.h>
#include <getopt.h>
#include <signal.h>
#include <algorithm>
#include <cassert>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "core/Solver.h"
#include "core/SolverTypes.h"

using namespace Glucose;

#define gpuErrchk(ans) \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort) exit(code);
  }
}

#include "maxsat.h"
#include "search_algs.h"

const struct option long_options[] = {
    {"help", no_argument, 0, 'h'},
    {"alpha", required_argument, 0, 0},
    {"beta", required_argument, 0, 0},
    {"sp", required_argument, 0, 0},
    {"hard_prob", required_argument, 0, 0},
    {"hard_inc", required_argument, 0, 0},
    {"iters", required_argument, 0, 'n'},
    {"restart_thresh", required_argument, 0, 0},
    {"restart_flip", required_argument, 0, 0},
    {"nblocks", required_argument, 0, 0},
    {"nthreads", required_argument, 0, 0},
    {0, 0, 0, 0}};

const std::string option_desc[]{
    "Display help",
    "Alpha for score",
    "Beta for score",
    "smoothening probability",
    "probability of choosing hard clause when local opt is hit",
    "Increment to internal hard weight when a hard clause is unsatisfied",
    "Number of iterations",
    "Number of iterations without improvement between two restarts",
    "Number of flips to make at restart",
    "Number of blocks to use",
    "Number of threads per block to use",
    ""};

void display_help(std::string exec) {
  std::cout << "Toolname version 0.1" << std::endl;
  std::cout << "\n"
            << "Description ..." << std::endl;

  std::cout << "\nUsage:\n";
  std::cout << " " << exec << " [Options]"
            << "\n\n";

  std::cout << "Options:\n";
  for (int i = 0; long_options[i].name != 0; i++) {
    if (long_options[i].val)
      std::cout << '-' << char(long_options[i].val) << ", ";
    std::cout << std::left << "--" << std::setw(8) << long_options[i].name
              << "\t" << option_desc[i] << "\n";
  }
}

void display_error(std::string err) {
  std::cout << err << std::endl;
  std::cout << "Use --help option to know about supported options."
            << std::endl;
  exit(1);
}

MaxSATFormula f;
MaxSATFormula_d f_d;

void sigint_handler(int sig) {
  static int call = 0;
  if (call > 0) {
    exit(0);
  }
  if (sig == SIGINT || sig == SIGTERM) {
    call++;
    std::cout << "c Exiting ..." << std::endl;
    cudaStream_t s;
    cudaStreamCreate(&s);
    cudaMemcpyAsync(f.model, f_d.best_model, (f.nVars + 1) * sizeof(int),
                    cudaMemcpyDeviceToHost, s);
    cudaStreamDestroy(s);

    print_model(std::cout, f);
    exit(1);
  }
}

unsigned find_soft_sat(MaxSATFormula &f, const std::vector<int> &model) {
  unsigned ss = 0;
  for (int i = 0; i < f.soft_clauses.size() - 1; i++) {
    int s = 0;
    for (int j = f.soft_clauses[i]; j < f.soft_clauses[i + 1]; j++) {
      int l = f.soft_clauses_lits[j];
      if (l > 0) {
        if (model[l] == 1) s = 1;
      } else {
        if (model[-l] == 0) s = 1;
      }
      if (s) break;
    }
    ss += s * f.soft_weights[i];
  }
  return ss;
}

void find_initial_model(MaxSATFormula &f, unsigned nModels = 3) {
  // TODO
  Solver solver;
  std::vector<Var> solver_vars;
  solver_vars.push_back(0);
  while (f.nVars >= solver.nVars()) solver.newVar();
  for (int i = 0; i < f.hard_clauses.size() - 1; i++) {
    vec<Lit> lits;
    for (int j = f.hard_clauses[i]; j < f.hard_clauses[i + 1]; j++) {
      int lit = f.hard_clauses_lits[j];
      int var = abs(lit) - 1;
      lits.push((lit > 0) ? mkLit(var) : ~mkLit(var));
    }
    solver.addClause(lits);
  }
  vec<Lit> assumps;
  unsigned best_sat = 0;
  for (unsigned k = 0; k < nModels; k++) {
    bool solved = solver.solve(assumps);
    std::vector<int> model(f.nVars + 1, 1);
    if (solved) {
      for (int i = 0; i < f.nVars; i++) {
        if (solver.model[i] == l_True)
          model[i + 1] = 1;
        else
          model[i + 1] = 0;
      }
      unsigned sat = find_soft_sat(f, model);
      if (sat > best_sat) {
        for (int v = 1; v <= f.nVars; v++) f.model[v] = model[v];
        best_sat = sat;
      }
      vec<Lit> c1;
      for (int i = 1; i <= f.nVars; i++) {
        c1.push((model[i] == 0) ? mkLit(i) : ~mkLit(i));
      }
      solver.addClause(c1);
    } else {
      break;
    }
  }

  if (best_sat != 0) {
    std::cout << "c Glucose initial cost: " << f.sum_soft_weights - best_sat
              << std::endl;
  }
}

__global__ void start_kernel(curandState_t *state) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  curand_init(9876543210l, idx, 0, &state[idx]);
}

int main(int argc, char **argv) {
  int opt_iters = 1000000;
  float alpha = 0.1, beta = 1.0, sp = 0.001, hard_prob = 0.99;
  int restart_thresh = 100;
  int restart_flip = 5;
  int nBlocks = 1;
  int nThreads = -1;
  int hard_inc = 100;

  while (1) {
    int option_index = 0;
    int c = getopt_long(argc, argv, "hn:", long_options, &option_index);
    if (c == -1) break;

    switch (c) {
      case 0:
        if (std::string(long_options[option_index].name) == "alpha") {
          if (optarg) {
            alpha = std::stof(optarg);
          }
        } else if (std::string(long_options[option_index].name) == "beta") {
          if (optarg) {
            beta = std::stof(optarg);
          }
        } else if (std::string(long_options[option_index].name) == "sp") {
          if (optarg) {
            sp = std::stof(optarg);
          }
        } else if (std::string(long_options[option_index].name) ==
                   "hard_prob") {
          if (optarg) {
            hard_prob = std::stof(optarg);
          }
        } else if (std::string(long_options[option_index].name) == "hard_inc") {
          if (optarg) {
            hard_inc = std::stoi(optarg);
          }
        } else if (std::string(long_options[option_index].name) ==
                   "restart_thresh") {
          if (optarg) {
            restart_thresh = std::stoi(optarg);
          }
        } else if (std::string(long_options[option_index].name) ==
                   "restart_flip") {
          if (optarg) {
            restart_flip = std::stoi(optarg);
          }
        } else if (std::string(long_options[option_index].name) == "nblocks") {
          if (optarg) {
            nBlocks = std::stoi(optarg);
          }
        } else if (std::string(long_options[option_index].name) == "nthreads") {
          if (optarg) {
            nThreads = std::stoi(optarg);
          }
        } else {
          display_error("Unrecognised argument: " +
                        std::string(long_options[option_index].name));
        }
        break;

      case 'h':
        display_help(std::string(argv[0]));
        exit(0);

      case 'n':
        opt_iters = std::stoi(optarg);
        break;

      default:
        display_error("Unrecognised argument: " + std::to_string(c));
    }
  }

  if (optind < argc) {
    display_error("Unrecognised argument: " + std::string(argv[optind]));
  }

  std::cout << "c no. of iterations " << opt_iters << std::endl;
  std::cout << "c alpha " << alpha << std::endl;
  std::cout << "c beta " << beta << std::endl;
  std::cout << "c sp " << sp << std::endl;
  std::cout << "c hard_prob " << hard_prob << std::endl;
  std::cout << "c restart_thresh " << restart_thresh << std::endl;
  std::cout << "c restart_flip " << restart_flip << std::endl;
  std::cout << "c hard_inc " << hard_inc << std::endl;
  std::cout << "c nBlocks " << nBlocks << std::endl;
  std::cout << "c nThreads " << nThreads << std::endl;

  signal(SIGINT, sigint_handler);
  signal(SIGTERM, sigint_handler);

  if (!parse_formula(std::cin, f)) {
    display_error("Invalid input!");
  }

  f.model[0] = 0;

  if (f.hard_clauses.size() > 1) find_initial_model(f);

  size_t avail, tot;
  cudaMemGetInfo(&avail, &tot);
  std::cout << "c Avail: " << avail << " Tot: " << tot << std::endl;

  if (nThreads == -1) {
    nThreads = std::min(((f.nVars + 31) / 32) * 32, 1024u);
  }

  std::cout << "c Using " << nThreads << " threads" << std::endl;

  alloc_formuala_d(f, f_d, nThreads);

  curandState_t *d_state;
  cudaMalloc(&d_state, nBlocks * nThreads * sizeof(curandState_t));

  cudaStream_t s;
  cudaStreamCreate(&s);

  start_kernel<<<nBlocks, nThreads, 0, s>>>(d_state);
  search_alg<<<nBlocks, nThreads, 0, s>>>(
      f_d, f.sum_soft_weights, alpha, beta, sp, hard_prob, hard_inc, opt_iters,
      restart_thresh, restart_flip, d_state);
  gpuErrchk(cudaPeekAtLastError());

  cudaMemcpyAsync(f.model, f_d.best_model, (f.nVars + 1) * sizeof(int),
                  cudaMemcpyDeviceToHost, s);
  cudaStreamSynchronize(s);
  cudaStreamDestroy(s);

  print_model(std::cout, f);

  free_formula_d(f_d);
  free_formula(f);
  cudaFree(d_state);

  return 0;
}
