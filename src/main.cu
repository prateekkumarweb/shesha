// // for __syncthreads()
// #ifndef __CUDACC__
// #define __CUDACC__
// #endif

#include <curand.h>
#include <device_launch_parameters.h>
#include <getopt.h>
#include <signal.h>

#include <algorithm>
#include <cassert>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

// #include "core/Solver.h"
// #include "core/SolverTypes.h"
#include "basis_pms.h"
#include "maxsat.h"
#include "pms.h"

// using namespace Glucose;

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
    {"soft_max_delta", required_argument, 0, 0},
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
    "Delta to increase the maximum soft weight",
    "Number of iterations",
    "Number of iterations without improvement between two restarts",
    "Number of flips to make at restart",
    "Number of blocks to use",
    "Number of threads per block to use",
    ""};

void display_help(std::string exec) {
  std::cout << "Shesha version 0.1" << std::endl;
  std::cout << "\n"
            << "Shesha is an incomplete MaxSAT solver for GPU." << std::endl;

  std::cout << "\nUsage:\n";
  std::cout << " " << exec << " [Options] filename"
            << "\n\n";

  std::cout << "Options:\n";
  for (int i = 0; long_options[i].name != 0; i++) {
    if (long_options[i].val)
      std::cout << '-' << char(long_options[i].val) << ", ";
    std::cout << std::left << "--" << std::setw(16) << long_options[i].name
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

int nBlocks = 1;
int nThreads = -1;

void sigint_handler(int sig) {
  static int call = 0;
  if (call > 0) {
    exit(0);
  }
  if (sig == SIGINT || sig == SIGTERM) {
    call++;
    printf("c Exiting ...\n");
    cudaStream_t s;
    cudaStreamCreate(&s);
    cudaMemcpyAsync(f.best_model, f_d.best_model,
                    (2 * f.nVars + 1) * sizeof(unsigned) * nBlocks,
                    cudaMemcpyDeviceToHost, s);
    cudaStreamDestroy(s);

    print_model(f, nBlocks);
    exit(1);
  }
}

void print_deci_stats(MaxSATFormula &f, const std::vector<unsigned> &model) {
  weight_t hard_ss = 0;
  for (unsigned i = 0; i < f.hard_clauses.size() - 1; i++) {
    unsigned s = 0;
    for (unsigned j = f.hard_clauses[i]; j < f.hard_clauses[i + 1]; j++) {
      unsigned l = f.hard_clauses_lits[j];
      if ((model[l >> 1] ^ 1) == (l & 1)) {
        s = 1;
        break;
      }
    }
    hard_ss += s;
  }
  weight_t soft_ss = 0;
  for (unsigned i = 0; i < f.soft_clauses.size() - 1; i++) {
    unsigned s = 0;
    for (unsigned j = f.soft_clauses[i]; j < f.soft_clauses[i + 1]; j++) {
      unsigned l = f.soft_clauses_lits[j];
      if ((model[l >> 1] ^ 1) == (l & 1)) {
        s = 1;
        break;
      }
    }
    soft_ss += s * f.soft_weights[i];
  }
  printf("c Deci Hard: %ld Soft: %ld Cost: %ld\n", hard_ss, soft_ss,
         f.sum_soft_weights - soft_ss);
  if (hard_ss + 1 == f.hard_clauses.size()) {
    printf("o %ld\n", f.sum_soft_weights - soft_ss);
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
  int hard_inc = 100;
  int soft_max_delta = 100;
  std::string filename = "";
  int c = 0;
  int option_index = 0;

  while ((c = getopt_long(argc, argv, "hn:", long_options, &option_index)) !=
         -1) {
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
                   "soft_max_delta") {
          if (optarg) {
            soft_max_delta = std::stoi(optarg);
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
    filename += argv[optind];
    std::cout << "Input file: " << filename << std::endl;
  }

  if (optind + 1 < argc) {
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
  std::cout << "c soft_max_delta " << soft_max_delta << std::endl;
  std::cout << "c nBlocks " << nBlocks << std::endl;
  std::cout << "c nThreads " << nThreads << std::endl;

  signal(SIGINT, sigint_handler);
  signal(SIGTERM, sigint_handler);

  Satlike sl;
  std::vector<int> init_solution;

  if (filename == "") {
    display_error("Filename required");
    exit(1);
    if (!parse_formula(std::cin, f)) {
      display_error("Invalid input!");
    }
  } else {
    sl.build_instance(filename.c_str());
    std::ifstream fin(filename);
    if (!parse_formula(fin, f)) {
      display_error("Invalid input!");
    }
  }

  f.best_model = new unsigned[(2 * f.nVars + 1) * nBlocks];

  // sl.decimation_only(init_solution, argv[1]);
  // sl.print_best_solution();

  // if (f.hard_clauses.size() > 1) find_initial_model(f);

  std::vector<unsigned> model(f.nVars + 1, 1);
  model[0] = 0;
  for (unsigned i = 1; i <= f.nVars; i++) {
    model[i] = sl.best_soln[i];
  }
  print_deci_stats(f, model);
  sl.free_memory();

  // size_t avail, tot;
  // cudaMemGetInfo(&avail, &tot);
  // std::cout << "c Avail: " << avail << " Tot: " << tot << std::endl;

  if (nThreads == -1) {
    nThreads = std::min(((f.nVars + 31) / 32) * 32, 1024u);
  }

  printf("c Using %d threads\n", nThreads);

  alloc_formula_d(f, f_d, nThreads, nBlocks, model);

  curandState_t *d_state;
  cudaMalloc(&d_state, nBlocks * nThreads * sizeof(curandState_t));

  cudaStream_t s;
  cudaStreamCreate(&s);

  start_kernel<<<nBlocks, nThreads, 0, s>>>(d_state);
  gpuErrchk(cudaPeekAtLastError());
  search_alg<<<nBlocks, nThreads, 0, s>>>(
      f_d, soft_max_delta, alpha, beta, sp, hard_prob, hard_inc, opt_iters,
      restart_thresh, restart_flip, d_state);
  gpuErrchk(cudaPeekAtLastError());

  cudaMemcpyAsync(f.best_model, f_d.best_model,
                  (2 * f.nVars + 1) * sizeof(unsigned) * nBlocks,
                  cudaMemcpyDeviceToHost, s);
  cudaStreamSynchronize(s);
  cudaStreamDestroy(s);

  print_model(f, nBlocks);

  free_formula_d(f, f_d);
  cudaFree(d_state);

  return 0;
}
