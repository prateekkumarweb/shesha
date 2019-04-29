#include <curand.h>
#include <getopt.h>
#include <signal.h>
#include <cassert>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

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
    {"alg", required_argument, 0, 'a'},
    {"alpha", required_argument, 0, 0},
    {"beta", required_argument, 0, 0},
    {"sp", required_argument, 0, 0},
    {"hard_prob", required_argument, 0, 0},
    {"iters", required_argument, 0, 'n'},
    {"restart_thresh", required_argument, 0, 0},
    {"nblocks", required_argument, 0, 0},
    {"nthreads", required_argument, 0, 0},
    {0, 0, 0, 0}};

const std::string option_desc[]{
    "Display help",
    "Algorithm 1/2",
    "Alpha for Alg1",
    "Beta for Alg1",
    "smoothening probability",
    "probability of choosing hard clause for score in Alg2",
    "Number of iterations",
    "Number of iterations without improvement between two restarts",
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

__global__ void start_kernel(curandState_t *state) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  curand_init(9876543210l, idx, 0, &state[idx]);
}

int main(int argc, char **argv) {
  int opt_alg = 1, opt_iters = 1000;
  float alpha = 0.9, beta = 0.1, sp = 0.1, hard_prob = 0.5,
        flip_hard_prob = 0.5;
  int restart_thresh = 100;
  int nBlocks = 1;
  int nThreads = 1024;

  while (1) {
    int option_index = 0;
    int c = getopt_long(argc, argv, "ha:n:", long_options, &option_index);
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
        } else if (std::string(long_options[option_index].name) ==
                   "flip_hard_prob") {
          if (optarg) {
            flip_hard_prob = std::stof(optarg);
          }
        } else if (std::string(long_options[option_index].name) ==
                   "restart_thresh") {
          if (optarg) {
            restart_thresh = std::stoi(optarg);
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

      case 'a':
        opt_alg = std::stoi(optarg);
        break;

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
  std::cout << "c algorithm " << opt_alg << std::endl;

  signal(SIGINT, sigint_handler);
  signal(SIGTERM, sigint_handler);

  if (!parse_formula(std::cin, f)) {
    display_error("Invalid input!");
  }

  size_t avail, tot;
  cudaMemGetInfo(&avail, &tot);
  std::cout << "c Avail: " << avail << " Tot: " << tot << std::endl;

  alloc_formuala_d(f, f_d, nThreads);

  curandState_t *d_state;
  cudaMalloc(&d_state, nBlocks * nThreads * sizeof(curandState_t));

  cudaStream_t s;
  cudaStreamCreate(&s);

  start_kernel<<<nBlocks, nThreads, 0, s>>>(d_state);
  if (opt_alg == 1) {
    search_alg1<<<nBlocks, nThreads, 0, s>>>(
        f_d, f.sum_soft_weights, alpha, beta, sp, flip_hard_prob, opt_iters,
        restart_thresh, d_state);
    gpuErrchk(cudaPeekAtLastError());
  }
  // else if (opt_alg == 2)
  // search_alg2<<<nBlocks, nThreads, 0, s>>>(f_d, f.sum_soft_weights,
  // hard_prob, sp, opt_iters, restart_thresh, d_state);

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
