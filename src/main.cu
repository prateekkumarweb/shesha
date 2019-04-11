#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <iomanip>
#include <signal.h>
#include <getopt.h>
#include <curand.h>

#include "maxsat.h"
#include "search_algs.h"

const struct option long_options[] = {
  {"help", no_argument, 0, 'h'},
  {"alg", required_argument, 0, 'a'},
  {0, 0, 0, 0}
};

const std::string option_desc[] {
  "Display help",
  "Algorithm 1/2"
  ""
};

void display_help(std::string exec) {
  std::cout << "Toolname version 0.1" << std::endl;
  std::cout << "\n"
            << "Description ..."
            << std::endl;
  
  std::cout << "\nUsage:\n";
  std::cout << " " << exec
            << "[Options]"
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
  if (sig == SIGINT) {
    std::cout << "Exiting ..." << std::endl;
    cudaStream_t s;
    cudaStreamCreate(&s);
    cudaMemcpyAsync(f.model, f_d.best_model, (f.nVars+1) * sizeof(int), cudaMemcpyDeviceToHost, s);
    cudaStreamDestroy(s);

    print_model(std::cout, f);
    exit(1);
  }
}

__global__ void start_kernel(curandState_t *state) {
  int idx = threadIdx.x+blockDim.x*blockIdx.x;
  curand_init(1234, idx, 0, &state[idx]);
}

int main(int argc, char **argv) {

  int opt_alg = 1;
  float alpha = 0.9, beta = 0.1, sp = 0.4;
  
  while (1) {
    int option_index = 0;
    int c = getopt_long(argc, argv, "ha:", long_options, &option_index);
    if (c == -1) break;

    switch (c) {
    case 0:
      break;

    case 'h':
      display_help(std::string(argv[0]));
      exit(0);

    case 'a':
      opt_alg = std::stoi(optarg);
      break;
    
    default:
      display_error("Unrecognised argument: " + std::to_string(c));
    }
  }

  if (optind < argc) {
    display_error("Unrecognised argument: " + std::string(argv[optind]));
  }

  signal(SIGINT, sigint_handler);

  if (!parse_formula(std::cin, f)) {
    display_error("Invalid input!");
  }

  size_t avail, tot;
  cudaMemGetInfo(&avail, &tot);
  std::cout << "Avail: " << avail << " Tot: " << tot << std::endl;

  alloc_formuala_d(f, f_d);

  curandState_t *d_state;
  cudaMalloc(&d_state, sizeof(curandState_t));

  cudaStream_t s;
  cudaStreamCreate(&s);

  search_alg1<<<1 + nClauses / 200, 256, 0, s>>>(f_d, alpha, beta, sp, d_state);

  cudaMemcpyAsync(f.model, f_d.best_model, (f.nVars+1) * sizeof(int), cudaMemcpyDeviceToHost, s);
  cudaStreamSynchronize(s);
  cudaStreamDestroy(s);

  print_model(std::cout, f);

  free_formula_d(f_d);
  cudaFree(d_state);

  return 0;
}