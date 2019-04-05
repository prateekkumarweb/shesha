#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <signal.h>
#include "search_algs.h"

int nVars;
int *d_best_model;
int *best_model;

void sigint_handler(int sig) {
  if (sig == SIGINT) {
    std::cout << "Exiting ..." << std::endl;
    //exit(1);
    cudaStream_t s;
    cudaStreamCreate(&s);
    cudaMemcpyAsync(best_model, d_best_model, nVars * sizeof(int),
             cudaMemcpyDeviceToHost, s);
    cudaStreamDestroy(s);
    for (int i = 0; i < nVars; i++) {
      std::cout << (best_model[i] == 1 ? (i + 1) : -(i + 1)) << " ";
    }
    std::cout << std::endl;
    exit(1);
  }
}

bool parse_input(int &nVars, int &nClauses, int &hardWeight,
                 std::vector<int> &input_vec, std::vector<int> &clauses,
                 std::vector<int> &weights) {

  std::string line;
  bool first_line = true;

  clauses.push_back(0);
  while (std::getline(std::cin, line)) {
    if (line == "")
      break;

    if (line.length() == 0)
      continue;
    if (line[0] == 'c')
      continue;

    std::stringstream ss(line);

    if (first_line) {
      std::string s;
      ss >> s;
      if (s != "p") {
        return false;
      }
      ss >> s;
      if (s != "wcnf") {
        return false;
      }
      ss >> nVars;
      ss >> nClauses;
      ss >> hardWeight;

      first_line = false;
    } else {
      int lit;
      ss >> lit;
      weights.push_back(lit);
      ss >> lit;
      while (lit != 0) {
        input_vec.push_back(lit);
        ss >> lit;
      }
      clauses.push_back(input_vec.size());
    }
  }
  return true;
}

int main() {
  signal(SIGINT, sigint_handler);
  int nClauses, hardWeight;
  std::vector<int> input_vec, clauses_vec, weights_vec;

  if (!parse_input(nVars, nClauses, hardWeight, input_vec, clauses_vec,
                   weights_vec)) {
    std::cout << "Input error!" << std::endl;
    return -1;
  }

  int *input_arr = input_vec.data();
  int *clauses = clauses_vec.data();
  int *weights = weights_vec.data();
  int *model = new int[nVars];
  best_model = new int[nVars];
  float *scores = new float[nVars];
  float alpha = 0.5, beta = 0.5;

  for (int i = 0; i < nVars; i++) {
    model[i] = best_model[i] = 1;
    scores[i] = 0.0;
  }

  int *d_input_arr;
  int *d_clauses;
  int *d_weights;
  int *d_model;
  float *d_scores;
  int *d_sat;

  size_t avail, tot;
  cudaMemGetInfo(&avail, &tot);
  std::cout << "Avail: " << avail << " Tot: " << tot << std::endl;

  cudaMalloc(&d_input_arr, input_vec.size() * sizeof(int));
  cudaMalloc(&d_clauses, clauses_vec.size() * sizeof(int));
  cudaMalloc(&d_weights, weights_vec.size() * sizeof(int));
  cudaMalloc(&d_model, nVars * sizeof(int));
  cudaMalloc(&d_best_model, nVars * sizeof(int));
  cudaMalloc(&d_scores, nVars * sizeof(float));
  cudaMalloc(&d_sat, nClauses * sizeof(int));

  cudaMemcpy(d_input_arr, input_arr, input_vec.size() * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_clauses, clauses, clauses_vec.size() * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_weights, weights, weights_vec.size() * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_model, model, nVars * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_best_model, best_model, nVars * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_scores, scores, nVars * sizeof(float), cudaMemcpyHostToDevice);

  cudaStream_t s;
  cudaStreamCreate(&s);
  search_alg1<<<1 + nClauses / 200, 256, 0, s>>>(
      nVars, nClauses, hardWeight, d_input_arr, d_clauses, d_weights, d_model,
      d_best_model, d_scores, d_sat, alpha, beta);

  cudaMemcpyAsync(best_model, d_best_model, nVars * sizeof(int),
             cudaMemcpyDeviceToHost, s);
  cudaStreamSynchronize(s);
  cudaStreamDestroy(s);
  for (int i = 0; i < nVars; i++) {
    std::cout << (best_model[i] == 1 ? (i + 1) : -(i + 1)) << " ";
  }
  std::cout << std::endl;

  delete[] model;
  delete[] best_model;
  delete[] scores;
  cudaFree(d_input_arr);
  cudaFree(d_clauses);
  cudaFree(d_weights);
  cudaFree(d_model);
  cudaFree(d_best_model);
  cudaFree(d_scores);
  cudaFree(d_sat);

  return 0;
}