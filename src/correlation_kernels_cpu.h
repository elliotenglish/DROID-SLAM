#pragma once

#include <torch/extension.h>

std::vector<torch::Tensor> corr_index_cpu_forward(
  torch::Tensor volume,
  torch::Tensor coords,
  int radius);
