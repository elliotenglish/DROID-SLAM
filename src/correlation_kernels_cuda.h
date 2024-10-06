#pragma once

std::vector<torch::Tensor> corr_index_cuda_forward(
  torch::Tensor volume,
  torch::Tensor coords,
  int radius);

std::vector<torch::Tensor> corr_index_cuda_backward(
  torch::Tensor volume,
  torch::Tensor coords,
  torch::Tensor corr_grad,
  int radius);
