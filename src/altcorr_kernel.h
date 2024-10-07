#pragma once

std::vector<torch::Tensor> altcorr_cuda_forward(
  torch::Tensor fmap1,
  torch::Tensor fmap2,
  torch::Tensor coords,
  int radius);

std::vector<torch::Tensor> altcorr_cuda_backward(
  torch::Tensor fmap1,
  torch::Tensor fmap2,
  torch::Tensor coords,
  torch::Tensor corr_grad,
  int radius);
