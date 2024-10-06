#pragma once

#include <torch/extension.h>

// CUDA forward declarations
std::vector<torch::Tensor> projective_transform_cpu(
  torch::Tensor poses,
  torch::Tensor disps,
  torch::Tensor intrinsics,
  torch::Tensor ii,
  torch::Tensor jj);

torch::Tensor depth_filter_cpu(
    torch::Tensor poses,
    torch::Tensor disps,
    torch::Tensor intrinsics,
    torch::Tensor ix,
    torch::Tensor thresh);

torch::Tensor frame_distance_cpu(
  torch::Tensor poses,
  torch::Tensor disps,
  torch::Tensor intrinsics,
  torch::Tensor ii,
  torch::Tensor jj,
  const float beta);

std::vector<torch::Tensor> projmap_cpu(
  torch::Tensor poses,
  torch::Tensor disps,
  torch::Tensor intrinsics,
  torch::Tensor ii,
  torch::Tensor jj);

torch::Tensor iproj_cpu(
  torch::Tensor poses,
  torch::Tensor disps,
  torch::Tensor intrinsics);

std::vector<torch::Tensor> ba_cpu(
    torch::Tensor poses,
    torch::Tensor disps,
    torch::Tensor intrinsics,
    torch::Tensor disps_sens,
    torch::Tensor targets,
    torch::Tensor weights,
    torch::Tensor eta,
    torch::Tensor ii,
    torch::Tensor jj,
    const int t0,
    const int t1,
    const int iterations,
    const float lm,
    const float ep,
    const bool motion_only);
