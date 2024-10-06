#include "droid_kernels_cpu.h"
#include "cuda_utilities.h"

std::vector<torch::Tensor> projective_transform_cpu(
  torch::Tensor poses,
  torch::Tensor disps,
  torch::Tensor intrinsics,
  torch::Tensor ii,
  torch::Tensor jj)
{assert_not_implemented();}

torch::Tensor depth_filter_cpu(
    torch::Tensor poses,
    torch::Tensor disps,
    torch::Tensor intrinsics,
    torch::Tensor ix,
    torch::Tensor thresh)
{assert_not_implemented();}

torch::Tensor frame_distance_cpu(
  torch::Tensor poses,
  torch::Tensor disps,
  torch::Tensor intrinsics,
  torch::Tensor ii,
  torch::Tensor jj,
  const float beta)
{assert_not_implemented();}

std::vector<torch::Tensor> projmap_cpu(
  torch::Tensor poses,
  torch::Tensor disps,
  torch::Tensor intrinsics,
  torch::Tensor ii,
  torch::Tensor jj)
{assert_not_implemented();}

torch::Tensor iproj_cpu(
  torch::Tensor poses,
  torch::Tensor disps,
  torch::Tensor intrinsics)
{assert_not_implemented();}

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
  const bool motion_only)
{assert_not_implemented();}
