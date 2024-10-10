#pragma once

#include <torch/extension.h>

void projective_transform_cuda(
  const torch::Tensor& targets,
  const torch::Tensor& weights,
  const torch::Tensor& poses,
  const torch::Tensor& disps,
  const torch::Tensor& intrinsics,
  const torch::Tensor& ii,
  const torch::Tensor& jj,
  torch::Tensor& Hs,
  torch::Tensor& vs,
  torch::Tensor& Eii,
  torch::Tensor& Eij,
  torch::Tensor& Cii,
  torch::Tensor& wi);

std::vector<torch::Tensor> projmap_cuda(
  torch::Tensor poses,
  torch::Tensor disps,
  torch::Tensor intrinsics,
  torch::Tensor ii,
  torch::Tensor jj);

torch::Tensor frame_distance_cuda(
  torch::Tensor poses,
  torch::Tensor disps,
  torch::Tensor intrinsics,
  torch::Tensor ii,
  torch::Tensor jj,
  const float beta);

torch::Tensor depth_filter_cuda(
    torch::Tensor poses,
    torch::Tensor disps,
    torch::Tensor intrinsics,
    torch::Tensor ix,
    torch::Tensor thresh);

torch::Tensor iproj_cuda(
  torch::Tensor poses,
  torch::Tensor disps,
  torch::Tensor intrinsics);

torch::Tensor accum_cuda(
  const torch::Tensor data,
  const torch::Tensor ix,
  const torch::Tensor jx);

void pose_retr_cuda(
    torch::Tensor poses,
    const torch::Tensor dx,
    const int t0,
    const int t1);

void disp_retr_cuda(
    torch::Tensor disps,
    const torch::Tensor dz,
    const torch::Tensor inds);

void EEt6x6_cuda(
    const torch::Tensor E,
    const torch::Tensor Q,
    const torch::Tensor idx,
    torch::Tensor S);

void Ev6x1_cuda(
    const torch::Tensor E,
    const torch::Tensor Q,
    const torch::Tensor w,
    const torch::Tensor idx,
    torch::Tensor v);

void EvT6x1_cuda(
  const torch::Tensor E,
  const torch::Tensor x,
  const torch::Tensor idx,
  torch::Tensor w);
