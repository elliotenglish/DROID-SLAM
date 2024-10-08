#include <torch/extension.h>
#include <vector>

#include "droid_kernels_cpu.h"
#include "droid_kernels_cuda.h"
#include "correlation_kernels_cpu.h"
#include "correlation_kernels_cuda.h"
#include "altcorr_kernel.h"
#include "debug_utilities.h"


#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CONTIGUOUS(x)


std::vector<torch::Tensor> ba(
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
    const bool motion_only) {

  CHECK_INPUT(targets);
  CHECK_INPUT(weights);
  CHECK_INPUT(poses);
  CHECK_INPUT(disps);
  CHECK_INPUT(intrinsics);
  CHECK_INPUT(disps_sens);
  CHECK_INPUT(ii);
  CHECK_INPUT(jj);

  if (targets.device().type() == torch::DeviceType::CPU) {
    return ba_cpu(poses, disps, intrinsics, disps_sens, targets, weights,
                  eta, ii, jj, t0, t1, iterations, lm, ep, motion_only);
  }
#if PORTABLE_EXTENSION_CUDA_ENABLED
  if (targets.device().type() == torch::DeviceType::CUDA) {
    return ba_cuda(poses, disps, intrinsics, disps_sens, targets, weights,
                   eta, ii, jj, t0, t1, iterations, lm, ep, motion_only);
  }
#endif
  assert_not_implemented();
}

torch::Tensor frame_distance(
    torch::Tensor poses,
    torch::Tensor disps,
    torch::Tensor intrinsics,
    torch::Tensor ii,
    torch::Tensor jj,
    const float beta) {

  CHECK_INPUT(poses);
  CHECK_INPUT(disps);
  CHECK_INPUT(intrinsics);
  CHECK_INPUT(ii);
  CHECK_INPUT(jj);

  if (poses.device().type() == torch::DeviceType::CPU) {
    return frame_distance_cpu(poses, disps, intrinsics, ii, jj, beta);
  }
#if PORTABLE_EXTENSION_CUDA_ENABLED
  if (poses.device().type() == torch::DeviceType::CUDA) {
    return frame_distance_cuda(poses, disps, intrinsics, ii, jj, beta);
  }
#endif
  assert_not_implemented();
}


std::vector<torch::Tensor> projmap(
    torch::Tensor poses,
    torch::Tensor disps,
    torch::Tensor intrinsics,
    torch::Tensor ii,
    torch::Tensor jj) {

  CHECK_INPUT(poses);
  CHECK_INPUT(disps);
  CHECK_INPUT(intrinsics);
  CHECK_INPUT(ii);
  CHECK_INPUT(jj);

  if (poses.device().type() == torch::DeviceType::CPU) {
    return projmap_cpu(poses, disps, intrinsics, ii, jj);
  }
#if PORTABLE_EXTENSION_CUDA_ENABLED
  if (poses.device().type() == torch::DeviceType::CUDA) {
    return projmap_cuda(poses, disps, intrinsics, ii, jj);
  }
#endif
  assert_not_implemented();
}


torch::Tensor iproj(
    torch::Tensor poses,
    torch::Tensor disps,
    torch::Tensor intrinsics) {
  CHECK_INPUT(poses);
  CHECK_INPUT(disps);
  CHECK_INPUT(intrinsics);

  if (poses.device().type() == torch::DeviceType::CPU) {
    return iproj_cpu(poses, disps, intrinsics);
  }
#if PORTABLE_EXTENSION_CUDA_ENABLED
  if (poses.device().type() == torch::DeviceType::CUDA) {
    return iproj_cuda(poses, disps, intrinsics);
  }
#endif
  assert_not_implemented();
}


// c++ python binding
std::vector<torch::Tensor> corr_index_forward(
    torch::Tensor volume,
    torch::Tensor coords,
    int radius) {
  CHECK_INPUT(volume);
  CHECK_INPUT(coords);

  if (volume.device().type() == torch::DeviceType::CPU) {
    return corr_index_cpu_forward(volume, coords, radius);
  }
#if PORTABLE_EXTENSION_CUDA_ENABLED
  if (volume.device().type() == torch::DeviceType::CUDA) {
    return corr_index_cuda_forward(volume, coords, radius);
  }
#endif
  assert_not_implemented();
}

std::vector<torch::Tensor> corr_index_backward(
    torch::Tensor volume,
    torch::Tensor coords,
    torch::Tensor corr_grad,
    int radius) {
  CHECK_INPUT(volume);
  CHECK_INPUT(coords);
  CHECK_INPUT(corr_grad);

#if PORTABLE_EXTENSION_CUDA_ENABLED
  if (volume.device().type() == torch::DeviceType::CUDA) {
    auto volume_grad = corr_index_cuda_backward(volume, coords, corr_grad, radius);
    return {volume_grad};
  }
#endif
  assert_not_implemented();
}

std::vector<torch::Tensor> altcorr_forward(
    torch::Tensor fmap1,
    torch::Tensor fmap2,
    torch::Tensor coords,
    int radius) {
  CHECK_INPUT(fmap1);
  CHECK_INPUT(fmap2);
  CHECK_INPUT(coords);

#if PORTABLE_EXTENSION_CUDA_ENABLED
  if (fmap1.device().type() == torch::DeviceType::CUDA) {
    return altcorr_cuda_forward(fmap1, fmap2, coords, radius);
  }
#endif
  assert_not_implemented();
}

std::vector<torch::Tensor> altcorr_backward(
    torch::Tensor fmap1,
    torch::Tensor fmap2,
    torch::Tensor coords,
    torch::Tensor corr_grad,
    int radius) {
  CHECK_INPUT(fmap1);
  CHECK_INPUT(fmap2);
  CHECK_INPUT(coords);
  CHECK_INPUT(corr_grad);

#if PORTABLE_EXTENSION_CUDA_ENABLED
  if (fmap1.device().type() == torch::DeviceType::CUDA) {
    return altcorr_cuda_backward(fmap1, fmap2, coords, corr_grad, radius);
  }
#endif
  assert_not_implemented();
}

torch::Tensor depth_filter(
  torch::Tensor poses,
  torch::Tensor disps,
  torch::Tensor intrinsics,
  torch::Tensor ix,
  torch::Tensor thresh) {

  CHECK_INPUT(poses);
  CHECK_INPUT(disps);
  CHECK_INPUT(intrinsics);
  CHECK_INPUT(ix);
  CHECK_INPUT(thresh);

  if (poses.device().type() == torch::DeviceType::CPU) {
    return depth_filter_cpu(poses, disps, intrinsics, ix, thresh);
  }
#if PORTABLE_EXTENSION_CUDA_ENABLED
  if (poses.device().type() == torch::DeviceType::CUDA) {
    return depth_filter_cuda(poses, disps, intrinsics, ix, thresh);
  }
#endif
  assert_not_implemented();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // bundle adjustment kernels
  m.def("ba", &ba, "bundle adjustment");
  m.def("frame_distance", &frame_distance, "frame_distance");
  m.def("projmap", &projmap, "projmap");
  m.def("depth_filter", &depth_filter, "depth_filter");
  m.def("iproj", &iproj, "back projection");

  // correlation volume kernels
  m.def("altcorr_forward", &altcorr_forward, "ALTCORR forward");
  m.def("altcorr_backward", &altcorr_backward, "ALTCORR backward");
  m.def("corr_index_forward", &corr_index_forward, "INDEX forward");
  m.def("corr_index_backward", &corr_index_backward, "INDEX backward");
}