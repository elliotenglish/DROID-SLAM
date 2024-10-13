#include <torch/extension.h>
#include <vector>

#include "droid_kernels_cpu.h"
#include "droid_kernels_cuda.h"
#include "correlation_kernels_cpu.h"
#include "correlation_kernels_cuda.h"
#include "altcorr_kernel.h"
#include "debug_utilities.h"
#include "droid_kernels.h"

#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CONTIGUOUS(x)

// std::vector<torch::Tensor> ba(
//     torch::Tensor poses,
//     torch::Tensor disps,
//     torch::Tensor intrinsics,
//     torch::Tensor disps_sens,
//     torch::Tensor targets,
//     torch::Tensor weights,
//     torch::Tensor eta,
//     torch::Tensor ii,
//     torch::Tensor jj,
//     const int t0,
//     const int t1,
//     const int iterations,
//     const float lm,
//     const float ep,
//     const bool motion_only) {

//   CHECK_INPUT(targets);
//   CHECK_INPUT(weights);
//   CHECK_INPUT(poses);
//   CHECK_INPUT(disps);
//   CHECK_INPUT(intrinsics);
//   CHECK_INPUT(disps_sens);
//   CHECK_INPUT(ii);
//   CHECK_INPUT(jj);

//   if (targets.device().type() == torch::DeviceType::CPU) {
//     return ba_cpu(poses, disps, intrinsics, disps_sens, targets, weights,
//                   eta, ii, jj, t0, t1, iterations, lm, ep, motion_only);
//   }
// #if PORTABLE_EXTENSION_CUDA_ENABLED
//   if (targets.device().type() == torch::DeviceType::CUDA) {
//     return ba_cuda(poses, disps, intrinsics, disps_sens, targets, weights,
//                    eta, ii, jj, t0, t1, iterations, lm, ep, motion_only);
//   }
// #endif
//   assert_not_implemented();
// }

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
  const torch::Tensor poses,
  const torch::Tensor disps,
  const torch::Tensor intrinsics,
  const torch::Tensor ix,
  const torch::Tensor thresh)
{
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

void projective_transform(
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
  torch::Tensor& wi)
{
  CHECK_INPUT(targets);
  CHECK_INPUT(weights);
  CHECK_INPUT(poses);
  CHECK_INPUT(disps);
  CHECK_INPUT(intrinsics);
  CHECK_INPUT(ii);
  CHECK_INPUT(jj);
  CHECK_INPUT(Hs);
  CHECK_INPUT(vs);
  CHECK_INPUT(Eii);
  CHECK_INPUT(Eij);
  CHECK_INPUT(Cii);
  CHECK_INPUT(wi);

if (poses.device().type() == torch::DeviceType::CPU) {
    projective_transform_cpu(targets,weights,poses,disps,intrinsics,ii,jj,Hs,vs,Eii,Eij,Cii,wi);
  }
#if PORTABLE_EXTENSION_CUDA_ENABLED
  if (poses.device().type() == torch::DeviceType::CUDA) {
    projective_transform_cuda(targets,weights,poses,disps,intrinsics,ii,jj,Hs,vs,Eii,Eij,Cii,wi);
  }
#endif
}

void pose_retr(
    torch::Tensor poses,
    const torch::Tensor dx,
    const int t0,
    const int t1)
{
if (poses.device().type() == torch::DeviceType::CPU) {
    pose_retr_cpu(poses,dx,t0,t1);
  }
#if PORTABLE_EXTENSION_CUDA_ENABLED
  if (poses.device().type() == torch::DeviceType::CUDA) {
    pose_retr_cuda(poses,dx,t0,t1);
  }
#endif
}

void disp_retr(
    torch::Tensor disps,
    const torch::Tensor dz,
    const torch::Tensor inds)
{
if (dz.device().type() == torch::DeviceType::CPU) {
    disp_retr_cpu(disps,dz,inds);
  }
#if PORTABLE_EXTENSION_CUDA_ENABLED
  if (dz.device().type() == torch::DeviceType::CUDA) {
    disp_retr_cuda(disps,dz,inds);
  }
#endif
}

torch::Tensor accum(
  const torch::Tensor& inps,
  const torch::Tensor& ptrs,
  const torch::Tensor& idxs)
{
  if (inps.device().type() == torch::DeviceType::CPU) {
    return accum_cpu(inps,ptrs,idxs);
  }
#if PORTABLE_EXTENSION_CUDA_ENABLED
  if (inps.device().type() == torch::DeviceType::CUDA) {
    return accum_cuda(inps,ptrs,idxs);
  }
#endif
  assert_not_implemented();
}

void EEt6x6(
    const torch::Tensor E,
    const torch::Tensor Q,
    const torch::Tensor idx,
    torch::Tensor S)
{
  if (E.device().type() == torch::DeviceType::CPU) {
    EEt6x6_cpu(E,Q,idx,S);
    return;
  }
#if PORTABLE_EXTENSION_CUDA_ENABLED
  if (E.device().type() == torch::DeviceType::CUDA) {
    EEt6x6_cuda(E,Q,idx,S);
    return;
  }
#endif
  assert_not_implemented();
}

void Ev6x1(
    const torch::Tensor E,
    const torch::Tensor Q,
    const torch::Tensor w,
    const torch::Tensor idx,
    torch::Tensor v)
{
  if (E.device().type() == torch::DeviceType::CPU) {
    Ev6x1_cpu(E,Q,w,idx,v);
    return;
  }
#if PORTABLE_EXTENSION_CUDA_ENABLED
  if (E.device().type() == torch::DeviceType::CUDA) {
    Ev6x1_cuda(E,Q,w,idx,v);
    return;
  }
#endif
  assert_not_implemented();
}

void EvT6x1(
  const torch::Tensor E,
  const torch::Tensor x,
  const torch::Tensor idx,
  torch::Tensor w)
{
  if (E.device().type() == torch::DeviceType::CPU) {
    EvT6x1_cpu(E,x,idx,w);
    return;
  }
#if PORTABLE_EXTENSION_CUDA_ENABLED
  if (E.device().type() == torch::DeviceType::CUDA) {
    EvT6x1_cuda(E,x,idx,w);
    return;
  }
#endif
  assert_not_implemented();
}

torch::Tensor accum2(
  const torch::Tensor data,
  const torch::Tensor ix,
  const torch::Tensor jx)
{
  torch::Tensor ix_cpu = ix.to(torch::kCPU);
  torch::Tensor jx_cpu = jx.to(torch::kCPU);
  torch::Tensor inds = torch::argsort(ix_cpu);

  long* ix_data = ix_cpu.data_ptr<long>();
  long* jx_data = jx_cpu.data_ptr<long>();
  long* kx_data = inds.data_ptr<long>();

  int count = jx.size(0);
  std::vector<int> cols;

  torch::Tensor ptrs_cpu = torch::zeros({count+1}, 
    torch::TensorOptions().dtype(torch::kInt64));
  
  long* ptrs_data = ptrs_cpu.data_ptr<long>();
  ptrs_data[0] = 0;

  int i = 0;
  for (int j=0; j<count; j++) {
    while (i < ix.size(0) && ix_data[kx_data[i]] <= jx_data[j]) {
      if (ix_data[kx_data[i]] == jx_data[j])
        cols.push_back(kx_data[i]);
      i++;
    }
    ptrs_data[j+1] = cols.size();
  }

  torch::Tensor idxs_cpu = torch::zeros({long(cols.size())}, 
    torch::TensorOptions().dtype(torch::kInt64));

  long* idxs_data = idxs_cpu.data_ptr<long>();

  for (unsigned int i=0; i<cols.size(); i++) {
    idxs_data[i] = cols[i];
  }

  torch::Tensor ptrs = ptrs_cpu.to(data.device().type());
  torch::Tensor idxs = idxs_cpu.to(data.device().type());

  return accum(data,ptrs,idxs);
}

///////////////////////////////////////////////////////////

typedef Eigen::SparseMatrix<double> SpMat;
typedef Eigen::Triplet<double> T;

class SparseBlock {
  public:

    Eigen::SparseMatrix<double> A;
    Eigen::VectorX<double> b;

    

    SparseBlock(int N, int M) : N(N), M(M) {
      A = Eigen::SparseMatrix<double>(N*M, N*M);
      b = Eigen::VectorXd::Zero(N*M);
    }

    SparseBlock(Eigen::SparseMatrix<double> const& A, Eigen::VectorX<double> const& b, 
        int N, int M) : A(A), b(b), N(N), M(M) {}

    void update_lhs(torch::Tensor As, torch::Tensor ii, torch::Tensor jj) {

      auto As_cpu = As.to(torch::kCPU).to(torch::kFloat64);
      auto ii_cpu = ii.to(torch::kCPU).to(torch::kInt64);
      auto jj_cpu = jj.to(torch::kCPU).to(torch::kInt64);

      auto As_acc = As_cpu.accessor<double,3>();
      auto ii_acc = ii_cpu.accessor<long,1>();
      auto jj_acc = jj_cpu.accessor<long,1>();

      std::vector<T> tripletList;
      for (int n=0; n<ii.size(0); n++) {
        const int i = ii_acc[n];
        const int j = jj_acc[n];

        if (i >= 0 && j >= 0) {
          for (int k=0; k<M; k++) {
            for (int l=0; l<M; l++) {
              double val = As_acc[n][k][l];
              tripletList.push_back(T(M*i + k, M*j + l, val));
            }
          }
        }
      }
      A.setFromTriplets(tripletList.begin(), tripletList.end());
    }

    void update_rhs(torch::Tensor bs, torch::Tensor ii) {
      auto bs_cpu = bs.to(torch::kCPU).to(torch::kFloat64);
      auto ii_cpu = ii.to(torch::kCPU).to(torch::kInt64);

      auto bs_acc = bs_cpu.accessor<double,2>();
      auto ii_acc = ii_cpu.accessor<long,1>();

      for (int n=0; n<ii.size(0); n++) {
        const int i = ii_acc[n];
        if (i >= 0) {
          for (int j=0; j<M; j++) {
            b(i*M + j) += bs_acc[n][j];
          }
        }
      }
    }

    SparseBlock operator-(const SparseBlock& S) {
      return SparseBlock(A - S.A, b - S.b, N, M);
    }

    std::tuple<torch::Tensor, torch::Tensor> get_dense(const c10::Device& device) {
      Eigen::MatrixXd Ad = Eigen::MatrixXd(A);

      torch::Tensor H = torch::from_blob(Ad.data(), {N*M, N*M}, torch::TensorOptions()
        .dtype(torch::kFloat64)).to(device).to(torch::kFloat32);

      torch::Tensor v = torch::from_blob(b.data(), {N*M, 1}, torch::TensorOptions()
        .dtype(torch::kFloat64)).to(device).to(torch::kFloat32);

      return std::make_tuple(H, v);

    }

    torch::Tensor solve(const c10::Device& device,const float lm=0.0001, const float ep=0.1) {
      torch::Tensor dx;

      Eigen::SparseMatrix<double> L(A);
      L.diagonal().array() += ep + lm * L.diagonal().array();

      Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> solver;
      solver.compute(L);

      if (solver.info() == Eigen::Success) {
        Eigen::VectorXd x = solver.solve(b);
        dx = torch::from_blob(x.data(), {N, M}, torch::TensorOptions()
          .dtype(torch::kFloat64)).to(device).to(torch::kFloat32);
      }
      else {
        dx = torch::zeros({N, M}, torch::TensorOptions()
          .device(device).dtype(torch::kFloat32));
      }
      
      return dx;
    }

  private:
    const int N;
    const int M;

};

SparseBlock schur_block(torch::Tensor E,
                        torch::Tensor Q,
                        torch::Tensor w,
                        torch::Tensor ii,
                        torch::Tensor jj,
                        torch::Tensor kk,
                        const int t0,
                        const int t1)
{
  torch::Tensor ii_cpu = ii.to(torch::kCPU);
  torch::Tensor jj_cpu = jj.to(torch::kCPU);
  torch::Tensor kk_cpu = kk.to(torch::kCPU);

  const int P = t1 - t0;
  //const long* ii_data = ii_cpu.data_ptr<long>();
  const long* jj_data = jj_cpu.data_ptr<long>();
  const long* kk_data = kk_cpu.data_ptr<long>();

  std::vector<std::vector<long>> graph(P);
  std::vector<std::vector<long>> index(P);

  for (int n=0; n<ii_cpu.size(0); n++) {
    const int j = jj_data[n];
    const int k = kk_data[n];

    if (j >= t0 && j <= t1) {
      const int t = j - t0;
      graph[t].push_back(k);
      index[t].push_back(n);
    }
  }

  std::vector<long> ii_list, jj_list, idx, jdx;

  typedef std::vector<long>::size_type TI;
  for (int i=0; i<P; i++) {
    for (int j=0; j<P; j++) {
      for (TI k=0; k < graph[i].size(); k++) {
        for (TI l=0; l < graph[j].size(); l++) {
          if (graph[i][k] == graph[j][l]) {
            ii_list.push_back(i);
            jj_list.push_back(j);

            idx.push_back(index[i][k]);
            idx.push_back(index[j][l]);
            idx.push_back(graph[i][k]);
          }
        }
      }
    }
  }

  torch::Tensor ix_dev = torch::from_blob(idx.data(), {long(idx.size())}, 
    torch::TensorOptions().dtype(torch::kInt64)).to(E.device().type()).view({-1, 3});

  torch::Tensor jx_dev = torch::stack({kk_cpu}, -1)
    .to(E.device()).to(torch::kInt64);

  torch::Tensor ii2_cpu = torch::from_blob(ii_list.data(), {long(ii_list.size())}, 
    torch::TensorOptions().dtype(torch::kInt64)).view({-1});

  torch::Tensor jj2_cpu = torch::from_blob(jj_list.data(), {long(jj_list.size())}, 
    torch::TensorOptions().dtype(torch::kInt64)).view({-1});

  torch::Tensor S = torch::zeros({ix_dev.size(0), 6, 6}, 
    torch::TensorOptions().dtype(torch::kFloat32).device(E.device().type()));

  torch::Tensor v = torch::zeros({jx_dev.size(0), 6},
    torch::TensorOptions().dtype(torch::kFloat32).device(E.device().type()));

  EEt6x6(E,Q,ix_dev,S);

  Ev6x1(E,Q,w,jx_dev,v);

  // schur block
  SparseBlock A(P, 6);
  A.update_lhs(S, ii2_cpu, jj2_cpu);
  A.update_rhs(v, jj_cpu - t0);

  return A;
}

///////////////////////////////////////////////////////////

//42 is growing over time
std::vector<torch::Tensor> ba(
    torch::Tensor poses, //512x7
    torch::Tensor disps, //512xHxW
    torch::Tensor intrinsics, //4
    torch::Tensor disps_sens, //512xHxW
    torch::Tensor targets, //42x2xHxW
    torch::Tensor weights, //42x2xHxW
    torch::Tensor eta, //9xHxW
    torch::Tensor ii, //42
    torch::Tensor jj, //42
    const int t0,
    const int t1,
    const int iterations,
    const float lm,
    const float ep,
    const bool motion_only)
{
  CHECK_INPUT(targets);
  CHECK_INPUT(weights);
  CHECK_INPUT(poses);
  CHECK_INPUT(disps);
  CHECK_INPUT(intrinsics);
  CHECK_INPUT(disps_sens);
  CHECK_INPUT(targets);
  CHECK_INPUT(weights);
  CHECK_INPUT(eta);
  CHECK_INPUT(ii);
  CHECK_INPUT(jj);

  auto opts = poses.options();
  const int num = ii.size(0);
  const int ht = disps.size(1);
  const int wd = disps.size(2);

  torch::Tensor ts = torch::arange(t0, t1).to(poses.device().type());
  torch::Tensor ii_exp = torch::cat({ts, ii}, 0);
  torch::Tensor jj_exp = torch::cat({ts, jj}, 0);

  std::tuple<torch::Tensor, torch::Tensor> kuniq = 
    torch::_unique(ii_exp, true, true);

  torch::Tensor kx = std::get<0>(kuniq);
  torch::Tensor kk_exp = std::get<1>(kuniq);
    
  torch::Tensor dx;
  torch::Tensor dz;

  // initialize buffers
  torch::Tensor Hs = torch::zeros({4, num, 6, 6}, opts);
  torch::Tensor vs = torch::zeros({2, num, 6}, opts);
  torch::Tensor Eii = torch::zeros({num, 6, ht*wd}, opts);
  torch::Tensor Eij = torch::zeros({num, 6, ht*wd}, opts);
  torch::Tensor Cii = torch::zeros({num, ht*wd}, opts);
  torch::Tensor wi = torch::zeros({num, ht*wd}, opts);

  for (int itr=0; itr<iterations; itr++) {
    // projective_transform_kernel(targets,weights,poses,disps,intrinsics,ii,jj,Hs,vs,Eii,Eij,Cii,wi);
    projective_transform(targets,weights,poses,disps,intrinsics,ii,jj,Hs,vs,Eii,Eij,Cii,wi);

    // pose x pose block
    SparseBlock A(t1 - t0, 6);

    A.update_lhs(Hs.reshape({-1, 6, 6}), 
        torch::cat({ii, ii, jj, jj}) - t0, 
        torch::cat({ii, jj, ii, jj}) - t0);

    A.update_rhs(vs.reshape({-1, 6}), 
        torch::cat({ii, jj}) - t0);

    if (motion_only) {
      dx = A.solve(poses.device().type(),lm, ep);

      // update poses
      // pose_retr_kernel(poses,dx,t0,t1);
      pose_retr(poses,dx,t0,t1);
    }
    
    else {
      // add depth residual if there are depth sensor measurements
      const float alpha = 0.05;
      torch::Tensor m = (disps_sens.index({kx, "..."}) > 0).to(torch::TensorOptions().dtype(torch::kFloat32)).view({-1, ht*wd});
      torch::Tensor C = accum2(Cii, ii, kx) + m * alpha + (1 - m) * eta.view({-1, ht*wd});
      torch::Tensor w = accum2(wi, ii, kx) - m * alpha * (disps.index({kx, "..."}) - disps_sens.index({kx, "..."})).view({-1, ht*wd});
      torch::Tensor Q = 1.0 / C;

      torch::Tensor Ei = accum2(Eii.view({num, 6*ht*wd}), ii, ts).view({t1-t0, 6, ht*wd});
      torch::Tensor E = torch::cat({Ei, Eij}, 0);

      SparseBlock S = schur_block(E, Q, w, ii_exp, jj_exp, kk_exp, t0, t1);
      dx = (A - S).solve(poses.device().type(),lm, ep);

      torch::Tensor ix = jj_exp - t0;
      torch::Tensor dw = torch::zeros({ix.size(0), ht*wd}, opts);

      // EvT6x1_kernel(E,dx,ix,dw);
      EvT6x1(E,dx,ix,dw);

      dz = Q * (w - accum2(dw, ii_exp, kx));

      // update poses
      // pose_retr_kernel(poses,dx,t0,t1);
      pose_retr(poses,dx,t0,t1);

      // update disparity maps
      //disp_retr_kernel(disps,dz,kx);
      disp_retr(disps,dz,kx);
    }
  }

  return {dx, dz};
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