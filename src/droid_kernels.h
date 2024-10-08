#pragma once

#include <torch/extension.h>

#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Parallel.h>

#include <Eigen/Sparse>
#include <Eigen/SparseCore>
#include <Eigen/SparseCholesky>

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

template<typename F1,typename F2>
SparseBlock schur_block(torch::Tensor E,
                        torch::Tensor Q,
                        torch::Tensor w,
                        torch::Tensor ii,
                        torch::Tensor jj,
                        torch::Tensor kk,
                        const int t0,
                        const int t1,
                        const c10::Device& device,
                        const F1& EEt6x6_kernel,
                        const F2& Ev6x1_kernel)
{
  torch::Tensor ii_cpu = ii.to(torch::kCPU);
  torch::Tensor jj_cpu = jj.to(torch::kCPU);
  torch::Tensor kk_cpu = kk.to(torch::kCPU);

  const int P = t1 - t0;
  const long* ii_data = ii_cpu.data_ptr<long>();
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

  for (int i=0; i<P; i++) {
    for (int j=0; j<P; j++) {
      for (int k=0; k < graph[i].size(); k++) {
        for (int l=0; l < graph[j].size(); l++) {
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
    torch::TensorOptions().dtype(torch::kInt64)).to(device).view({-1, 3});

  torch::Tensor jx_dev = torch::stack({kk_cpu}, -1)
    .to(torch::kCUDA).to(torch::kInt64);

  torch::Tensor ii2_cpu = torch::from_blob(ii_list.data(), {long(ii_list.size())}, 
    torch::TensorOptions().dtype(torch::kInt64)).view({-1});

  torch::Tensor jj2_cpu = torch::from_blob(jj_list.data(), {long(jj_list.size())}, 
    torch::TensorOptions().dtype(torch::kInt64)).view({-1});

  torch::Tensor S = torch::zeros({ix_dev.size(0), 6, 6}, 
    torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

  torch::Tensor v = torch::zeros({jx_dev.size(0), 6},
    torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

  EEt6x6_kernel(E,Q,ix_dev,S);

  Ev6x1_kernel(E,Q,w,jx_dev,v);

  // schur block
  SparseBlock A(P, 6);
  A.update_lhs(S, ii2_cpu, jj2_cpu);
  A.update_rhs(v, jj_cpu - t0);

  return A;
}

template<typename F1,typename F2,typename F3,typename F4,typename F5,typename F6,typename F7>
std::vector<torch::Tensor> ba_generic(
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
    const bool motion_only,
    const c10::Device& device,
    const F1& projective_transform_kernel,
    const F2& pose_retr_kernel,
    const F3& EvT6x1_kernel,
    const F4& disp_retr_kernel,
    const F5& accum,
    const F6& EEt6x6_kernel,
    const F7& Ev6x1_kernel)
{
  auto opts = poses.options();
  const int num = ii.size(0);
  const int ht = disps.size(1);
  const int wd = disps.size(2);

  torch::Tensor ts = torch::arange(t0, t1).to(device);
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
    projective_transform_kernel(targets,weights,poses,disps,intrinsics,ii,jj,Hs,vs,Eii,Eij,Cii,wi);

    // pose x pose block
    SparseBlock A(t1 - t0, 6);

    A.update_lhs(Hs.reshape({-1, 6, 6}), 
        torch::cat({ii, ii, jj, jj}) - t0, 
        torch::cat({ii, jj, ii, jj}) - t0);

    A.update_rhs(vs.reshape({-1, 6}), 
        torch::cat({ii, jj}) - t0);

    if (motion_only) {
      dx = A.solve(device,lm, ep);

      // update poses
      pose_retr_kernel(poses,dx,t0,t1);
    }
    
    else {
      // add depth residual if there are depth sensor measurements
      const float alpha = 0.05;
      torch::Tensor m = (disps_sens.index({kx, "..."}) > 0).to(torch::TensorOptions().dtype(torch::kFloat32)).view({-1, ht*wd});
      torch::Tensor C = accum(Cii, ii, kx) + m * alpha + (1 - m) * eta.view({-1, ht*wd});
      torch::Tensor w = accum(wi, ii, kx) - m * alpha * (disps.index({kx, "..."}) - disps_sens.index({kx, "..."})).view({-1, ht*wd});
      torch::Tensor Q = 1.0 / C;

      torch::Tensor Ei = accum(Eii.view({num, 6*ht*wd}), ii, ts).view({t1-t0, 6, ht*wd});
      torch::Tensor E = torch::cat({Ei, Eij}, 0);

      SparseBlock S = schur_block(E, Q, w, ii_exp, jj_exp, kk_exp, t0, t1, device, EEt6x6_kernel, Ev6x1_kernel);
      dx = (A - S).solve(device,lm, ep);

      torch::Tensor ix = jj_exp - t0;
      torch::Tensor dw = torch::zeros({ix.size(0), ht*wd}, opts);

      EvT6x1_kernel(E,dx,ix,dw);

      dz = Q * (w - accum(dw, ii_exp, kx));

      // update poses
      pose_retr_kernel(poses,dx,t0,t1);

      // update disparity maps
      disp_retr_kernel(disps,dz,kx);
    }

  }

  return {dx, dz};
}
