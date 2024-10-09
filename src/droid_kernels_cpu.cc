#include "droid_kernels_cpu.h"
#include "debug_utilities.h"

std::vector<torch::Tensor> projective_transform_cpu(
  torch::Tensor poses,
  torch::Tensor disps,
  torch::Tensor intrinsics,
  torch::Tensor ii,
  torch::Tensor jj)
{
  int ht=disps.size(1);
  int wd=poses.size(2);

  float fx = intrinsics[0];
  float fy = intrinsics[1];
  float cx = intrinsics[2];
  float cy = intrinsics[3];

  float ti[3], tj[3], tij[3];
  float qi[4], qj[4], qij[4];

  //points 
  float Xi[4];
  float Xj[4];

  // jacobians
  float Jx[12];
  float Jz;

  float* Ji = &Jx[0];
  float* Jj = &Jx[6];

  // hessians
  float hij[12*(12+1)/2];

  float vi[6], vj[6];

  int l;
  for (l=0; l<12*(12+1)/2; l++) {
    hij[l] = 0;
  }

  for (int n=0; n<6; n++) {
    vi[n] = 0;
    vj[n] = 0;
  }

  for(int i=0;i<h1;i++)
    for(int j=0;j<w1;j++)
    {
      float u=j;
      float v=i;

      Xi[0] = (u - cx) / fx;
      Xi[1] = (v - cy) / fy;
      Xi[2] = 1;
      Xi[3] = disps[ix][i][j];

      
    }
}

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
