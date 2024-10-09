#include "droid_kernels_cpu.h"
#include "debug_utilities.h"

void projective_transform_kernel(
    const torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> target,
    const torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> weight,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> poses,
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> disps,
    const torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> intrinsics,
    const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> ii,
    const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> jj,
    torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> Hs,
    torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> vs,
    torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> Eii,
    torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> Eij,
    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> Cii,
    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> bz)
{
  int ht=disps.size(1);
  int wd=poses.size(2);

  //For each ii/jj frame pair/graph edge (i.e. block_id)
  int num_blocks=ii.size(0);
  for(int block_id=0;block_id<num_blocks;block_id++)
  {
    int ix = static_cast<int>(ii[block_id]);
    int jx = static_cast<int>(jj[block_id]);

    float fx = intrinsics[0];
    float fy = intrinsics[1];
    float cx = intrinsics[2];
    float cy = intrinsics[3];

    float ti[3], tj[3], tij[3];
    float qi[4], qj[4], qij[4];

    if (ix == jx) {
      tij[0] =  -0.1;
      tij[1] =     0;
      tij[2] =     0;
      qij[0] =     0;
      qij[1] =     0;
      qij[2] =     0;
      qij[3] =     1;
    }

    else {

      // load poses from global memory
      memcpy(ti,poses[ix],3*sizeof(float));
      memcpy(tj,poses[jx],3*sizeof(float));
      memcpy(qi,poses[ix]+3,4*sizeof(float));
      memcpy(qj,poses[jx]+3,4*sizeof(float));

      relSE3(ti, qi, tj, qj, tij, qij);
    }

    int l;//used as a counter in a few places;

    for(int i=0;i<h1;i++)
      for(int j=0;j<w1;j++)
      {
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
        memset(hij,0,12*(12+1)/2*sizeof(float));

        float vi[6], vj[6];
        memset(vi,0,6*sizeof(float));
        memset(vj,0,6*sizeof(float));

        const float u = static_cast<float>(j);
        const float v = static_cast<float>(i);

        // homogenous coordinates
        Xi[0] = (u - cx) / fx;
        Xi[1] = (v - cy) / fy;
        Xi[2] = 1;
        Xi[3] = disps[ix][i][j];
      }

    for (int n=0; n<6; n++)
    {
      vs[0][block_id][n]=0;
      for(int i=0;i<h1;i++)
        for(int j=0;j<w1;j++)



        // transform homogenous point
        actSE3(tij, qij, Xi, Xj);

        const float x = Xj[0];
        const float y = Xj[1];
        const float h = Xj[3];

        const float d = (Xj[2] < MIN_DEPTH) ? 0.0 : 1.0 / Xj[2];
        const float d2 = d * d;

        float wu = (Xj[2] < MIN_DEPTH) ? 0.0 : .001 * weight[block_id][0][i][j];
        float wv = (Xj[2] < MIN_DEPTH) ? 0.0 : .001 * weight[block_id][1][i][j];
        const float ru = target[block_id][0][i][j] - (fx * d * x + cx);
        const float rv = target[block_id][1][i][j] - (fy * d * y + cy);

}

std::vector<torch::Tensor> projective_transform_cpu(
  torch::Tensor poses,
  torch::Tensor disps,
  torch::Tensor intrinsics,
  torch::Tensor ii,
  torch::Tensor jj)
{

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
