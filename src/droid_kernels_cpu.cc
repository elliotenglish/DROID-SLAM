#include "droid_kernels_cpu.h"
#include "debug_utilities.h"

void projective_transform_kernel(
    const torch::PackedTensorAccessor32<float,4> target,
    const torch::PackedTensorAccessor32<float,4> weight,
    const torch::PackedTensorAccessor32<float,2> poses,
    const torch::PackedTensorAccessor32<float,3> disps,
    const torch::PackedTensorAccessor32<float,1> intrinsics,
    const torch::PackedTensorAccessor32<long,1> ii,
    const torch::PackedTensorAccessor32<long,1> jj,
    torch::PackedTensorAccessor32<float,4> Hs,
    torch::PackedTensorAccessor32<float,3> vs,
    torch::PackedTensorAccessor32<float,3> Eii,
    torch::PackedTensorAccessor32<float,3> Eij,
    torch::PackedTensorAccessor32<float,2> Cii,
    torch::PackedTensorAccessor32<float,2> bz)
{
  // int ht=disps.size(1);
  // int wd=poses.size(2);

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
      memcpy(ti,poses[ix].data(),3*sizeof(float));
      memcpy(tj,poses[jx].data(),3*sizeof(float));
      memcpy(qi,poses[ix].data()+3,4*sizeof(float));
      memcpy(qj,poses[jx].data()+3,4*sizeof(float));

      relSE3(ti, qi, tj, qj, tij, qij);
    }

    int l;//used as a counter in a few places;

    int k=-1;
    for(int i=0;i<h1;i++)
      for(int j=0;j<w1;j++)
      {
        k++;

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

        // x - coordinate

        Jj[0] = fx * (h*d);
        Jj[1] = fx * 0;
        Jj[2] = fx * (-x*h*d2);
        Jj[3] = fx * (-x*y*d2);
        Jj[4] = fx * (1 + x*x*d2);
        Jj[5] = fx * (-y*d);

        Jz = fx * (tij[0] * d - tij[2] * (x * d2));
        Cii[block_id][k] = wu * Jz * Jz;
        bz[block_id][k] = wu * ru * Jz;

        if (ix == jx) wu = 0;

        adjSE3(tij, qij, Jj, Ji);
        for (int n=0; n<6; n++) Ji[n] *= -1;

        l=0;
        for (int n=0; n<12; n++) {
          for (int m=0; m<=n; m++) {
            hij[l] += wu * Jx[n] * Jx[m];
            l++;
          }
        }

        for (int n=0; n<6; n++) {
          vi[n] += wu * ru * Ji[n];
          vj[n] += wu * ru * Jj[n];

          Eii[block_id][n][k] = wu * Jz * Ji[n];
          Eij[block_id][n][k] = wu * Jz * Jj[n];
        }


        Jj[0] = fy * 0;
        Jj[1] = fy * (h*d);
        Jj[2] = fy * (-y*h*d2);
        Jj[3] = fy * (-1 - y*y*d2);
        Jj[4] = fy * (x*y*d2);
        Jj[5] = fy * (x*d);

        Jz = fy * (tij[1] * d - tij[2] * (y * d2));
        Cii[block_id][k] += wv * Jz * Jz;
        bz[block_id][k] += wv * rv * Jz;

        if (ix == jx) wv = 0;

        adjSE3(tij, qij, Jj, Ji);
        for (int n=0; n<6; n++) Ji[n] *= -1;

        l=0;
        for (int n=0; n<12; n++) {
          for (int m=0; m<=n; m++) {
            hij[l] += wv * Jx[n] * Jx[m];
            l++;
          }
        }

        for (int n=0; n<6; n++) {
          vi[n] += wv * rv * Ji[n];
          vj[n] += wv * rv * Jj[n];

          Eii[block_id][n][k] += wv * Jz * Ji[n];
          Eij[block_id][n][k] += wv * Jz * Jj[n];
        }

        //Reductions
        for (int n=0; n<6; n++) {
          vs[0][block_id][n]+=vi[n];
          vs[1][block_id][n]+=vj[n];
        }

        l=0;
        for (int n=0; n<12; n++)
          for (int m=0; m<=n; m++) {
            if (n<6 && m<6) {
              Hs[0][block_id][n][m] += hij[l];
              Hs[0][block_id][m][n] += hij[l];
            }
            else if (n >=6 && m<6) {
              Hs[1][block_id][m][n-6] += hij[l];
              Hs[2][block_id][n-6][m] += hij[l];
            }
            else {
              Hs[3][block_id][n-6][m-6] += hij[l];
              Hs[3][block_id][m-6][n-6] += hij[l];
            }
          }
      }
  }
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

void accum_kernel(
  const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> inps,
  const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> ptrs,
  const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> idxs,
  torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> outs)
{
  int count=ptrs.size(0)-1;
  int D = inps.size(2);
  for(int block_id=0;block_id<count;block_id++)
  {
    const int start = ptrs[block_id];
    const int end = ptrs[block_id+1];
    for(int k=0; k<D; k+=blockDim.x)
    {
      float x = 0;
      for (int i=start; i<end; i++) {
        x += inps[idxs[i]][k];
      }
      outs[block_id][k] = x;
    }
  }
}

std::vector<torch::Tensor> ba_cpu(
    torch::Tensor poses,
    torch::Tensor disps,
    torch::Tensor intrinsics,
    torch::Tensor disps_sense,
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
{
  const int num = ii.size(0);
  return ba_generic(
    poses,
    disps,
    intrinsics,
    disps_sense,
    targets,
    weights,
    eta,
    ii,
    jj,
    t0,
    t1,
    iterations,
    lm,
    ep,
    motion_only,
    poses.device(),
    [](torch::Tensor& targets,
       torch::Tensor& weights,
       torch::Tensor& poses,
       torch::Tensor& disps,
       torch::Tensor& intrinsics,
       torch::Tensor& ii,
       torch::Tensor& jj,
       torch::Tensor& Hs,
       torch::Tensor& vs,
       torch::Tensor& Eii,
       torch::Tensor& Eij,
       torch::Tensor& Cii,
       torch::Tensor& wi)
      {
        // const int num = ii.size(0);
        projective_transform_kernel(
          targets.packed_accessor32<float,4>(),
          weights.packed_accessor32<float,4>(),
          poses.packed_accessor32<float,2>(),
          disps.packed_accessor32<float,3>(),
          intrinsics.packed_accessor32<float,1>(),
          ii.packed_accessor32<long,1>(),
          jj.packed_accessor32<long,1>(),
          Hs.packed_accessor32<float,4>(),
          vs.packed_accessor32<float,3>(),
          Eii.packed_accessor32<float,3>(),
          Eij.packed_accessor32<float,3>(),
          Cii.packed_accessor32<float,2>(),
          wi.packed_accessor32<float,2>()
        );
      },
    [](torch::Tensor& poses,
       torch::Tensor& dx,
       int t0,
       int t1)
      {
        pose_retr_kernel(
          poses.packed_accessor32<float,2>(),
          dx.packed_accessor32<float,2>(), t0, t1);
      },
    [](torch::Tensor& E,
       torch::Tensor& dx,
       torch::Tensor& ix,
       torch::Tensor& dw)
      {
        EvT6x1_kernel(
          E.packed_accessor32<float,3>(),
          dx.packed_accessor32<float,2>(),
          ix.packed_accessor32<long,1>(),
          dw.packed_accessor32<float,2>());
      },
    [](torch::Tensor& disps,
       torch::Tensor& dz,
       torch::Tensor& kx)
      {
        disp_retr_kernel(
        disps.packed_accessor32<float,3>(),
        dz.packed_accessor32<float,2>(),
        kx.packed_accessor32<long,1>());
      },
    [](const torch::Tensor& A,
       const torch::Tensor& B,
       const torch::Tensor& C)
      {
        return accum_cuda(A,B,C);
      },
    [](torch::Tensor& E,
       torch::Tensor& Q,
       torch::Tensor& ix_dev,
       torch::Tensor& S)
      {
        EEt6x6_kernel(
          E.packed_accessor32<float,3>(),
          Q.packed_accessor32<float,2>(),
          ix_dev.packed_accessor32<long,2>(),
          S.packed_accessor32<float,3>());
      },
    [](torch::Tensor& E,
       torch::Tensor& Q,
       torch::Tensor& w,
       torch::Tensor& jx_dev,
       torch::Tensor& v)
      {
        Ev6x1_kernel(
          E.packed_accessor32<float,3>(),
          Q.packed_accessor32<float,2>(),
          w.packed_accessor32<float,2>(),
          jx_dev.packed_accessor32<long,2>(),
          v.packed_accessor32<float,2>());
      }
    );
}
