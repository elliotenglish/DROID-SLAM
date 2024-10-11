#include "droid_kernels.h"
#include "droid_kernels_cpu.h"
#include "debug_utilities.h"
#include "lie_groups.h"

void load_intrinsics(
  const torch::PackedTensorAccessor32<float,1>& intrinsics,
  float& fx,float& fy,float& cx,float& cy)
{
  fx = intrinsics[0];
  fy = intrinsics[1];
  cx = intrinsics[2];
  cy = intrinsics[3];
}

void load_pose(
  const torch::PackedTensorAccessor32<float,2>& poses,
  const int ix,
  float ti[],
  float qi[])
{
  memcpy(ti,poses[ix].data(),3*sizeof(float));
  memcpy(qi,poses[ix].data()+3,4*sizeof(float));
}

void store_pose(
  const torch::PackedTensorAccessor32<float,2>& poses,
  const int ix,
  float ti[],
  float qi[])
{
  memcpy(poses[ix].data(),ti,3*sizeof(float));
  memcpy(poses[ix].data()+3,qi,4*sizeof(float));
}

void load_relative_pose(
  const torch::PackedTensorAccessor32<float,2>& poses,
  const int ix,
  const int jx,
  float ti[],float tj[],float tij[],
  float qi[],float qj[],float qij[])
{
  if (ix == jx) {
    tij[0] =  -0.1;
    tij[1] =     0;
    tij[2] =     0;
    qij[0] =     0;
    qij[1] =     0;
    qij[2] =     0;
    qij[3] =     1;
  }
  else
  {
    // load poses from global memory
    memcpy(ti,poses[ix].data(),3*sizeof(float));
    memcpy(tj,poses[jx].data(),3*sizeof(float));
    memcpy(qi,poses[ix].data()+3,4*sizeof(float));
    memcpy(qj,poses[jx].data()+3,4*sizeof(float));

    relSE3(ti, qi, tj, qj, tij, qij);
  }
}

///////////////////////////////////////////////////////////

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
  int ht=disps.size(1);
  int wd=poses.size(2);

  //For each ii/jj frame pair/graph edge (i.e. block_id)
  int num_blocks=ii.size(0);
  for(int block_id=0;block_id<num_blocks;block_id++)
  {
    int ix = static_cast<int>(ii[block_id]);
    int jx = static_cast<int>(jj[block_id]);

    float fx,fy,cx,cy;
    load_intrinsics(intrinsics,fx,fy,cx,cy);
    float ti[3], tj[3], tij[3];
    float qi[4], qj[4], qij[4];
    load_relative_pose(poses,ix,jx,ti,tj,tij,qi,qj,qij);

    int l;//used as a counter in a few places;
    int k=-1;
    for(int i=0;i<ht;i++)
      for(int j=0;j<wd;j++)
      {
        k++;

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

        //points 
        float Xi[4];
        float Xj[4];

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

void projective_transform_cpu(
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
    wi.packed_accessor32<float,2>());
}

///////////////////////////////////////////////////////////

void projmap_kernel(
  const torch::PackedTensorAccessor32<float,2> poses,
  const torch::PackedTensorAccessor32<float,3> disps,
  const torch::PackedTensorAccessor32<float,1> intrinsics,
  const torch::PackedTensorAccessor32<long,1> ii,
  const torch::PackedTensorAccessor32<long,1> jj,
  torch::PackedTensorAccessor32<float,4> coords,
  torch::PackedTensorAccessor32<float,4> valid)
{
  const int num = ii.size(0);
  const int ht = disps.size(1);
  const int wd = disps.size(2);

  for(int block_id=0;block_id<num;block_id++)
  {
    int ix = static_cast<int>(ii[block_id]);
    int jx = static_cast<int>(jj[block_id]);

    float fx,fy,cx,cy;
    load_intrinsics(intrinsics,fx,fy,cx,cy);
    float ti[3], tj[3], tij[3];
    float qi[4], qj[4], qij[4];
    load_relative_pose(poses,ix,jx,ti,tj,tij,qi,qj,qij);

    for(int i=0;i<ht;i++)
      for(int j=0;j<wd;j++)
      {
        const float u = static_cast<float>(j);
        const float v = static_cast<float>(i);

        //points 
        float Xi[4];
        float Xj[4];
    
        // homogenous coordinates
        Xi[0] = (u - cx) / fx;
        Xi[1] = (v - cy) / fy;
        Xi[2] = 1;
        Xi[3] = disps[ix][i][j];

        // transform homogenous point
        actSE3(tij, qij, Xi, Xj);

        coords[block_id][i][j][0] = u;
        coords[block_id][i][j][1] = v;

        if (Xj[2] > 0.01) {
          coords[block_id][i][j][0] = fx * (Xj[0] / Xj[2]) + cx;
          coords[block_id][i][j][1] = fy * (Xj[1] / Xj[2]) + cy;
        }

        valid[block_id][i][j][0] = (Xj[2] > MIN_DEPTH) ? 1.0 : 0.0;
      }
  }
}

std::vector<torch::Tensor> projmap_cpu(
  const torch::Tensor& poses,
  const torch::Tensor& disps,
  const torch::Tensor& intrinsics,
  const torch::Tensor& ii,
  const torch::Tensor& jj)
{
  auto opts = poses.options();
  const int num = ii.size(0);
  const int ht = disps.size(1);
  const int wd = disps.size(2);

  torch::Tensor coords = torch::zeros({num, ht, wd, 3}, opts);
  torch::Tensor valid = torch::zeros({num, ht, wd, 1}, opts);

  projmap_kernel(
    poses.packed_accessor32<float,2>(),
    disps.packed_accessor32<float,3>(),
    intrinsics.packed_accessor32<float,1>(),
    ii.packed_accessor32<long,1>(),
    jj.packed_accessor32<long,1>(),
    coords.packed_accessor32<float,4>(),
    valid.packed_accessor32<float,4>());

  return {coords, valid};
}

///////////////////////////////////////////////////////////

void frame_distance_kernel(
  const torch::PackedTensorAccessor32<float,2> poses,
  const torch::PackedTensorAccessor32<float,3> disps,
  const torch::PackedTensorAccessor32<float,1> intrinsics,
  const torch::PackedTensorAccessor32<long,1> ii,
  const torch::PackedTensorAccessor32<long,1> jj,
  torch::PackedTensorAccessor32<float,1> dist,
  const float beta)
{
  const int num = ii.size(0);
  const int ht = disps.size(1);
  const int wd = disps.size(2);

  for(int block_id=0;block_id<num;block_id++)
  {
    int ix = static_cast<int>(ii[block_id]);
    int jx = static_cast<int>(jj[block_id]);

    float fx,fy,cx,cy;
    load_intrinsics(intrinsics,fx,fy,cx,cy);
    float ti[3], tj[3], tij[3];
    float qi[4], qj[4], qij[4];
    load_relative_pose(poses,ix,jx,ti,tj,tij,qi,qj,qij);

    float accum=0;
    float valid=0;
    float total=0;

    for(int i=0;i<ht;i++)
      for(int j=0;j<wd;j++)
      {
        const float u = static_cast<float>(j);
        const float v = static_cast<float>(i);

        //points 
        float Xi[4];
        float Xj[4];

        // homogenous coordinates
        Xi[0] = (u - cx) / fx;
        Xi[1] = (v - cy) / fy;
        Xi[2] = 1;
        Xi[3] = disps[ix][i][j];

        // transform homogenous point
        actSE3(tij, qij, Xi, Xj);

        float du = fx * (Xj[0] / Xj[2]) + cx - u;
        float dv = fy * (Xj[1] / Xj[2]) + cy - v;
        float d = sqrtf(du*du + dv*dv);

        total += beta;
        
        if (Xj[2] > MIN_DEPTH) {
          accum += beta * d;
          valid += beta;
        }

        Xi[0] = (u - cx) / fx;
        Xi[1] = (v - cy) / fy;
        Xi[2] = 1;
        Xi[3] = disps[ix][i][j];

        Xj[0] = Xi[0] + Xi[3] * tij[0];
        Xj[1] = Xi[1] + Xi[3] * tij[1];
        Xj[2] = Xi[2] + Xi[3] * tij[2];

        du = fx * (Xj[0] / Xj[2]) + cx - u;
        dv = fy * (Xj[1] / Xj[2]) + cy - v;
        d = sqrtf(du*du + dv*dv);

        total += (1 - beta);
        
        if (Xj[2] > MIN_DEPTH) {
          accum += (1 - beta) * d;
          valid += (1 - beta);
        }
      }

    dist[block_id] = (valid / (total + 1e-8) < 0.75) ? 1000.0 : accum / valid;
  }
}

torch::Tensor frame_distance_cpu(
  const torch::Tensor& poses,
  const torch::Tensor& disps,
  const torch::Tensor& intrinsics,
  const torch::Tensor& ii,
  const torch::Tensor& jj,
  const float beta)
{
  auto opts = poses.options();
  const int num = ii.size(0);

  torch::Tensor dist = torch::zeros({num}, opts);

  frame_distance_kernel(
    poses.packed_accessor32<float,2>(),
    disps.packed_accessor32<float,3>(),
    intrinsics.packed_accessor32<float,1>(),
    ii.packed_accessor32<long,1>(),
    jj.packed_accessor32<long,1>(),
    dist.packed_accessor32<float,1>(), beta);

  return dist;
}

///////////////////////////////////////////////////////////

void depth_filter_kernel(
  const torch::PackedTensorAccessor32<float,2> poses,
  const torch::PackedTensorAccessor32<float,3> disps,
  const torch::PackedTensorAccessor32<float,1> intrinsics,
  const torch::PackedTensorAccessor32<long,1> inds,
  const torch::PackedTensorAccessor32<float,1> thresh,
  torch::PackedTensorAccessor32<float,3> counter)
{
  const int num = inds.size(0);
  const int ht = disps.size(1);
  const int wd = disps.size(2);

  for(int block_id=0;block_id<num;block_id++)
    for(int neigh_id=0;neigh_id<6;neigh_id++)
    {
      int ix = static_cast<int>(inds[block_id]);
      int jx = (neigh_id < 3) ? ix - neigh_id - 1 : ix + neigh_id;

      //Probably not relevant on CPU.
      if (jx < 0 || jx >= num) {
        return;
      }

      const float t = thresh[block_id];

      float fx,fy,cx,cy;
      load_intrinsics(intrinsics,fx,fy,cx,cy);
      float ti[3], tj[3], tij[3];
      float qi[4], qj[4], qij[4];
      load_relative_pose(poses,ix,jx,ti,tj,tij,qi,qj,qij);

      for(int i=0;i<ht;i++)
        for(int j=0;j<wd;j++)
        {
          const float ui = static_cast<float>(j);
          const float vi = static_cast<float>(i);
          const float di = disps[ix][i][j];

          //points 
          float Xi[4];
          float Xj[4];

          // homogenous coordinates
          Xi[0] = (ui - cx) / fx;
          Xi[1] = (vi - cy) / fy;
          Xi[2] = 1;
          Xi[3] = di;

          // transform homogenous point
          actSE3(tij, qij, Xi, Xj);

          const float uj = fx * (Xj[0] / Xj[2]) + cx;
          const float vj = fy * (Xj[1] / Xj[2]) + cy;
          const float dj = Xj[3] / Xj[2];

          const int u0 = static_cast<int>(floor(uj));
          const int v0 = static_cast<int>(floor(vj));

          if (u0 >= 0 && v0 >= 0 && u0 < wd-1 && v0 < ht-1) {
            const float wx = ceil(uj) - uj;
            const float wy = ceil(vj) - vj;

            const float d00 = disps[jx][v0+0][u0+0];
            const float d01 = disps[jx][v0+0][u0+1];
            const float d10 = disps[jx][v0+1][u0+0];
            const float d11 = disps[jx][v0+1][u0+1];

            const float dj_hat = wy*wx*d00 + wy*(1-wx)*d01 + (1-wy)*wx*d10 + (1-wy)*(1-wx)*d11;

            const float err = abs(1.0/dj - 1.0/dj_hat);
            if       (abs(1.0/dj - 1.0/d00) < t) counter[block_id][i][j]+=1.0f;
            else if  (abs(1.0/dj - 1.0/d01) < t) counter[block_id][i][j]+=1.0f;
            else if  (abs(1.0/dj - 1.0/d10) < t) counter[block_id][i][j]+=1.0f;
            else if  (abs(1.0/dj - 1.0/d11) < t) counter[block_id][i][j]+=1.0f;
          }
        }
    }
}

torch::Tensor depth_filter_cpu(
  const torch::Tensor& poses,
  const torch::Tensor& disps,
  const torch::Tensor& intrinsics,
  const torch::Tensor& ix,
  const torch::Tensor& thresh)
{
  const int num = ix.size(0);
  const int ht = disps.size(1);
  const int wd = disps.size(2);

  torch::Tensor counter = torch::zeros({num, ht, wd}, disps.options());

  depth_filter_kernel(
    poses.packed_accessor32<float,2>(),
    disps.packed_accessor32<float,3>(),
    intrinsics.packed_accessor32<float,1>(),
    ix.packed_accessor32<long,1>(),
    thresh.packed_accessor32<float,1>(),
    counter.packed_accessor32<float,3>());

  return counter;
}

///////////////////////////////////////////////////////////

void iproj_kernel(
  const torch::PackedTensorAccessor32<float,2> poses,
  const torch::PackedTensorAccessor32<float,3> disps,
  const torch::PackedTensorAccessor32<float,1> intrinsics,
  torch::PackedTensorAccessor32<float,4> points)
{
  const int num = poses.size(0);
  const int ht = disps.size(1);
  const int wd = disps.size(2);

  for(int block_id=0;block_id<num;block_id++)
  {
    float fx,fy,cx,cy;
    load_intrinsics(intrinsics,fx,fy,cx,cy);
    float ti[3];
    float qi[4];
    load_pose(poses,block_id,ti,qi);

    for(int i=0;i<ht;i++)
      for(int j=0;j<wd;j++)
      {
        const float ui = static_cast<float>(j);
        const float vi = static_cast<float>(i);
        const float di = disps[block_id][i][j];

        //points 
        float Xi[4];
        float Xj[4];

        // homogenous coordinates
        Xi[0] = (ui - cx) / fx;
        Xi[1] = (vi - cy) / fy;
        Xi[2] = 1;
        Xi[3] = di;

        // transform homogenous point
        actSE3(ti, qi, Xi, Xj);

        points[block_id][i][j][0] = Xj[0] / Xj[3];
        points[block_id][i][j][1] = Xj[1] / Xj[3];
        points[block_id][i][j][2] = Xj[2] / Xj[3];
      }
  }
}

torch::Tensor iproj_cpu(
  const torch::Tensor& poses,
  const torch::Tensor& disps,
  const torch::Tensor& intrinsics)
{
  const int nm = disps.size(0);
  const int ht = disps.size(1);
  const int wd = disps.size(2);

  auto opts = disps.options();
  torch::Tensor points = torch::zeros({nm, ht, wd, 3}, opts);

  iproj_kernel(
    poses.packed_accessor32<float,2>(),
    disps.packed_accessor32<float,3>(),
    intrinsics.packed_accessor32<float,1>(),
    points.packed_accessor32<float,4>());

  return points;
}

///////////////////////////////////////////////////////////

void accum_kernel(
  const torch::PackedTensorAccessor32<float,2> inps,
  const torch::PackedTensorAccessor32<long,1> ptrs,
  const torch::PackedTensorAccessor32<long,1> idxs,
  torch::PackedTensorAccessor32<float,2> outs)
{
  int count=ptrs.size(0)-1;
  int D = inps.size(2);
  for(int block_id=0;block_id<count;block_id++)
  {
    const int start = ptrs[block_id];
    const int end = ptrs[block_id+1];
    for(int k=0; k<D; k++)
    {
      float x = 0;
      for (int i=start; i<end; i++) {
        x += inps[idxs[i]][k];
      }
      outs[block_id][k] = x;
    }
  }
}

torch::Tensor accum_cpu(
  const torch::Tensor& inps,
  const torch::Tensor& ptrs,
  const torch::Tensor& idxs)
{
  int count=ptrs.size(0)-1;
  torch::Tensor out = torch::zeros({count,inps.size(1)},inps.options());
  accum_kernel(
    inps.packed_accessor32<float,2>(),
    ptrs.packed_accessor32<long,1>(),
    idxs.packed_accessor32<long,1>(),
    out.packed_accessor32<float,2>());
  return out;
}

///////////////////////////////////////////////////////////

void pose_retr_kernel(
  torch::PackedTensorAccessor32<float,2> poses,
  const torch::PackedTensorAccessor32<float,2> dx,
  const int t0, const int t1)
{
  for(int k=0;k<poses.size(0);k++)
  {
    float xi[6], q[4], q1[4], t[3], t1[3];
    load_pose(poses,k,q,t);
    for (int n=0; n<6; n++)
      xi[n] = dx[k-t0][n];
    retrSE3(xi, t, q, t1, q1);
    store_pose(poses,k,q1,t1);
  }
}

void pose_retr_cpu(
  torch::Tensor& poses,
  const torch::Tensor& dx,
  const int t0,
  const int t1)
{
  pose_retr_kernel(
    poses.packed_accessor32<float,2>(),
    dx.packed_accessor32<float,2>(),
    t0, t1);
}

void disp_retr_kernel(
  torch::PackedTensorAccessor32<float,3> disps,
  const torch::PackedTensorAccessor32<float,2> dz,
  const torch::PackedTensorAccessor32<long,1> inds)
{
  for(int block_id=0;block_id<disps.size(0);block_id++)
  {
    const int i = inds[block_id];
    const int ht = disps.size(1);
    const int wd = disps.size(2);

    int k=-1;
    for(int u=0;u<ht;u++)
      for(int v=0;v<wd;v++)
      {
        k++;
        float d = disps[i][u][v] + dz[block_id][k];
        disps[i][u][v] = d;
      }
  }
}

void disp_retr_cpu(
  torch::Tensor disps,
  const torch::Tensor& dz,
  const torch::Tensor& inds)
{
  disp_retr_kernel(
    disps.packed_accessor32<float,3>(),
    dz.packed_accessor32<float,2>(),
    inds.packed_accessor32<long,1>());
}

void EEt6x6_kernel(
    const torch::PackedTensorAccessor32<float,3> E,
    const torch::PackedTensorAccessor32<float,2> Q,
    const torch::PackedTensorAccessor32<long,2> idx,
    torch::PackedTensorAccessor32<float,3> S)
{
  for(int block_id=0;block_id<idx.size(0);block_id++)
  {
    // indices
    const int ix = idx[block_id][0];
    const int jx = idx[block_id][1];
    const int kx = idx[block_id][2];

    const int D = E.size(2);

    float dS[6][6];
    float ei[6];
    float ej[6];

    memset(dS,0,6*6*sizeof(float));

    for(int k=0;k<D;k++)
    {
      const float q = Q[kx][k];
      
      // coalesced memory read
      for (int n=0; n<6; n++) {
        ei[n] = E[ix][n][k] * q;
        ej[n] = E[jx][n][k];
      }

      // block EEt
      for (int n=0; n<6; n++) {
        for (int m=0; m<6; m++) {
          dS[n][m] += ei[n] * ej[m];
        }
      }
    }

    for (int n=0; n<6; n++) {
      for (int m=0; m<6; m++) {
        S[block_id][n][m] = dS[n][m];
      }
    }
  }
}

void EEt6x6_cpu(
  const torch::Tensor& E,
  const torch::Tensor& Q,
  const torch::Tensor& idx,
  torch::Tensor& S)
{
  EEt6x6_kernel(
    E.packed_accessor32<float,3>(),
    Q.packed_accessor32<float,2>(),
    idx.packed_accessor32<long,2>(),
    S.packed_accessor32<float,3>());
}

void Ev6x1_cpu(
  const torch::Tensor& E,
  const torch::Tensor& Q,
  const torch::Tensor& w,
  const torch::Tensor& idx,
  torch::Tensor& v);

void EvT6x1_cpu(
  const torch::Tensor& E,
  const torch::Tensor& x,
  const torch::Tensor& idx,
  torch::Tensor& w);
