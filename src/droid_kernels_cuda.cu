#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime.h>

#include <vector>
#include <iostream>

#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Parallel.h>

// #include "utils.cuh"

#include <Eigen/Sparse>
#include <Eigen/SparseCore>
#include <Eigen/SparseCholesky>

#include "lie_groups.h"
#include "droid_kernels.h"

// typedef Eigen::SparseMatrix<double> SpMat;
// typedef Eigen::Triplet<double> T;
// typedef std::vector<std::vector<int>> graph_t;
// typedef std::vector<torch::Tensor> tensor_list_t;

#define THREADS 256
#define NUM_BLOCKS(batch_size) ((batch_size + THREADS - 1) / THREADS)


#define GPU_1D_KERNEL_LOOP(k, n) \
  for (size_t k = threadIdx.x; k<n; k += blockDim.x)


__device__ void warpReduce(volatile float *sdata, unsigned int tid) {
  sdata[tid] += sdata[tid + 32];
  sdata[tid] += sdata[tid + 16];
  sdata[tid] += sdata[tid +  8];
  sdata[tid] += sdata[tid +  4];
  sdata[tid] += sdata[tid +  2];
  sdata[tid] += sdata[tid +  1];
}

__device__ void blockReduce(volatile float *sdata) {
  unsigned int tid = threadIdx.x;
  __syncthreads();

  // if (threadIdx.x < 256) {sdata[tid] += sdata[tid + 256]; } __syncthreads();
  if (threadIdx.x < 128) {sdata[tid] += sdata[tid + 128]; } __syncthreads();
  if (threadIdx.x <  64) {sdata[tid] += sdata[tid +  64]; } __syncthreads();

  if (tid < 32) warpReduce(sdata, tid);
  __syncthreads();
}

///////////////////////////////////////////////////////////

__global__ void projective_transform_kernel(
  const torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> target,
  const torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> weight,
  const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> poses,
  const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> disps,
  const torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> intrinsics,
  const torch::PackedTensorAccessor32<IndexType,1,torch::RestrictPtrTraits> ii,
  const torch::PackedTensorAccessor32<IndexType,1,torch::RestrictPtrTraits> jj,
  torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> Hs,
  torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> vs,
  torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> Eii,
  torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> Eij,
  torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> Cii,
  torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> bz)
{
  const int block_id = blockIdx.x;
  const int thread_id = threadIdx.x;

  const int ht = disps.size(1);
  const int wd = disps.size(2);

  IndexType ix = static_cast<IndexType>(ii[block_id]);
  IndexType jx = static_cast<IndexType>(jj[block_id]);

  __shared__ float fx;
  __shared__ float fy;
  __shared__ float cx;
  __shared__ float cy;

  __shared__ float ti[3], tj[3], tij[3];
  __shared__ float qi[4], qj[4], qij[4];

  // load intrinsics from global memory
  if (thread_id == 0) {
    fx = intrinsics[0];
    fy = intrinsics[1];
    cx = intrinsics[2];
    cy = intrinsics[3];
  }

  __syncthreads();

  // stereo frames
  if (ix == jx) {
    if (thread_id == 0) {
      tij[0] =  -0.1;
      tij[1] =     0;
      tij[2] =     0;
      qij[0] =     0;
      qij[1] =     0;
      qij[2] =     0;
      qij[3] =     1;
    }
  }

  else {

    // load poses from global memory
    if (thread_id < 3) {
      ti[thread_id] = poses[ix][thread_id];
      tj[thread_id] = poses[jx][thread_id];
    }

    if (thread_id < 4) {
      qi[thread_id] = poses[ix][thread_id+3];
      qj[thread_id] = poses[jx][thread_id+3];
    }

    __syncthreads();

    if (thread_id == 0) {
      relSE3(ti, qi, tj, qj, tij, qij);
    }
  }

  __syncthreads();

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

  __syncthreads();

  GPU_1D_KERNEL_LOOP(k, ht*wd) {

    const int i = k / wd;
    const int j = k % wd;

    // if(i!=3 || j!=2) continue;

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

    // printf("x k=%d i=%d j=%d Jj=(%g,%g,%g,%g,%g,%g) wu=%g wv=%g ru=%g rv=%g Jz=%g\n",
    //   k,i,j,Jj[0],Jj[1],Jj[2],Jj[3],Jj[4],Jj[5],wu,wv,ru,rv,Jz);

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
      // printf("[%d][%d][%d] Eii=%g Eij=%g\n",block_id,n,k,Eii[block_id][n][k],Eij[block_id][n][k]);
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

    // printf("x k=%d i=%d j=%d Jj=(%g,%g,%g,%g,%g,%g) wu=%g wv=%g ru=%g rv=%g Jz=%g\n",
    //   k,i,j,Jj[0],Jj[1],Jj[2],Jj[3],Jj[4],Jj[5],wu,wv,ru,rv,Jz);

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
      // printf("[%d][%d][%d] Eii=%g Eij=%g\n",block_id,n,k,Eii[block_id][n][k],Eij[block_id][n][k]);
    }

    // l=0;
    // float sum=0;
    // for (int n=0; n<12; n++)
    //   for (int m=0; m<=n; m++)
    //   {
    //     if(!(l%11))
    //       printf("hij[%d]=%g\n",l,hij[l]);
    //     l++;
    //     sum+=hij[l];
    //   }
    // printf("sum=%g\n",sum);
  }

  __syncthreads();

  __shared__ float sdata[THREADS];
  for (int n=0; n<6; n++) {
    sdata[threadIdx.x] = vi[n];
    blockReduce(sdata);
    if (threadIdx.x == 0) {
      vs[0][block_id][n] = sdata[0];
    }

    __syncthreads();

    sdata[threadIdx.x] = vj[n];
    blockReduce(sdata);
    if (threadIdx.x == 0) {
      vs[1][block_id][n] = sdata[0];
    }

  }

  l=0;
  for (int n=0; n<12; n++) {
    for (int m=0; m<=n; m++) {
      sdata[threadIdx.x] = hij[l];
      blockReduce(sdata);

      if (threadIdx.x == 0) {
        // if(!(l%11))
        //   printf("sdata[%d]=%g\n",l,sdata[0]);
        if (n<6 && m<6) {
          Hs[0][block_id][n][m] = sdata[0];
          Hs[0][block_id][m][n] = sdata[0];
        }
        else if (n >=6 && m<6) {
          Hs[1][block_id][m][n-6] = sdata[0];
          Hs[2][block_id][n-6][m] = sdata[0];
        }
        else {
          Hs[3][block_id][n-6][m-6] = sdata[0];
          Hs[3][block_id][m-6][n-6] = sdata[0];
        }

        // printf("Hs0=%g\n",Hs[0][0][0][0]);
      }

      l++;
    }
  }
}

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
  torch::Tensor& wi)
{
  const int num = ii.size(0);
  projective_transform_kernel<<<num, THREADS>>>(
    targets.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
    weights.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
    poses.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
    disps.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
    intrinsics.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
    ii.packed_accessor32<IndexType,1,torch::RestrictPtrTraits>(),
    jj.packed_accessor32<IndexType,1,torch::RestrictPtrTraits>(),
    Hs.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
    vs.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
    Eii.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
    Eij.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
    Cii.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
    wi.packed_accessor32<float,2,torch::RestrictPtrTraits>());
}

///////////////////////////////////////////////////////////

__global__ void projmap_kernel(
  const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> poses,
  const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> disps,
  const torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> intrinsics,
  const torch::PackedTensorAccessor32<IndexType,1,torch::RestrictPtrTraits> ii,
  const torch::PackedTensorAccessor32<IndexType,1,torch::RestrictPtrTraits> jj,
  torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> coords,
  torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> valid)
{

  const int block_id = blockIdx.x;
  const int thread_id = threadIdx.x;

  const int ht = disps.size(1);
  const int wd = disps.size(2);

  __shared__ int ix;
  __shared__ int jx;

  __shared__ float fx;
  __shared__ float fy;
  __shared__ float cx;
  __shared__ float cy;

  __shared__ float ti[3], tj[3], tij[3];
  __shared__ float qi[4], qj[4], qij[4];

  // load intrinsics from global memory
  if (thread_id == 0) {
    ix = static_cast<IndexType>(ii[block_id]);
    jx = static_cast<IndexType>(jj[block_id]);
    fx = intrinsics[0];
    fy = intrinsics[1];
    cx = intrinsics[2];
    cy = intrinsics[3];
  }

  __syncthreads();

  // load poses from global memory
  if (thread_id < 3) {
    ti[thread_id] = poses[ix][thread_id];
    tj[thread_id] = poses[jx][thread_id];
  }

  if (thread_id < 4) {
    qi[thread_id] = poses[ix][thread_id+3];
    qj[thread_id] = poses[jx][thread_id+3];
  }

  __syncthreads();

  if (thread_id == 0) {
    relSE3(ti, qi, tj, qj, tij, qij);
  }

  //points 
  float Xi[4];
  float Xj[4];

  __syncthreads();

  GPU_1D_KERNEL_LOOP(k, ht*wd) {
    const int i = k / wd;
    const int j = k % wd;

    const float u = static_cast<float>(j);
    const float v = static_cast<float>(i);
    
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

std::vector<torch::Tensor> projmap_cuda(
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

  projmap_kernel<<<num, THREADS>>>(
    poses.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
    disps.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
    intrinsics.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
    ii.packed_accessor32<IndexType,1,torch::RestrictPtrTraits>(),
    jj.packed_accessor32<IndexType,1,torch::RestrictPtrTraits>(),
    coords.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
    valid.packed_accessor32<float,4,torch::RestrictPtrTraits>());

  return {coords, valid};
}

///////////////////////////////////////////////////////////

__global__ void frame_distance_kernel(
  const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> poses,
  const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> disps,
  const torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> intrinsics,
  const torch::PackedTensorAccessor32<IndexType,1,torch::RestrictPtrTraits> ii,
  const torch::PackedTensorAccessor32<IndexType,1,torch::RestrictPtrTraits> jj,
  torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> dist,
  const float beta) {

  const int block_id = blockIdx.x;
  const int thread_id = threadIdx.x;

  const int ht = disps.size(1);
  const int wd = disps.size(2);

  __shared__ int ix;
  __shared__ int jx;

  __shared__ float fx;
  __shared__ float fy;
  __shared__ float cx;
  __shared__ float cy;

  __shared__ float ti[3], tj[3], tij[3];
  __shared__ float qi[4], qj[4], qij[4];

  // load intrinsics from global memory
  if (thread_id == 0) {
    ix = static_cast<IndexType>(ii[block_id]);
    jx = static_cast<IndexType>(jj[block_id]);
    fx = intrinsics[0];
    fy = intrinsics[1];
    cx = intrinsics[2];
    cy = intrinsics[3];
  }

  __syncthreads();

  //points 
  float Xi[4];
  float Xj[4];

  __shared__ float accum[THREADS]; accum[thread_id] = 0;
  __shared__ float valid[THREADS]; valid[thread_id] = 0;
  __shared__ float total[THREADS]; total[thread_id] = 0;

  __syncthreads();

  for (int n=0; n<1; n++) {

    if (thread_id < 3) {
      ti[thread_id] = poses[ix][thread_id];
      tj[thread_id] = poses[jx][thread_id];
    }

    if (thread_id < 4) {
      qi[thread_id] = poses[ix][thread_id+3];
      qj[thread_id] = poses[jx][thread_id+3];
    }

    __syncthreads();


    relSE3(ti, qi, tj, qj, tij, qij);

    float d, du, dv;

    GPU_1D_KERNEL_LOOP(k, ht*wd) {
      const int i = k / wd;
      const int j = k % wd;

      const float u = static_cast<float>(j);
      const float v = static_cast<float>(i);


      // if (disps[ix][i][j] < 0.01) {
      //   continue;
      // }
      
      // homogenous coordinates
      Xi[0] = (u - cx) / fx;
      Xi[1] = (v - cy) / fy;
      Xi[2] = 1;
      Xi[3] = disps[ix][i][j];

      // transform homogenous point
      actSE3(tij, qij, Xi, Xj);

      du = fx * (Xj[0] / Xj[2]) + cx - u;
      dv = fy * (Xj[1] / Xj[2]) + cy - v;
      d = sqrtf(du*du + dv*dv);

      total[threadIdx.x] += beta;
      
      if (Xj[2] > MIN_DEPTH) {
        accum[threadIdx.x] += beta * d;
        valid[threadIdx.x] += beta;
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

      total[threadIdx.x] += (1 - beta);
      
      if (Xj[2] > MIN_DEPTH) {
        accum[threadIdx.x] += (1 - beta) * d;
        valid[threadIdx.x] += (1 - beta);
      }
    }

    if (threadIdx.x == 0) {
      int tmp = ix;
      ix = jx;
      jx = tmp;
    }

    __syncthreads();

  }
  __syncthreads(); blockReduce(accum);
  __syncthreads(); blockReduce(total);
  __syncthreads(); blockReduce(valid);

  __syncthreads();

  if (thread_id == 0) {
    dist[block_id] = (valid[0] / (total[0] + 1e-8) < 0.75) ? 1000.0 : accum[0] / valid[0];
  }
}

torch::Tensor frame_distance_cuda(
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

  frame_distance_kernel<<<num, THREADS>>>(
    poses.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
    disps.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
    intrinsics.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
    ii.packed_accessor32<IndexType,1,torch::RestrictPtrTraits>(),
    jj.packed_accessor32<IndexType,1,torch::RestrictPtrTraits>(),
    dist.packed_accessor32<float,1,torch::RestrictPtrTraits>(), beta);

  return dist;
}

///////////////////////////////////////////////////////////

__global__ void depth_filter_kernel(
  const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> poses,
  const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> disps,
  const torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> intrinsics,
  const torch::PackedTensorAccessor32<IndexType,1,torch::RestrictPtrTraits> inds,
  const torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> thresh,
  torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> counter)
{

  const int block_id = blockIdx.x;
  const int neigh_id = blockIdx.y;
  const int index = blockIdx.z * blockDim.x + threadIdx.x;

  // if (threadIdx.x == 0) {
  //   printf("%d %d %d %d\n", blockIdx.x, blockIdx.y, blockDim.x, threadIdx.x);
  // }

  const int num = disps.size(0);
  const int ht = disps.size(1);
  const int wd = disps.size(2);

  __shared__ IndexType ix;
  __shared__ IndexType jx;

  __shared__ float fx;
  __shared__ float fy;
  __shared__ float cx;
  __shared__ float cy;

  __shared__ float ti[3], tj[3], tij[3];
  __shared__ float qi[4], qj[4], qij[4];

  if (threadIdx.x == 0) {
    ix = static_cast<IndexType>(inds[block_id]);
    jx = (neigh_id < 3) ? ix - neigh_id - 1 : ix + neigh_id;
    fx = intrinsics[0];
    fy = intrinsics[1];
    cx = intrinsics[2];
    cy = intrinsics[3];
  }

  __syncthreads();

  if (jx < 0 || jx >= num) {
    return;
  }

  const float t = thresh[block_id];

  // load poses from global memory
  if (threadIdx.x < 3) {
    ti[threadIdx.x] = poses[ix][threadIdx.x];
    tj[threadIdx.x] = poses[jx][threadIdx.x];
  }

  if (threadIdx.x < 4) {
    qi[threadIdx.x] = poses[ix][threadIdx.x+3];
    qj[threadIdx.x] = poses[jx][threadIdx.x+3];
  }

  __syncthreads();

  if (threadIdx.x == 0) {
    relSE3(ti, qi, tj, qj, tij, qij);
  }

  //points 
  float Xi[4];
  float Xj[4];

  __syncthreads();

  if (index < ht*wd) {
    const int i = index / wd;
    const int j = index % wd;

    const float ui = static_cast<float>(j);
    const float vi = static_cast<float>(i);
    const float di = disps[ix][i][j];
    
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
      if       (abs(1.0/dj - 1.0/d00) < t) atomicAdd(&counter[block_id][i][j], 1.0f);
      else if  (abs(1.0/dj - 1.0/d01) < t) atomicAdd(&counter[block_id][i][j], 1.0f);
      else if  (abs(1.0/dj - 1.0/d10) < t) atomicAdd(&counter[block_id][i][j], 1.0f);
      else if  (abs(1.0/dj - 1.0/d11) < t) atomicAdd(&counter[block_id][i][j], 1.0f);
    }
  }
}

torch::Tensor depth_filter_cuda(
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

  dim3 blocks(num, 6, NUM_BLOCKS(ht * wd));

  depth_filter_kernel<<<blocks, THREADS>>>(
    poses.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
    disps.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
    intrinsics.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
    ix.packed_accessor32<IndexType,1,torch::RestrictPtrTraits>(),
    thresh.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
    counter.packed_accessor32<float,3,torch::RestrictPtrTraits>());

  return counter;
}

///////////////////////////////////////////////////////////

__global__ void iproj_kernel(
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> poses,
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> disps,
    const torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> intrinsics,
    torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> points)

{

  const int block_id = blockIdx.x;
  const int index = blockIdx.y * blockDim.x + threadIdx.x;


  const int num = disps.size(0);
  const int ht = disps.size(1);
  const int wd = disps.size(2);

  __shared__ float fx;
  __shared__ float fy;
  __shared__ float cx;
  __shared__ float cy;

  __shared__ float t[3];
  __shared__ float q[4];

  if (threadIdx.x == 0) {
    fx = intrinsics[0];
    fy = intrinsics[1];
    cx = intrinsics[2];
    cy = intrinsics[3];
  }

  __syncthreads();


  // load poses from global memory
  if (threadIdx.x < 3) {
    t[threadIdx.x] = poses[block_id][threadIdx.x];
  }

  if (threadIdx.x < 4) {
    q[threadIdx.x] = poses[block_id][threadIdx.x+3];
  }

  __syncthreads();

  //points 
  float Xi[4];
  float Xj[4];

  if (index < ht*wd) {
    const int i = index / wd;
    const int j = index % wd;

    const float ui = static_cast<float>(j);
    const float vi = static_cast<float>(i);
    const float di = disps[block_id][i][j];
    
    // homogenous coordinates
    Xi[0] = (ui - cx) / fx;
    Xi[1] = (vi - cy) / fy;
    Xi[2] = 1;
    Xi[3] = di;

    // transform homogenous point
    actSE3(t, q, Xi, Xj);

    points[block_id][i][j][0] = Xj[0] / Xj[3];
    points[block_id][i][j][1] = Xj[1] / Xj[3];
    points[block_id][i][j][2] = Xj[2] / Xj[3];

  }
}

torch::Tensor iproj_cuda(
  const torch::Tensor& poses,
  const torch::Tensor& disps,
  const torch::Tensor& intrinsics)
{

  const int nm = disps.size(0);
  const int ht = disps.size(1);
  const int wd = disps.size(2);

  auto opts = disps.options();
  torch::Tensor points = torch::zeros({nm, ht, wd, 3}, opts);

  dim3 blocks(nm, NUM_BLOCKS(ht * wd));

  iproj_kernel<<<blocks, THREADS>>>(
    poses.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
    disps.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
    intrinsics.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
    points.packed_accessor32<float,4,torch::RestrictPtrTraits>());

  return points;

}

///////////////////////////////////////////////////////////

__global__ void accum_kernel(
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> inps,
    const torch::PackedTensorAccessor32<IndexType,1,torch::RestrictPtrTraits> ptrs,
    const torch::PackedTensorAccessor32<IndexType,1,torch::RestrictPtrTraits> idxs,
    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> outs)
{
  
  const int block_id = blockIdx.x;
  const int D = inps.size(2);

  const IndexType start = ptrs[block_id];
  const IndexType end = ptrs[block_id+1];

  for (int k=threadIdx.x; k<D; k+=blockDim.x) {
    float x = 0;
    for (IndexType i=start; i<end; i++) {
      x += inps[idxs[i]][k];
    }
    outs[block_id][k] = x;
  }  
}

torch::Tensor accum_cuda(
  const torch::Tensor& inps,
  const torch::Tensor& ptrs,
  const torch::Tensor& idxs)
{
  int count=ptrs.size(0)-1;
  torch::Tensor out = torch::zeros({count,inps.size(1)},inps.options());
  accum_kernel<<<count, THREADS>>>(
    inps.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
    ptrs.packed_accessor32<IndexType,1,torch::RestrictPtrTraits>(),
    idxs.packed_accessor32<IndexType,1,torch::RestrictPtrTraits>(),
    out.packed_accessor32<float,2,torch::RestrictPtrTraits>());
  return out;
}

///////////////////////////////////////////////////////////

__global__ void pose_retr_kernel(
  torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> poses,
  const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> dx,
  const int t0, const int t1)
{

  for (int k=t0+threadIdx.x; k<t1; k+=blockDim.x) {
    float xi[6], q[4], q1[4], t[3], t1[3];

    t[0] = poses[k][0];
    t[1] = poses[k][1];
    t[2] = poses[k][2];

    q[0] = poses[k][3];
    q[1] = poses[k][4];
    q[2] = poses[k][5];
    q[3] = poses[k][6];
    
    for (int n=0; n<6; n++) {
      xi[n] = dx[k-t0][n];
    }

    retrSE3(xi, t, q, t1, q1);

    poses[k][0] = t1[0];
    poses[k][1] = t1[1];
    poses[k][2] = t1[2];

    poses[k][3] = q1[0];
    poses[k][4] = q1[1];
    poses[k][5] = q1[2];
    poses[k][6] = q1[3];
  }
}

void pose_retr_cuda(
  torch::Tensor& poses,
  const torch::Tensor& dx,
  const int t0,
  const int t1)
{
  pose_retr_kernel<<<1, THREADS>>>(
    poses.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
    dx.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
    t0, t1);
}

///////////////////////////////////////////////////////////

__global__ void disp_retr_kernel(
  torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> disps,
  const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> dz,
  const torch::PackedTensorAccessor32<IndexType,1,torch::RestrictPtrTraits> inds)
{
  const int i = inds[blockIdx.x];
  const int ht = disps.size(1);
  const int wd = disps.size(2);

  for (int k=threadIdx.x; k<ht*wd; k+=blockDim.x) {
    float d = disps[i][k/wd][k%wd] + dz[blockIdx.x][k];
    disps[i][k/wd][k%wd] = d;
  }
}

void disp_retr_cuda(
  torch::Tensor disps,
  const torch::Tensor& dz,
  const torch::Tensor& inds)
{
  disp_retr_kernel<<<inds.size(0), THREADS>>>(
    disps.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
    dz.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
    inds.packed_accessor32<IndexType,1,torch::RestrictPtrTraits>());
}

///////////////////////////////////////////////////////////

__global__ void EEt6x6_kernel(
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> E,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> Q,
    const torch::PackedTensorAccessor32<IndexType,2,torch::RestrictPtrTraits> idx,
    torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> S)
{
  // indices
  const IndexType ix = idx[blockIdx.x][0];
  const IndexType jx = idx[blockIdx.x][1];
  const IndexType kx = idx[blockIdx.x][2];

  const int D = E.size(2);

  float dS[6][6];
  float ei[6];
  float ej[6];

  for (int i=0; i<6; i++) {
    for (int j=0; j<6; j++) {
      dS[i][j] = 0;
    }
  }

  for (int k=threadIdx.x; k<D; k+=blockDim.x) {
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

  __syncthreads();
  __shared__ float sdata[THREADS];

  for (int n=0; n<6; n++) {
    for (int m=0; m<6; m++) {
      sdata[threadIdx.x] = dS[n][m];

      blockReduce(sdata);

      if (threadIdx.x == 0) {
        S[blockIdx.x][n][m] = sdata[0];
      }
    }
  }
}

void EEt6x6_cuda(
  const torch::Tensor& E,
  const torch::Tensor& Q,
  const torch::Tensor& idx,
  torch::Tensor& S)
{
  EEt6x6_kernel<<<idx.size(0), THREADS>>>(
    E.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
    Q.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
    idx.packed_accessor32<IndexType,2,torch::RestrictPtrTraits>(),
    S.packed_accessor32<float,3,torch::RestrictPtrTraits>());
}

///////////////////////////////////////////////////////////

__global__ void Ev6x1_kernel(
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> E,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> Q,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> w,
    const torch::PackedTensorAccessor32<IndexType,2,torch::RestrictPtrTraits> idx,
    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> v)
{
  const int D = E.size(2);
  const int kx = idx[blockIdx.x][0];

  float b[6];
  for (int n=0; n<6; n++) {
    b[n] = 0.0;
  }

  for (int k=threadIdx.x; k<D; k+=blockDim.x) {
    const float q_w = Q[kx][k] * w[kx][k];

    for (int n=0; n<6; n++) {
      b[n] += q_w * E[blockIdx.x][n][k];
    }
  }

  __syncthreads();
  __shared__ float sdata[THREADS];

  for (int n=0; n<6; n++) {
    sdata[threadIdx.x] = b[n];
    blockReduce(sdata);

    if (threadIdx.x == 0) {
      v[blockIdx.x][n] += sdata[0];
    }
  }
}

void Ev6x1_cuda(
  const torch::Tensor& E,
  const torch::Tensor& Q,
  const torch::Tensor& w,
  const torch::Tensor& idx,
  torch::Tensor& v)
{
  Ev6x1_kernel<<<idx.size(0), THREADS>>>(
    E.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
    Q.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
    w.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
    idx.packed_accessor32<IndexType,2,torch::RestrictPtrTraits>(),
    v.packed_accessor32<float,2,torch::RestrictPtrTraits>());
}

///////////////////////////////////////////////////////////

__global__ void EvT6x1_kernel(
  const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> E,
  const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> x,
  const torch::PackedTensorAccessor32<IndexType,1,torch::RestrictPtrTraits> idx,
  torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> w)
{

  const int D = E.size(2);
  const IndexType ix = idx[blockIdx.x];

  if (idx[blockIdx.x] <= 0 || idx[blockIdx.x] >= x.size(0))
    return;

  for (int k=threadIdx.x; k<D; k+=blockDim.x) {
    float dw = 0;
    for (int n=0; n<6; n++) {
      dw += E[blockIdx.x][n][k] * x[ix][n];
    }
    w[blockIdx.x][k] = dw;
  }
}

void EvT6x1_cuda(
  const torch::Tensor& E,
  const torch::Tensor& x,
  const torch::Tensor& idx,
  torch::Tensor& w)
{
  EvT6x1_kernel<<<idx.size(0), THREADS>>>(
    E.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
    x.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
    idx.packed_accessor32<IndexType,1,torch::RestrictPtrTraits>(),
    w.packed_accessor32<float,2,torch::RestrictPtrTraits>());
}
