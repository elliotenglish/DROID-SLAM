#include "correlation_kernels_cpu.h"
#include "array_utilities.h"
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Parallel.h>

std::vector<torch::Tensor> corr_index_cpu_forward(
  torch::Tensor volume_,
  torch::Tensor coords_,
  int radius)
{
  typedef float scalar_t;

  torch::PackedTensorAccessor32<scalar_t,5> volume=
    volume_.packed_accessor32<scalar_t,5>();
  torch::PackedTensorAccessor32<float,4> coords=
    coords_.packed_accessor32<float,4>();

  int batch_size=volume.size(0);
  int h1=volume.size(1);
  int w1=volume.size(2);
  int h2=volume.size(3);
  int w2=volume.size(4);
  int r=radius;

  int rd=2*r+1;

  auto opts = volume_.options();
  torch::Tensor corr_ = torch::zeros(
    {batch_size, rd, rd, h1, w1}, opts);
  torch::PackedTensorAccessor32<scalar_t,5> corr=
    corr_.packed_accessor32<scalar_t,5>();

  int n=0;
  for(int x=0;x<w1;x++)
    for(int y=0;y<h1;y++)
    {
      // if(x!=0 || y!=0)
      //   continue;

      float x0=coords[n][0][y][x];
      float y0=coords[n][1][y][x];

      int x0f=int(floor(x0));
      int y0f=int(floor(y0));

      float dx=x0-x0f;
      float dy=y0-y0f;

      //print(dx,dy)

      for(int i=0;i<(rd+1);i++)
        for(int j=0;j<(rd+1);j++)
        {
          int x1=x0f - r + i;
          int y1=y0f - r + j;

          //cnt+=1

          if(within_bounds(y1,x1,h2,w2))
          {
            float s=volume[n][y][x][y1][x1];
            // printf("(%d,%d,%d,%d,%d)=%g\n",n,y,x,y1,x1,s);

            if(i > 0 && j > 0)
              corr[n][i-1][j-1][y][x] += s * (dx * dy);

            if(i > 0 and j < rd)
              corr[n][i-1][j][y][x] += s * (dx * (1-dy));

            if(i < rd and j > 0)
              corr[n][i][j-1][y][x] += s * ((1-dx) * dy);

            if(i < rd and j < rd)
              corr[n][i][j][y][x] += s * ((1-dx) * (1-dy));
          }
        }
    }

  // print("cnt",cnt);

  return {corr_};
}
