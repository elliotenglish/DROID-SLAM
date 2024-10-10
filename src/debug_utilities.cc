#include "debug_utilities.h"

double tensor_get(torch::Tensor& x,int i0)
{
  return x[i0].item().to<double>();
}
double tensor_get(torch::Tensor& x,int i0,int i1)
{
  return x[i0][i1].item().to<double>();
}
double tensor_get(torch::Tensor& x,int i0,int i1,int i2)
{
  return x[i0][i1][i2].item().to<double>();
}
double tensor_get(torch::Tensor& x,int i0,int i1,int i2,int i3)
{
  return x[i0][i1][i2][i3].item().to<double>();
}
double tensor_get(torch::Tensor& x,int i0,int i1,int i2,int i3,int i4)
{
  return x[i0][i1][i2][i3][i4].item().to<double>();
}
int tensor_size(torch::Tensor& x,int i)
{
  return x.size(i);
}
