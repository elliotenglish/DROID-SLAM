#pragma once

#include "torch/extension.h"

#define assert_not_implemented() \
{ \
  printf("Not implemented\n"); \
  exit(1); \
}

double tensor_get(torch::Tensor& x,int i0);
double tensor_get(torch::Tensor& x,int i0,int i1);
double tensor_get(torch::Tensor& x,int i0,int i1,int i2);
double tensor_get(torch::Tensor& x,int i0,int i1,int i2,int i3);
double tensor_get(torch::Tensor& x,int i0,int i1,int i2,int i3,int i4);
int tensor_size(torch::Tensor& x,int i);
