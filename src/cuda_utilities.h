#pragma once

#include <cuda.h>

//CUDA_ERROR_CHECK()
//cudaDeviceSynchronize()

inline void cudaCheckError() {                                          \
 cudaError_t e=cudaGetLastError();                                 \
 if(e!=cudaSuccess) {                                              \
   printf("CUDA failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
   exit(0); \
 }                                                                 \
}

#define assert_not_implemented() \
{ \
  printf("Not implemented"); \
  exit(1); \
}
