#pragma once

#ifdef __CUDACC__
#include <cuda.h>
#include <cuda_runtime.h>
#endif

//CUDA_ERROR_CHECK()
//cudaDeviceSynchronize()

#ifdef __CUDACC__
inline void cudaCheckError() {                                          \
 cudaError_t e=cudaGetLastError();                                 \
 if(e!=cudaSuccess) {                                              \
   printf("CUDA failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
   exit(0); \
 }                                                                 \
}
#endif

#ifdef __CUDACC__
#define DEVICE_DECORATOR __device__
#else
#define DEVICE_DECORATOR
#endif
