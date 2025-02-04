#pragma once

#include <torch/extension.h>

#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Parallel.h>

#include <Eigen/Sparse>
#include <Eigen/SparseCore>
#include <Eigen/SparseCholesky>

#define MIN_DEPTH 0.25

typedef int32_t IndexType;
const c10::ScalarType IndexTypeTorch=torch::kInt32;
