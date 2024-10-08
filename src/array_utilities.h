#pragma once

#include "cuda_utilities.h"

inline DEVICE_DECORATOR
bool within_bounds(int h, int w, int H, int W) {
  return h >= 0 && h < H && w >= 0 && w < W;
}
