import os
from torch.utils.cpp_extension import CUDAExtension, CppExtension, BuildExtension

def cuda_enabled():
  return os.path.exists("/usr/local/cuda")

def PortableExtension(*args,**kwargs):
  if cuda_enabled():
    return CUDAExtension(
      *args,
      **{k:v for k,v in kwargs.items() if k not in["extra_compile_args"]},
      extra_compile_args={
        "cxx":kwargs["extra_compile_args"]["cxx"]+["-DPORTABLE_EXTENSION_CUDA_ENABLED=1"],
        "nvcc":kwargs["extra_compile_args"]["nvcc"]+["-DPORTABLE_EXTENSION_CUDA_ENABLED=1"]
      }
    )
  else:
    return CppExtension(
      *args,
      **{k:v for k,v in kwargs.items() if k not in["sources","extra_compile_args"]},
      sources=[v for v in kwargs["sources"] if not v.endswith(".cu")],
      extra_compile_args=kwargs["extra_compile_args"]["cxx"]+["-DPORTABLE_EXTENSION_CUDA_ENABLED=0"]
    )
