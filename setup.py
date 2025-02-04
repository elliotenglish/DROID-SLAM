from setuptools import setup
from portable_extension import PortableExtension,BuildExtension,cuda_enabled

import os.path as osp
ROOT = osp.dirname(osp.abspath(__file__))

optimize=True
if optimize:
    compile_flags=["-O3","-std=c++17"]
    link_flags=["-O3"]
else:
    compile_flags=["-g","-O0","-UNDEBUG","-std=c++17"]
    link_flags=["-g","-O0"]#"-Wl,--no-as-needed",

setup(
    name='droid_slam',
    packages=["droid_slam"],
    ext_modules=[
        PortableExtension('droid_backends',
            include_dirs=[
              "/usr/include/eigen3",
              "/usr/local/include/eigen3"],
            sources=[
                'src/droid.cpp',
                'src/droid_kernels_cpu.cc',
                'src/droid_kernels_cuda.cu',
                'src/correlation_kernels_cpu.cc',
                'src/correlation_kernels_cuda.cu',
                'src/altcorr_kernel.cu',
                "src/debug_utilities.cc",
            ],
            extra_compile_args={
                'cxx': compile_flags,
                'nvcc': compile_flags + [
                    #'-gencode=arch=compute_60,code=sm_60',
                    #'-gencode=arch=compute_61,code=sm_61',
                    #'-gencode=arch=compute_70,code=sm_70',
                    #'-gencode=arch=compute_75,code=sm_75',
                    #'-gencode=arch=compute_80,code=sm_80',
                    '-gencode=arch=compute_86,code=sm_86',
                ]
            },
            extra_link_args=link_flags)
    ],
    cmdclass={ 'build_ext' : BuildExtension }
)
