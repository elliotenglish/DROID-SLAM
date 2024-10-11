from setuptools import setup
from cyborg.utilities.portable_extension import PortableExtension,BuildExtension,cuda_enabled

import os.path as osp
ROOT = osp.dirname(osp.abspath(__file__))

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
                'cxx': ['-g',"-O0"],
                'nvcc': ['-g',"-O0",
                    #'-gencode=arch=compute_60,code=sm_60',
                    #'-gencode=arch=compute_61,code=sm_61',
                    #'-gencode=arch=compute_70,code=sm_70',
                    #'-gencode=arch=compute_75,code=sm_75',
                    #'-gencode=arch=compute_80,code=sm_80',
                    '-gencode=arch=compute_86,code=sm_86',
                ]
            }),
    ],
    cmdclass={ 'build_ext' : BuildExtension }
)
