from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name='hough_voting_cuda',
    ext_modules=[
        CUDAExtension(
            'hough_voting_cuda',
            ['hough_voting_kernel.cu'],
            include_dirs = ['/usr/include/eigen3', '/usr/local/include'],
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)

# from setuptools import setup
# from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# setup(
#     name='hough_voting_cuda',
#     version='0.0.0',
#     author='Your Name',
#     description='Hough Voting CUDA Extension',
#     ext_modules=[
#         CUDAExtension(
#             name='hough_voting_cuda',
#             sources=['hough_voting_kernel.cu'],
#             include_dirs = ['/usr/include/eigen3', '/usr/local/include'],
#             extra_compile_args={'cxx': ['-O3'], 'nvcc': ['-O3', '-arch=sm_70']}
#         ),
#     ],
#     cmdclass={'build_ext': BuildExtension},
#     install_requires=['torch'],
#     packages=[],
#     zip_safe=False,
# )
