import os
import shutil
import platform
import sys
import argparse
import glob

parser = argparse.ArgumentParser(description='build fastllm libs')
parser.add_argument('--cuda', dest='cuda', action='store_true', default=False,
                    help='build with cuda support')

IS_WINDOWS = (platform.system() == 'Windows')
IS_DARWIN = (platform.system() == 'Darwin')
IS_LINUX = (platform.system() == 'Linux')

BUILD_DIR = 'build-py' # build path

def build_libs():
    # create build dir
    root_dir = os.path.dirname(os.getcwd())
    cmake_build_dir = os.path.join(root_dir, BUILD_DIR)
    if os.path.exists(cmake_build_dir):
        shutil.rmtree(cmake_build_dir)
    os.makedirs(cmake_build_dir)
    os.chdir(cmake_build_dir)

    # build it
    cpu_num = min(os.cpu_count(), 4)
    args = parser.parse_args()
    if IS_WINDOWS:
        os.system('cmake -G Ninja -DPY_API=ON .. && ninja pyfastllm')
    elif IS_LINUX:
        extra_opts = ' -DPY_API=ON '
        extra_opts += ' -DUSE_CUDA=ON ' if args.cuda else ' '
        build_cmd = f"cmake {extra_opts} .. && make pyfastllm -j{cpu_num}"
        print(build_cmd)
        os.system(f"cmake {extra_opts} .. && make pyfastllm -j{cpu_num}")
    else:
        extra_opts = '-DPY_API=ON'
        os.system(f"cmake {extra_opts} .. && make pyfastllm -j{cpu_num}")
    
    so_files = glob.glob("*.so", root_dir=cmake_build_dir)
    for file in so_files:
        shutil.copy(os.path.join(cmake_build_dir, file), os.path.join(root_dir, "pyfastllm/fastllm"))

if __name__ == '__main__':
    build_libs()