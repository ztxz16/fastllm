import os
import shutil
import platform
import sys
import argparse

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
    args = parser.parse_args()
    if IS_WINDOWS:
        os.system('cmake -G "Ninja" -DPY_API=ON .. && ninja pyfastllm')
    elif IS_LINUX:
        extra_opts = ' -DPY_API=ON '
        extra_opts += ' -DUSE_CUDA=ON ' if args.cuda else ' '
        build_cmd = 'cmake ' + extra_opts + ' .. && make pyfastllm -j4'
        print(build_cmd)
        os.system('cmake ' + extra_opts + ' .. && make pyfastllm -j4')
    else:
        extra_opts = '-DPY_API=ON'
        os.system('cmake ' + extra_opts + '.. && make pyfastllm -j4')


if __name__ == '__main__':
    build_libs()
