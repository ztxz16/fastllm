# pyfastllm 

本地编译安装fastllm的python接口，以两种方式编译运行：
1. cpp方式：编译为动态库，需放在python运行加载目录下
2. python方式：编译为wheel包，但暂不支持cuda

### cpp方式

手动编译：
```
mkdir build-py
cd build-py
cmake .. -DUSE_CUDA=ON -DPY_API=ON
make -j4
python cli.py -p chatglm-6b-int8.bin -t 8  # 与cpp编译的运行结果保持一致
```

脚本编译：

```
cd pyfastllm
python build_libs --cuda
python cli.py -p chatglm-6b-int8.bin -t 8 
```

### python方式

```
cd pyfastllm
python setup.py build
python setup.py install 
python cli.py -p chatglm-6b-int8.bin -t 8 
```