# -*- coding: utf-8 -*-
import sys
import platform
import logging
import argparse
import pyfastllm

logging.info(f"python gcc version:{platform.python_compiler()}")

sys.path.append('./build-py')

def args_parser():
    parser = argparse.ArgumentParser(description='pyfastllm')
    parser.add_argument('-m', '--model', type=int, required=False, default=0, help='模型类型，默认为0, 可以设置为0(chatglm),1(moss),2(vicuna)')
    parser.add_argument('-p', '--path', type=str, required=True, default='', help='模型文件的路径')
    parser.add_argument('-t', '--threads', type=int, default=4,  help='使用的线程数量')
    parser.add_argument('-l', '--low', action='store_true', help='使用低内存模式')
    args = parser.parse_args()
    return args

def print_back(idx:int, content: str):
    if idx == 0:
        print(f"ChatGLM:{content}", end='')
    elif idx > 0:
        print(f"{content}", end='')
    elif idx == -1:
        print()

    sys.stdout.flush()

def main(args):
    model_path = args.path
    model = pyfastllm.ChatGLMModel()
    model.load_weights(model_path)
    model.warmup()

    prompt = ""
    while prompt != "exit":
        prompt = input("User: ")
        model.response(prompt, print_back)
        print()
        sys.stdout.flush()

if __name__ == "__main__":
    args = args_parser()
    main(args)