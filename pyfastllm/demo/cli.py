# -*- coding: utf-8 -*-
import sys
import platform
import logging
import argparse
sys.path.append('./build-py')
import pyfastllm # 或fastllm

logging.info(f"python gcc version:{platform.python_compiler()}")


def args_parser():
    parser = argparse.ArgumentParser(description='pyfastllm')
    parser.add_argument('-m', '--model', type=int, required=False, default=0, help='模型类型，默认为0, 可以设置为0(chatglm),1(moss),2(vicuna),3(baichuan)')
    parser.add_argument('-p', '--path', type=str, required=True, default='', help='模型文件的路径')
    parser.add_argument('-t', '--threads', type=int, default=4,  help='使用的线程数量')
    parser.add_argument('-l', '--low', action='store_true', help='使用低内存模式')
    args = parser.parse_args()
    return args

LLM_TYPE = ""
def print_back(idx:int, content: bytearray):
    content = content.decode(encoding="utf-8", errors="replace")
    if idx >= 0:
        print(f"\r{LLM_TYPE}:{content}", end='', flush=True)
    elif idx == -1:
        print()
    sys.stdout.flush()

def main(args):
    model_path = args.path
    OLD_API = False
    if OLD_API:
        model = pyfastllm.ChatGLMModel()
        model.load_weights(model_path)
        model.warmup()
    else:
        global LLM_TYPE 
        LLM_TYPE = pyfastllm.get_llm_type(model_path)
        print(f"llm model: {LLM_TYPE}")
        model = pyfastllm.create_llm(model_path)
        

    prompt = ""
    while prompt != "stop":
        prompt = input("User: ")
        config = pyfastllm.GenerationConfig()
        model.response(model.make_input("", 0, prompt), print_back, config)
        print()
        sys.stdout.flush()

if __name__ == "__main__":
    args = args_parser()
    main(args)
