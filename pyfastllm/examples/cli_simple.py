# -*- coding: utf-8 -*-
import sys
import platform
import logging
import argparse
import fastllm

logging.info(f"python gcc version:{platform.python_compiler()}")

def args_parser():
    parser = argparse.ArgumentParser(description='fastllm')
    parser.add_argument('-m', '--model', type=int, required=False, default=0, help='模型类型，默认为0, 可以设置为0(chatglm),1(moss),2(vicuna),3(baichuan)')
    parser.add_argument('-p', '--path', type=str, required=True, default='', help='模型文件的路径')
    parser.add_argument('-t', '--threads', type=int, default=4,  help='使用的线程数量')
    parser.add_argument('-l', '--low', action='store_true', help='使用低内存模式')
    args = parser.parse_args()
    return args


def response(model, prompt_input:str, stream_output:bool=False):
    gmask_token_id = 130001
    bos_token_id = 130004

    input_ids = model.weight.tokenizer.encode(prompt_input)
    input_ids = input_ids.to_list()
    input_ids.extend([gmask_token_id, bos_token_id])
    input_ids = [int(v) for v in input_ids]
    # print(input_ids)

    handle = model.launch_response(input_ids)
    continue_token = True

    ret_byte = b""
    ret_str = ""
    
    while continue_token:
        resp_token = model.fetch_response(handle)
        continue_token = (resp_token != -1)

        content = model.weight.tokenizer.decode_byte([resp_token])
        ret_byte += content
        ret_str = ret_byte.decode(errors='ignore')

        if stream_output:
            yield ret_str

    return ret_str

def run_with_response(args):
    model_path = args.path
    OLD_API = False
    if OLD_API:
        model = fastllm.ChatGLMModel()
        model.load_weights(model_path)
        model.warmup()
    else:
        model = fastllm.create_llm(model_path)
        print(f"llm model: {model.model_type}")
    
    prompt = ""
    while prompt != "stop":
        prompt = input("User: ")

        outputs = response(model, prompt_input=prompt, stream_output=True)

        print(f"{model.model_type}:", end=' ')
        for output in outputs:
            print(f"\r{model.model_type}: {output}", end='', flush=True)
        print()


def run_with_callback(args):
    model_path = args.path
    OLD_API = False
    LLM_TYPE = ""
    if OLD_API:
        model = fastllm.ChatGLMModel()
        model.load_weights(model_path)
        model.warmup()
    else:
        LLM_TYPE = fastllm.get_llm_type(model_path)
        model = fastllm.create_llm(model_path)
    
    def print_back(idx:int, content: bytearray):
        content = content.decode(encoding="utf-8", errors="replace")
        if idx >= 0:
            print(f"\r{LLM_TYPE}:{content}", end='', flush=True)
        elif idx == -1:
            print()
        sys.stdout.flush()
        
    prompt = ""
    while prompt != "stop":
        prompt = input("User: ")
        config = fastllm.GenerationConfig()
        model.response(model.make_input("", 0, prompt), print_back, config)
        print()
        sys.stdout.flush()


if __name__ == "__main__":
    args = args_parser()
    # run_with_callback(args)
    run_with_response(args)
