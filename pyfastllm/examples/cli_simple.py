# -*- coding: utf-8 -*-
import sys, os
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

    input_ids = model.weight.tokenizer.encode(prompt_input)
    input_ids = input_ids.to_list()
    input_ids = [int(v) for v in input_ids]
    if model.model_type == "chatglm":
        input_ids = [model.gmask_token_id, model.bos_token_id] + input_ids
    # print(input_ids)

    handle = model.launch_response(input_ids, fastllm.GenerationConfig())
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
        fastllm.set_threads(args.threads)
        fastllm.set_low_memory(args.low)
        if not os.path.exists(model_path):
            print(f"模型文件{args.path}不存在！")
            exit(-1)
        model = fastllm.create_llm(model_path)
        print(f"llm model: {model.model_type}")
    print(f"欢迎使用 {model.model_type} 模型. 输入内容对话，reset清空历史记录，stop退出程序");
    
    input_text = ""
    history = ""
    dialog_round = 0
    while input_text != "stop":
        input_text = input("User: ")
        if 'stop' == input_text:
            break
        if 'reset' == input_text:
            history = ''
            continue
        prompt = model.make_input(history, dialog_round, input_text)

        outputs = response(model, prompt_input=prompt, stream_output=True)

        print(f"{model.model_type}:", end=' ')
        past_len = 0
        for output in outputs:
            print(output[past_len:], end='', flush=True)
            past_len = len(output)
        print()
        model.make_history(history, dialog_round, input_text, output)
        dialog_round += 1


def run_with_callback(args):
    model_path = args.path
    OLD_API = False
    LLM_TYPE = ""
    if OLD_API:
        model = fastllm.ChatGLMModel()
        model.load_weights(model_path)
        model.warmup()
    else:
        fastllm.set_threads(args.threads)
        fastllm.set_low_memory(args.low)
        if not os.path.exists(model_path):
            print(f"模型文件{args.path}不存在！")
            exit(-1)
        LLM_TYPE = fastllm.get_llm_type(model_path)
        model = fastllm.create_llm(model_path)
    
    def print_back(idx:int, content: bytearray):
        content = content.decode(encoding="utf-8", errors="replace")
        if idx >= 0:
            print(f"\r{LLM_TYPE}:{content}", end='', flush=True)
        elif idx == -1:
            print()
        sys.stdout.flush()
        
    print(f"欢迎使用 {LLM_TYPE} 模型. 输入内容对话，reset清空历史记录，stop退出程序");
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
