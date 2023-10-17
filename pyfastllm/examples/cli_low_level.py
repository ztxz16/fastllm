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

# 请谨慎使用该函数，目前仍存在bug，仅作为low level api调用示例，请勿在生产环境使用
def response(model, prompt_input:str, stream_output:bool=False):
    gmask_token_id = 130001
    bos_token_id = 130004
    eos_token_id = 130005
    
    input_ids = model.weight.tokenizer.encode(prompt_input)
    gmask_bos = fastllm.Tensor(fastllm.float32, [1, 2], [gmask_token_id, bos_token_id])
    input_ids = fastllm.cat([input_ids, gmask_bos], 0)

    seq_len = input_ids.count(0)
    vmask = [0] * (seq_len * seq_len)
    vpids = [0] * (seq_len * 2)
    for i in range(seq_len-1):
        vmask[i*seq_len + seq_len -1] = 1
        vpids[i] = i
    vpids[seq_len - 1] = seq_len - 2
    vpids[seq_len * 2 - 1] = 1
    attention_mask = fastllm.Tensor(fastllm.float32, [seq_len, seq_len], vmask)
    position_ids = fastllm.Tensor(fastllm.float32, [2, seq_len], vpids)

    pastKeyValues = []
    for _ in range(model.block_cnt):
        pastKeyValues.append([fastllm.Tensor(fastllm.float32), fastllm.Tensor(fastllm.float32)])
    
    ret_str = ""
    ret_len = 1
    mask_ids = -1
    output_tokens = []
    penalty_factor = fastllm.Tensor()

    while len(output_tokens) < 2048: # config.max_seq_len
        ret, pastKeyValues = model.forward(input_ids, attention_mask, position_ids, penalty_factor, pastKeyValues)
        if ret == eos_token_id:
            break

        output_tokens.append(ret)
        cur_str = model.weight.tokenizer.decode(fastllm.Tensor(fastllm.float32, [len(output_tokens)], output_tokens))
        ret_str += cur_str

        print(cur_str, end="")
        sys.stdout.flush()
        if stream_output:
            yield cur_str

        ret_len += 1
        output_tokens = []

        if mask_ids == -1:
            mask_ids = seq_len - 2
        
        input_ids = fastllm.Tensor(fastllm.float32, [1, 1], [ret])
        attention_mask = fastllm.Tensor()
        position_ids = fastllm.Tensor(fastllm.float32, [2, 1], [mask_ids, ret_len])
    
    print()
    return ret_str


def run_with_low_level(args):
    model_path = args.path 
    llm_type = fastllm.get_llm_type(model_path)
    print(f"llm model: {llm_type}")
    model = fastllm.create_llm(model_path)
        
    prompt = ""
    while prompt != "stop":
        prompt = input("User: ")
        outputs = response(model, prompt_input=prompt)
        for output in outputs:
            print(output)
            sys.stdout.flush()

if __name__ == "__main__":
    args = args_parser()
    run_with_low_level(args)