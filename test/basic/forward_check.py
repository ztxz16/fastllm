from config import default_messages_list

import argparse
import ctypes
import logging
import os
import sys
import tempfile
import torch
import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from ftllm import llm

def args_parser():
    parser = argparse.ArgumentParser(description = 'fastllm_test')
    parser.add_argument('--model', type = str, required = True, default = '', help = '模型文件目录')
    parser.add_argument('--tokens', type = int, required = False, default = 8, help = '每条测试输出的token数')
    parser.add_argument('--top_logprobs', type = int, required = False, default = 10, help = '同一次FastLLM请求记录的首步TopK数量')
    parser.add_argument('--hf_device', type = str, required = False, default = 'cuda', help = 'transformer模型的device')
    parser.add_argument('--flm_dtype', type = str, required = False, default = 'float16', help = 'fastllm模型的权重类型')
    parser.add_argument('--flm_atype', type = str, required = False, default = 'float32', help = 'fastllm模型的推理类型')
    parser.add_argument('--flm_threads', type = int, required = False, default = 4, help = 'fastllm读取模型、推理使用的线程数')
    parser.add_argument('--flm_device', type = str, required = False, default = 'cuda', help = 'fastllm推理的设备')
    args = parser.parse_args()
    return args


def parse_first_top_logprobs(output):
    """从正式生成请求的原生日志解析首步TopK logprob。"""
    prefix = "[fastllm][sampling-topk] "
    for line in output.splitlines():
        if not line.startswith(prefix):
            continue
        fields = {}
        for item in line[len(prefix):].split():
            key, separator, value = item.partition("=")
            if separator:
                fields[key] = value
        try:
            token_ids = [int(value) for value in fields["token_ids"].split(",")]
            logits = [float(value) for value in fields["logits"].split(",")]
            logsumexp = float(fields["logsumexp"])
            if len(token_ids) != len(logits):
                continue
            return {
                token_id: logit - logsumexp
                for token_id, logit in zip(token_ids, logits)
            }
        except (KeyError, ValueError):
            continue
    raise RuntimeError("同一次FastLLM正式请求未输出sampling-topk诊断")


def response_with_first_top_logprobs(model, prompt, max_length, top_logprobs,
                                     repeat_penalty):
    """执行一次正式FastLLM请求，并从同次请求取得首步TopK。"""
    trace_variables = {
        "FASTLLM_SAMPLING_TOP1_TRACE": "1",
        "FASTLLM_SAMPLING_TOPK_TRACE": str(top_logprobs),
    }
    previous = {key: os.environ.get(key) for key in trace_variables}
    os.environ.update(trace_variables)

    saved_stdout = os.dup(sys.stdout.fileno())
    try:
        with tempfile.TemporaryFile(mode="w+b") as trace_file:
            sys.stdout.flush()
            os.dup2(trace_file.fileno(), sys.stdout.fileno())
            try:
                response = model.response(
                    prompt, max_length=max_length, top_k=1,
                    temperature=0.01, repeat_penalty=repeat_penalty)
            finally:
                ctypes.CDLL(None).fflush(None)
                os.dup2(saved_stdout, sys.stdout.fileno())

            trace_file.seek(0)
            trace_output = trace_file.read().decode("utf-8", errors="replace")
    finally:
        os.close(saved_stdout)
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    return response, parse_first_top_logprobs(trace_output)

if __name__ == "__main__":
    args = args_parser()

    llm.set_cpu_threads(args.flm_threads)
    llm.set_device_map(args.flm_device)
    messages_list = default_messages_list

    logger = logging.getLogger()
    logging.basicConfig(level = logging.INFO, format = '%(asctime)s - %(levelname)s - %(message)s')

    logger.info(str(args))

    model_path = args.model
    logger.info("开始测试模型 " + model_path)    
    logger.info("正在用Transformer读取模型")
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype = "auto", attn_implementation = "eager")
    if not getattr(hf_model, "is_quantized", False):
        hf_model = hf_model.half()
    hf_tokenizer = AutoTokenizer.from_pretrained(model_path)
    logger.info("读取成功")
    logger.info("正在用Fastllm读取模型")
    fastllm_model = llm.model(model_path, dtype = args.flm_dtype)
    fastllm_model.set_atype(args.flm_atype)
    fastllm_tokenizer = llm.tokenizer(model_path)
    logger.info("读取成功")
    logger.info("使用fastllm进行推理")
    repeat_penalty = hf_model.generation_config.repetition_penalty
    if repeat_penalty is None:
        repeat_penalty = 1.0

    # fastllm模型推理
    fastllm_top_logprobs_list = []
    fastllm_response_list = []
    for messages in tqdm.tqdm(messages_list):
        fastllm_text = fastllm_tokenizer.apply_chat_template(messages, tokenize = False, add_generation_prompt = True)
        fastllm_model.direct_query = True
        fastllm_response, fastllm_top_logprobs = response_with_first_top_logprobs(
            fastllm_model, fastllm_text, args.tokens,
            args.top_logprobs, repeat_penalty)

        fastllm_top_logprobs_list.append(fastllm_top_logprobs)
        fastllm_response_list.append(fastllm_response)
    logger.info("释放fastllm模型")
    fastllm_model.release_memory()

    logger.info("使用Transformer进行推理")
    hf_top_logprobs_list = []
    hf_response_list = []
    hf_model.to(args.hf_device)
    for messages in tqdm.tqdm(messages_list):
        # hf模型推理
        hf_text = hf_tokenizer.apply_chat_template (messages, tokenize = False, add_generation_prompt = True)
        hf_inputs = hf_tokenizer([hf_text], return_tensors="pt").to(args.hf_device)
        with torch.no_grad():
            hf_logits = hf_model(hf_inputs["input_ids"])[0]
            hf_last_logits = hf_logits.reshape([-1, hf_logits.shape[-1]])[-1] #取末尾token的logits
            hf_generated_ids = hf_model.generate(hf_inputs.input_ids, max_new_tokens = args.tokens, top_k = 1, temperature = 0.01)
            hf_generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(hf_inputs.input_ids, hf_generated_ids)]
            hf_response = hf_tokenizer.batch_decode(hf_generated_ids, skip_special_tokens = True)[0]
            hf_logprobs = torch.log_softmax(hf_last_logits.float(), dim=-1)
            values, token_ids = torch.topk(
                hf_logprobs, min(args.top_logprobs, hf_logprobs.numel()))
            hf_top_logprobs_list.append({
                int(token_id): float(value)
                for token_id, value in zip(token_ids.tolist(), values.tolist())
            })
            hf_response_list.append(hf_response)        
    # 结果对比
    overlaps = []
    for i in range(len(messages_list)):
        if (hf_response_list[i] != fastllm_response_list[i]):
            logger.warning("数据" + str(i) + "的生成结果不同" + 
                           "\n\n输入:\n" + str(messages_list[i]) +
                           "\n\nhf结果:\n" + hf_response_list[i] +
                           "\n\nfastllm结果:\n" + fastllm_response_list[i]);
        else:
            logger.info("数据 " + str(i) + " 的生成结果相同，结果为 \"" +
                        hf_response_list[i][:10] + "...\"")
        fastllm_tokens = set(fastllm_top_logprobs_list[i])
        hf_tokens = set(hf_top_logprobs_list[i])
        overlap = len(fastllm_tokens & hf_tokens) / max(
            1, min(len(fastllm_tokens), len(hf_tokens)))
        overlaps.append(overlap)
        logger.info("数据 " + str(i) + " 的首步TopK重合率为" + str(overlap))
    logger.info("平均首步TopK重合率: " + str(sum(overlaps) / len(overlaps)))
