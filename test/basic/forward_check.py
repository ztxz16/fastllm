from config import compute_cosine_similarity
from config import default_messages_list

import argparse
import logging
import os
import torch
import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from ftllm import llm

def args_parser():
    parser = argparse.ArgumentParser(description = 'fastllm_test')
    parser.add_argument('--model', type = str, required = True, default = '', help = '模型文件目录')
    parser.add_argument('--tokens', type = int, required = False, default = 8, help = '每条测试输出的token数')
    parser.add_argument('--hf_device', type = str, required = False, default = 'cuda', help = 'transformer模型的device')
    parser.add_argument('--flm_dtype', type = str, required = False, default = 'float16', help = 'fastllm模型的权重类型')
    parser.add_argument('--flm_atype', type = str, required = False, default = 'float32', help = 'fastllm模型的推理类型')
    parser.add_argument('--flm_threads', type = int, required = False, default = 4, help = 'fastllm读取模型、推理使用的线程数')
    parser.add_argument('--flm_device', type = str, required = False, default = 'cuda', help = 'fastllm推理的设备')
    args = parser.parse_args()
    return args

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
    hf_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype = "auto", device_map = "cpu").half()
    hf_tokenizer = AutoTokenizer.from_pretrained(model_path)
    logger.info("读取成功")
    logger.info("正在用Fastllm读取模型")
    fastllm_model = llm.model(model_path, dtype = args.flm_dtype)
    fastllm_model.set_atype(args.flm_atype)
    fastllm_tokenizer = llm.tokenizer(model_path)
    logger.info("读取成功")
    logger.info("使用fastllm进行推理")

    # fastllm模型推理
    fastllm_logits_list = []
    fastllm_response_list = []
    for messages in tqdm.tqdm(messages_list):
        fastllm_text = fastllm_tokenizer.apply_chat_template(messages, tokenize = False, add_generation_prompt = True)
        fastllm_model.direct_query = True
        fastllm_logits = fastllm_model.response_logits(fastllm_text)
        fastllm_response = fastllm_model.response(fastllm_text, max_length = args.tokens, top_k = 1, temperature = 0.01, repeat_penalty = hf_model.generation_config.repetition_penalty)

        fastllm_logits_list.append(fastllm_logits)
        fastllm_response_list.append(fastllm_response)
    logger.info("释放fastllm模型")
    fastllm_model.release_memory()

    logger.info("使用Transformer进行推理")
    hf_logits_list = []
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
            hf_logits_list.append(hf_last_logits.tolist())
            hf_response_list.append(hf_response)        
    # 结果对比
    coss = []
    for i in range(len(messages_list)):
        if (hf_response_list[i] != fastllm_response_list[i]):
            logger.warning("数据" + str(i) + "的生成结果不同" + 
                           "\n\n输入:\n" + str(messages_list[i]) +
                           "\n\nhf结果:\n" + hf_response_list[i] +
                           "\n\nfastllm结果:\n" + fastllm_response_list[i]);
        else:
            logger.info("数据 " + str(i) + " 的生成结果相同，结果为 \"" +
                        hf_response_list[i][:10] + "...\"")
        coss.append(compute_cosine_similarity(fastllm_logits_list[i], hf_logits_list[i]))
        logger.info("数据 " + str(i) + " 的余弦相似度为" + str(coss[-1]))
    logger.info("平均余弦相似度: " + str(sum(coss) / len(coss)))
