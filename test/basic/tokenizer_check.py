from config import default_messages_list

import argparse
import logging
import os
from transformers import AutoTokenizer
from ftllm import llm

def args_parser():
    parser = argparse.ArgumentParser(description = 'fastllm_test')
    parser.add_argument('--model', type = str, required = True, default = '', help = '模型文件目录')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = args_parser()
    messages_list = default_messages_list

    logger = logging.getLogger()
    logging.basicConfig(level = logging.INFO, format = '%(asctime)s - %(levelname)s - %(message)s')

    model_path = args.model
        
    logger.info("开始测试模型 " + model_path)
    logger.info("正在用Transformer读取Tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    logger.info("读取成功")
    logger.info("正在用Fastllm读取Tokenizer")
    fastllm_tokenizer = llm.tokenizer(model_path)
    logger.info("读取成功")

    check_succ = True
    for messages in messages_list:
        hf_text = tokenizer.apply_chat_template (messages, tokenize = False, add_generation_prompt = True)
        fastllm_text = tokenizer.apply_chat_template (messages, tokenize = False, add_generation_prompt = True)
        if (hf_text != fastllm_text):
            check_succ = False
            logger.error("apply_chat_template结果比对错误" + 
                            "\n\n输入:\n" + str(messages) +
                            "\n\nhf结果:\n" + hf_text +
                            "\nfastllm结果:\n" + fastllm_text);
            break
        hf_tokens = tokenizer.encode(hf_text)
        fastllm_tokens = fastllm_tokenizer.encode(fastllm_text)
        if (hf_tokens != fastllm_tokens):
            check_succ = False
            logger.error("encode结果比对错误" + 
                            "\n\n输入:\n" + hf_text +
                            "\n\nhf结果:\n" + str(hf_tokens) +
                            "\nfastllm结果:\n" + str(fastllm_tokens));
            break

        if check_succ:
            logger.info("分词结果比对正确")
