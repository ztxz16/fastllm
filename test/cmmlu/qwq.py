import os
import torch
import numpy as np
import argparse
from CMMLU.src.mp_utils import choices, format_example, gen_prompt, softmax, run_eval
from transformers import AutoModel, AutoTokenizer
import threading
import time

def chat(model, tokenizer, prompt, output_list, idx):
    prompt = "请只回答一个字母。" + prompt;
    print(prompt);
    pred, history = model.chat(tokenizer, prompt, history=[], max_length = 5);
    if len(pred) < 1 or (pred[0] not in choices):
        pred, history = model.chat(tokenizer, prompt, history=[], max_length = 1000);
    output_list[idx] = pred;

def extract_result(pred):
    for i in pred[::-1]:
        if i in choices:
            return i
    return "X"

def eval_chat_multithread(model, tokenizer, subject, dev_df, test_df, num_few_shot, max_length, cot):
    cors = []
    all_preds = []
    answers = choices[: test_df.shape[1] - 2]

    batch_num = 1;
    output_list = ["" for i in range(test_df.shape[0])];
    ths = [None for i in range(test_df.shape[0])];

    for j in range(0, test_df.shape[0], batch_num):
        cur_len = min(test_df.shape[0] - j, batch_num);
        for i in range(j, j + cur_len):
            prompt_end = format_example(test_df, i, subject, include_answer=False, cot=cot)
            prompt = gen_prompt(dev_df=dev_df,
                                subject=subject,
                                prompt_end=prompt_end,
                                num_few_shot=num_few_shot,
                                tokenizer=tokenizer,
                                max_length=max_length,
                                cot=cot)
            ths[i] = threading.Thread(target = chat, args=(model, tokenizer, prompt, output_list, i));
            ths[i].start();
        for i in range(j, j + cur_len):
            ths[i].join();
            pred = output_list[i];
            label = test_df.iloc[i, test_df.shape[1] - 1]
            c = extract_result(pred);
            if c == label:
                cors.append(True);
            else:
                cors.append(False);
            all_preds.append(pred.replace("\n", ""))
            print(i, test_df.shape[0], np.mean(cors))

    acc = np.mean(cors)
    print("Average accuracy {:.3f} - {}".format(acc, subject))
    print("{} results, {} inappropriate formated answers.".format(len(cors), len(all_preds)-len(cors)))
    return acc, all_preds, None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="")
    parser.add_argument("--lora_weights", type=str, default="")
    parser.add_argument("--data_dir", type=str, default="./CMMLU/data")
    parser.add_argument("--save_dir", type=str, default="./results/ChatGLM2-6B")
    parser.add_argument("--num_few_shot", type=int, default=0)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--cot", action='store_true')
    args = parser.parse_args()

    from ftllm import llm;
    model = llm.model(args.model_name_or_path, tokenizer_type = "auto", dtype = "int8")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True,)
    run_eval(model, tokenizer, eval_chat_multithread, args)
