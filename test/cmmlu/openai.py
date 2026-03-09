import os
import numpy as np
import argparse
import threading
import pandas as pd
from CMMLU.src.mp_utils import choices, format_example, gen_prompt, softmax, run_eval, extract_choice, get_results

import requests

def chat(model, tok, prompt, output_list, idx):
    #prompt = "这是MMLU测试，请看下面的问题！" * 100 + prompt;
    url = 'http://10.10.21.21:1616/v1/chat/completions'
    json_payload = {
        "model": "ds",
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "top_k": 1,
        "temperature": 1, 
        "max_tokens" : 512,
        "chat_template_kwargs": {"enable_thinking": False}
    }
    response = requests.post(url, json = json_payload)
    response_json = response.json()
    output_list[idx] = response_json["choices"][0]["message"]["content"]    

def eval_chat_multithread(model, tokenizer, subject, dev_df, test_df, num_few_shot, max_length, cot):
    cors = []
    all_preds = []
    answers = choices[: test_df.shape[1] - 2]

    batch_num = 512
    output_list = ["" for i in range(test_df.shape[0])]
    ths = [None for i in range(test_df.shape[0])]

    for j in range(0, test_df.shape[0], batch_num):
        cur_len = min(test_df.shape[0] - j, batch_num)
        for i in range(j, j + cur_len):
            prompt_end = format_example(test_df, i, subject, include_answer=False, cot=cot)
            prompt = gen_prompt(dev_df=dev_df,
                                subject=subject,
                                prompt_end=prompt_end,
                                num_few_shot=num_few_shot,
                                tokenizer=tokenizer,
                                max_length=max_length,
                                cot=cot)
            ths[i] = threading.Thread(target = chat, args=(model, tokenizer, prompt, output_list, i))
            ths[i].start()
        for i in range(j, j + cur_len):
            ths[i].join()
            if (len(output_list[i]) == 0):
                pred = choices[0]
            else:
                pred = extract_choice(output_list[i])
            label = test_df.iloc[i, test_df.shape[1] - 1]
            if pred and pred[0] in choices:
                cors.append(pred[0] == label)
            all_preds.append(pred.replace("\n", ""))
            print(i, test_df.shape[0], np.mean(cors))
    acc = np.mean(cors)
    print("Average accuracy {:.3f} - {}".format(acc, subject))
    print("{} results, {} inappropriate formated answers.".format(len(cors), len(all_preds)-len(cors)))
    return acc, all_preds, None


def run_eval_send_all(args):
    """将所有 subject 的所有请求一次性全部发送，再统一收集结果"""
    subjects = sorted([f.split(".csv")[0] for f in os.listdir(os.path.join(args.data_dir, "test/"))])
    save_dir = f"{args.save_dir}_{args.num_few_shot}_shot"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    cot = args.cot if 'cot' in args else False

    # 第一步：收集所有 subject 的所有请求
    all_tasks = []  # (subject_idx, subject, row_idx, prompt, test_df)
    subject_data = []  # (subject, test_df, start_global_idx)

    for subject in subjects:
        out_file = os.path.join(save_dir, f"results_{subject}.csv")
        if os.path.exists(out_file):
            continue
        dev_df = pd.read_csv(os.path.join(args.data_dir, "dev", subject + ".csv"), header=0, index_col=0)
        test_df = pd.read_csv(os.path.join(args.data_dir, "test", subject + ".csv"), header=0, index_col=0)

        start_idx = len(all_tasks)
        for i in range(test_df.shape[0]):
            prompt_end = format_example(test_df, i, subject, include_answer=False, cot=cot)
            prompt = gen_prompt(dev_df=dev_df,
                                subject=subject,
                                prompt_end=prompt_end,
                                num_few_shot=args.num_few_shot,
                                tokenizer=None,
                                max_length=args.max_length,
                                cot=cot)
            all_tasks.append(prompt)
        subject_data.append((subject, test_df, start_idx, len(all_tasks)))

    total = len(all_tasks)
    if total == 0:
        print("所有 subject 均已完成，无需重新评测。")
        get_results(save_dir)
        return

    print(f"共收集 {total} 个请求，涉及 {len(subject_data)} 个 subject，开始一次性发送...")

    # 第二步：一次性发送所有请求
    output_list = ["" for _ in range(total)]
    ths = [None for _ in range(total)]
    for i in range(total):
        ths[i] = threading.Thread(target=chat, args=(None, None, all_tasks[i], output_list, i))
        ths[i].start()

    for i in range(total):
        ths[i].join()
        if (i + 1) % 500 == 0 or i == total - 1:
            print(f"已完成 {i + 1}/{total} 个请求")

    # 第三步：按 subject 分组处理结果并保存
    for subject, test_df, start_idx, end_idx in subject_data:
        cors = []
        all_preds = []
        for i in range(start_idx, end_idx):
            row = i - start_idx
            if len(output_list[i]) == 0:
                pred = choices[0]
            else:
                pred = extract_choice(output_list[i])
            label = test_df.iloc[row, test_df.shape[1] - 1]
            if pred and pred[0] in choices:
                cors.append(pred[0] == label)
            all_preds.append(pred.replace("\n", ""))

        acc = np.mean(cors)
        print("Average accuracy {:.3f} - {}".format(acc, subject))
        print("{} results, {} inappropriate formated answers.".format(len(cors), len(all_preds) - len(cors)))

        test_df['prediction'] = all_preds
        out_file = os.path.join(save_dir, f"results_{subject}.csv")
        test_df.to_csv(out_file, header=None)

    get_results(save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="")
    parser.add_argument("--lora_weights", type=str, default="")
    parser.add_argument("--data_dir", type=str, default="./CMMLU/data")
    parser.add_argument("--save_dir", type=str, default="./results/not_specified")
    parser.add_argument("--num_few_shot", type=int, default=0)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--load_in_8bit", action='store_true')
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--with_conf", action='store_true')
    parser.add_argument("--cot", action='store_true')
    parser.add_argument("--send_all", action='store_true', help="一次性发送所有 subject 的所有请求，而不是逐 subject 分批发送")
    args = parser.parse_args()

    if args.send_all:
        run_eval_send_all(args)
    else:
        run_eval(None, None, eval_chat_multithread, args)