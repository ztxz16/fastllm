import argparse
from .util import make_normal_parser
import readline

def args_parser():
    parser = make_normal_parser('fastllm_chat')
    args = parser.parse_args()
    return args

def fastllm_chat(args):
    from .util import make_normal_llm_model
    model = make_normal_llm_model(args)

    generation_config = {
        'repetition_penalty': 1.0,
        'top_p': 0.8,
        'top_k': 1,
        'temperature': 1.0
    }
    import os
    import json
    if (os.path.exists(os.path.join(args.path, "generation_config.json"))):
        with open(os.path.join(args.path, "generation_config.json"), "r", encoding="utf-8") as file:
            config = json.load(file)
            if ('do_sample' in config and config['do_sample']):
                for it in ["repetition_penalty", "top_p", "top_k", "temperature"]:
                    if (it in config):
                        generation_config[it] = config[it];

    hint = "输入内容开始对话\n'clear'清空记录\n'stop'终止程序."
    history = []

    print(hint)
    while True:
        query = input("\nUser：")
        if query.strip() == "stop":
            break
        if query.strip() == "clear":
            history = []
            print(hint)
            continue
        print("AI:", end = "");
        curResponse = "";
        for response in model.stream_response(query, history = history, 
                                              repeat_penalty = generation_config["repetition_penalty"],
                                              top_p = generation_config["top_p"],
                                              top_k = generation_config["top_k"],
                                              temperature = generation_config["temperature"]):
            curResponse += response;
            print(response, flush = True, end = "")
        history.append((query, curResponse))
    model.release_memory()

if __name__ == "__main__":
    args = args_parser()
    fastllm_chat(args)