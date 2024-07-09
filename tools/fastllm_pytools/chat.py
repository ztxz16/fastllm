import argparse
from ftllm import llm
from .util import make_normal_parser
from .util import make_normal_llm_model
import readline

def args_parser():
    parser = make_normal_parser('fastllm_chat')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = args_parser()
    model = make_normal_llm_model(args)
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
        for response in model.stream_response(query, history = history):
            curResponse += response;
            print(response, flush = True, end = "")
        history.append((query, curResponse))
    model.release_memory()