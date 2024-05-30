import argparse
from fastllm_pytools import llm
import readline

def args_parser():
    parser = argparse.ArgumentParser(description = 'fastllm_chat_demo')
    parser.add_argument('-p', '--path', type = str, required = True, default = '', help = '模型文件的路径')
    parser.add_argument('-t', '--threads', type=int, default=4,  help='使用的线程数量')
    parser.add_argument('-l', '--low', action='store_true', help='使用低内存模式')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = args_parser()
    llm.set_cpu_threads(args.threads)
    llm.set_cpu_low_mem(args.low)
    model = llm.model(args.path)
    model.set_save_history(True)

    history = []
    print("输入内容即可进行对话，clear 清空对话历史，stop 终止程序")
    while True:
        query = input("\n用户：")
        if query.strip() == "stop":
            break
        if query.strip() == "clear":
            history = []
            print("输入内容即可进行对话，clear 清空对话历史，stop 终止程序")
            continue
        print("AI:", end = "");
        curResponse = "";
        for response in model.stream_response(query, history = history):
            curResponse += response;
            print(response, flush = True, end = "")
        history.append((query, curResponse))
    model.release_memory()