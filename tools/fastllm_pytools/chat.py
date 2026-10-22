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

    generation_config = dict(model.default_generation_config)
    # Match the API server: native sampling needs a positive candidate count.
    do_sample = generation_config["temperature"] > 0
    if not do_sample:
        generation_config.update(temperature=1.0, top_k=1, top_p=1.0)
    elif generation_config["top_k"] <= 0:
        generation_config["top_k"] = model.get_vocab_size()

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
        for response in model.stream_response(query, history = history, do_sample = do_sample,
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