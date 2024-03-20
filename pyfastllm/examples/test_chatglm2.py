import fastllm
from fastllm.hub.chatglm2 import ChatGLM2, ChatGLMConfig
import fastllm.functions as ops

from transformers import AutoTokenizer

def load_weights():
    file = "/home/pan/Public/Models/models-flm/chatglm2-6b.flm"
    state_dict = ops.load(file)
    return state_dict

def run():
    # fastllm.set_device_map({"cuda:0": 28})
    state_dict = load_weights()
    cfg = ChatGLMConfig()
    model = ChatGLM2(cfg)
    model.set_weights(state_dict)
    print("model loaded!!!")

    model_path = "/home/pan/Public/Models/models-hf/chatglm2-6b"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    # model.warmup()
    res = ""
    for output in model.stream_chat(query="飞机为什么会飞", tokenizer=tokenizer):
        res = output

    print("最终问答", res)

if __name__ == "__main__":
    run()

