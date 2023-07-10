# -*- coding: utf-8 -*-
import sys
import platform
import logging
import argparse
sys.path.append('./build-py')
import pyfastllm # 或fastllm
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import threading, queue, uvicorn, json, time

logging.info(f"python gcc version:{platform.python_compiler()}")


def args_parser():
    parser = argparse.ArgumentParser(description='pyfastllm')
    parser.add_argument('-m', '--model', type=int, required=False, default=0, help='模型类型，默认为0, 可以设置为0(chatglm),1(moss),2(vicuna),3(baichuan)')
    parser.add_argument('-p', '--path', type=str, required=True, default='', help='模型文件的路径')
    parser.add_argument('-t', '--threads', type=int, default=4,  help='使用的线程数量')
    parser.add_argument('-l', '--low', action='store_true', help='使用低内存模式')
    args = parser.parse_args()
    return args

model = None
msg_queue = queue.Queue()

def save_msg(idx: int, content: bytes):
    content = content.decode(encoding="utf-8", errors="replace")
    msg_queue.put((idx, content))

def response_stream(prompt: str, config: pyfastllm.GenerationConfig):
    global model
    model.response(prompt, save_msg, config)

def chat_stream(prompt: str, config: pyfastllm.GenerationConfig):
    global model
    thread = threading.Thread(target = response_stream, args = (prompt, config))
    thread.start()
    idx = 0
    while idx != -1:
        if msg_queue.empty():
            time.sleep(0.1)
            continue
        msg_obj = msg_queue.get(block=False)
        idx = msg_obj[0]
        yield msg_obj[1]

    

app = FastAPI()
@app.post("/api/chat_stream")
async def api_chat_stream(request: Request):
    #print("request.json(): {}".format(json.loads(request.body(), errors='ignore')))
    data = await request.json()
    prompt = data.get("prompt")
    history = data.get("history")
    config = pyfastllm.GenerationConfig()
    if data.get("max_length") is not None:
        config.max_length = max_length
    if data.get("top_k") is not None:
        config.top_k = top_k
    if data.get("top_p") is not None:
        config.top_p = top_p
    return StreamingResponse(chat_stream(history + prompt, config), media_type='text/event-stream')



def main(args):
    model_path = args.path
    OLD_API = False
    global model
    if OLD_API:
        model = pyfastllm.ChatGLMModel()
        model.load_weights(model_path)
        model.warmup()
    else:
        global LLM_TYPE 
        LLM_TYPE = pyfastllm.get_llm_type(model_path)
        print(f"llm model: {LLM_TYPE}")
        model = pyfastllm.create_llm(model_path)
    uvicorn.run(app, host='0.0.0.0', port=8000, workers=1)

if __name__ == "__main__":
    args = args_parser()
    main(args)
