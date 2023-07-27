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
msg_dict = dict()

def save_msg(idx: int, content: bytes):
    global msg_dict
    content = content.decode(encoding="utf-8", errors="ignore")
    hash_id_idx = content.rindex("hash_id:") 
    hash_id = content[hash_id_idx+8:]
    content = content[:hash_id_idx].replace("<n>", "\n")
    if hash_id in msg_dict.keys():
        msg_dict[hash_id].put((idx, content))
    else:
        msg_queue = queue.Queue()
        msg_queue.put((idx, content))
        msg_dict[hash_id] = msg_queue

def response_stream(prompt: str, config: pyfastllm.GenerationConfig):
    global model
    model.response(prompt, save_msg, config)

def chat_stream(prompt: str, config: pyfastllm.GenerationConfig, uid:int=0, time_out=200):
    global model, msg_dict
    time_stamp = round(time.time() * 1000)
    hash_id = str(pyfastllm.std_hash(f"{prompt}time_stamp:{time_stamp}"))
    thread = threading.Thread(target = response_stream, args = (f"{prompt}time_stamp:{time_stamp}", config))
    thread.start()
    idx = 0
    start = time.time()
    pre_msg = ""
    while idx != -1:
        if hash_id in msg_dict.keys():
            msg_queue = msg_dict[hash_id]
            if msg_queue.empty():
                time.sleep(0)
                continue
            msg_obj = msg_queue.get(block=False)
            idx = msg_obj[0]
            yield msg_obj[1]
            pre_msg = msg_obj[1]
        else:
            if time.time() - start > time_out:
                yield pre_msg + f"\ntime_out: {time.time() - start} senconds"
                break
            time.sleep(0)
            continue

    

app = FastAPI()
@app.post("/api/chat_stream")
async def api_chat_stream(request: Request):
    #print("request.json(): {}".format(json.loads(request.body(), errors='ignore')))
    data = await request.json()
    prompt = data.get("prompt")
    history = data.get("history")
    if history is None:
        history = ""
    round_cnt = data.get("round_cnt")
    config = pyfastllm.GenerationConfig()
    if data.get("max_length") is not None:
        config.max_length = data.get("max_length") 
    if data.get("top_k") is not None:
        config.top_k = data.get("top_k")
    if data.get("top_p") is not None:
        config.top_p = data.get("top_p")
    if data.get("temperature") is not None:
        config.temperature = data.get("temperature")
    if data.get("repeat_penalty") is not None:
        config.repeat_penalty = data.get("repeat_penalty")
    uid = None
    if data.get("uid") is not None:
        uid = data.get("uid")
    config.enable_hash_id = True
    print(f"prompt:{prompt}")
    return StreamingResponse(chat_stream(history + prompt, config), media_type='text/event-stream')


@app.post("/api/batch_chat")
async def api_batch_chat(request: Request):
    data = await request.json()
    prompts = data.get("prompts")
    print(f"{prompts}  type:{type(prompts)}")
    if prompts is None:
        return "prompts should be list[str]"
    history = data.get("history")
    if history is None:
        history = ""
    config = pyfastllm.GenerationConfig()
    if data.get("max_length") is not None:
        config.max_length = data.get("max_length") 
    if data.get("top_k") is not None:
        config.top_k = data.get("top_k")
    if data.get("top_p") is not None:
        config.top_p = data.get("top_p")
    if data.get("temperature") is not None:
        config.temperature = data.get("temperature")
    if data.get("repeat_penalty") is not None:
        config.repeat_penalty = data.get("repeat_penalty")
    uid = None
    if data.get("uid") is not None:
        uid = data.get("uid")
    retV = ""
    batch_idx = 0
    for response in model.batch_response(prompts, None, config): 
        retV +=  f"({batch_idx + 1}/{len(prompts)})\n prompt: {prompts[batch_idx]} \n response: {response}\n"
        batch_idx += 1
    return retV

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
