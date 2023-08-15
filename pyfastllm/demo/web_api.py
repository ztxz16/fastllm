# -*- coding: utf-8 -*-
import sys
import platform
import logging
import argparse
from copy import deepcopy
import traceback
from typing import List
sys.path.append('../../build-py')
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
    parser.add_argument("--max_batch_size", type=int, default=32, help="动态batch的最大batch size")
    args = parser.parse_args()
    return args

g_model = None
g_msg_dict = dict()
g_prompt_queue = queue.Queue(maxsize=256)
g_max_batch_size = 32

def save_msg(idx: int, content: bytes):
    global g_msg_dict
    content = content.decode(encoding="utf-8", errors="ignore")
    hash_id_idx = content.rindex("hash_id:") 
    hash_id = content[hash_id_idx+8:]
    content = content[:hash_id_idx].replace("<n>", "\n")
    if hash_id in g_msg_dict.keys():
        g_msg_dict[hash_id].put((idx, content))
    else:
        msg_queue = queue.Queue()
        msg_queue.put((idx, content))
        g_msg_dict[hash_id] = msg_queue

def save_msgs(idx: int, content_list: List[bytes]):
    global g_msg_dict
    for content in content_list:
        content = content.decode(encoding="utf-8", errors="ignore")
        hash_id_idx = content.rindex("hash_id:") 
        hash_id = content[hash_id_idx+8:]
        content = content[:hash_id_idx].replace("<n>", "\n")
        if hash_id in g_msg_dict.keys():
            g_msg_dict[hash_id].put((idx, content))
        else:
            msg_queue = queue.Queue()
            msg_queue.put((idx, content))
            g_msg_dict[hash_id] = msg_queue


def response_stream(prompt: str, config: pyfastllm.GenerationConfig):
    global model
    model.response(prompt, save_msgs, config)

def batch_response_stream(prompt:str, config: pyfastllm.GenerationConfig):
    global g_config
    g_config = config
    g_prompt_queue.put(prompt)


g_running_lock = threading.Lock()
g_running = False
g_config: pyfastllm.GenerationConfig = None
def dynamic_batch_stream_func():
    global g_model, g_running_lock, g_running, g_prompt_queue, g_config, g_msg_dict
    print(f"call dynamic_batch_stream_func: running: {g_running}, prompt queue size: {g_prompt_queue.qsize()}")
    print(f"msg_dict size: {len(g_msg_dict)}")
    batch_size_this = min(g_max_batch_size, g_prompt_queue.qsize())
    if not g_running and batch_size_this>0:
        g_running_lock.acquire()
        g_running = True
        g_running_lock.release()

        batch_this = []
        for _ in range(batch_size_this):
            batch_this.append(g_prompt_queue.get_nowait())
        print(f"batch this: {batch_size_this}, queue len: {g_prompt_queue.qsize()}")

        try:
            if batch_size_this > 0:
                g_model.batch_response(batch_this, save_msgs, g_config)
        except Exception as e:
            hash_id_list = [str(pyfastllm.std_hash(prompt)) for prompt in batch_this]
            rtn_list = [bytes(f"hash_id:{hash_id}", 'utf8') for hash_id in hash_id_list]
            save_msgs(-1, rtn_list)
            traceback.print_exc()  
            print(e)

        g_running_lock.acquire()
        g_running = False
        g_running_lock.release()
        threading.Timer(0, dynamic_batch_stream_func).start()
    else:
        wait_time = float(g_max_batch_size-g_prompt_queue.qsize()-batch_size_this)/g_max_batch_size*1
        threading.Timer(wait_time, dynamic_batch_stream_func).start()
        
            

def chat_stream(prompt: str, config: pyfastllm.GenerationConfig, uid:int=0, time_out=200):
    global g_model, g_msg_dict
    time_stamp = round(time.time() * 1000)
    hash_id = str(pyfastllm.std_hash(f"{prompt}time_stamp:{time_stamp}"))
    thread = threading.Thread(target = batch_response_stream, args = (f"{prompt}time_stamp:{time_stamp}", config))
    thread.start()
    idx = 0
    start = time.time()
    pre_msg = ""
    while idx != -1:
        if hash_id in g_msg_dict.keys():
            msg_queue = g_msg_dict[hash_id]
            if msg_queue.empty():
                time.sleep(0.1)
                continue
            msg_obj = msg_queue.get(block=False)
            idx = msg_obj[0]
            if idx != -1:
                yield msg_obj[1]
            else: # end flag
                del g_msg_dict[hash_id]
                break
            pre_msg = msg_obj[1]
        else:
            if time.time() - start > time_out:
                yield pre_msg + f"\ntime_out: {time.time() - start} senconds"
                break
            time.sleep(0.1)
            continue
    

app = FastAPI()
@app.post("/api/chat_stream")
def api_chat_stream(request: dict):
    #print("request.json(): {}".format(json.loads(request.body(), errors='ignore')))
    data = request
    prompt = data.get("prompt")
    history = data.get("history", [])
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
    round_idx = 0
    history_str = ""
    for (q,a) in history:
        history_str = g_model.make_history(history_str, round_idx, q, a)
        round_idx += 1
    prompt = g_model.make_input(history_str, round_idx, prompt)

    return StreamingResponse(chat_stream(prompt, config), media_type='text/event-stream')


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
    for response in g_model.batch_response(prompts, None, config): 
        retV +=  f"({batch_idx + 1}/{len(prompts)})\n prompt: {prompts[batch_idx]} \n response: {response}\n"
        batch_idx += 1
    return retV

def main(args):
    model_path = args.path
    OLD_API = False
    global g_model, g_max_batch_size
    g_max_batch_size = args.max_batch_size
    if OLD_API:
        g_model = pyfastllm.ChatGLMModel()
        g_model.load_weights(model_path)
        g_model.warmup()
    else:
        global LLM_TYPE 
        LLM_TYPE = pyfastllm.get_llm_type(model_path)
        print(f"llm model: {LLM_TYPE}")
        g_model = pyfastllm.create_llm(model_path)
    threading.Timer(1, dynamic_batch_stream_func).start()
    uvicorn.run(app, host='0.0.0.0', port=8000, workers=1)

if __name__ == "__main__":
    args = args_parser()
    main(args)
