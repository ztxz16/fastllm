from ftllm import llm
from qwen2 import Qwen2Model
import os

root_path = "/mnt/hfmodels/"
model_path = os.path.join(root_path, "Qwen/Qwen2-7B-Instruct")

model = llm.model(model_path, graph = Qwen2Model)
prompt = "北京有什么景点？"
messages = [
    {"role": "system", "content": "你是一个爱说英文的人工智能，不管我跟你说什么语言，你都会用英文回复我"},
    {"role": "user", "content": prompt}
]
for response in model.stream_response(messages, one_by_one = True):
    print(response, flush = True, end = "")