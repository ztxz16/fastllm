import sys
from transformers import AutoTokenizer, AutoModel

import fastllm

def export():
    model_path = '/public/Models/chatglm-6b' # 仅支持fp32模型加载
    export_path = "chatglm-6b-fp32.flm"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True).float()
    model = model.eval()

    fastllm.utils.convert(model=model, tokenizer=tokenizer, output_path=export_path,  verbose=True)

def response(model, prompt_input:str, stream_output:bool=False):
    gmask_token_id = 130001
    bos_token_id = 130004

    input_ids = model.weight.tokenizer.encode(prompt_input)
    input_ids = input_ids.to_list()
    input_ids.extend([gmask_token_id, bos_token_id])
    input_ids = [int(v) for v in input_ids]

    handle = model.launch_response(input_ids)
    continue_token = True

    ret_byte = b""
    ret_str = ""
    
    while continue_token:
        resp_token = model.fetch_response(handle)
        continue_token = (resp_token != -1)

        content = model.weight.tokenizer.decode_byte([resp_token])
        ret_byte += content
        ret_str = ret_byte.decode(errors='ignore')

        if stream_output:
            yield ret_str

    return ret_str

def infer():
    model_path = "chatglm-6b-fp32.flm"
    model = fastllm.create_llm(model_path)

    prompt = "你好"
    outputs = response(model, prompt_input=prompt, stream_output=True)
    for output in outputs:
        print('\r LLM:' + output, end='', flush=True)

    print()


if __name__ == "__main__":
    # export()
    infer()