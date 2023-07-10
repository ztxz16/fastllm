import json
import requests
import sys



if __name__ == '__main__':
    #stream api
    url = 'http://127.0.0.1:8000/api/chat_stream'
    prompt='请用emoji写一首短诗赞美世界'
    json_obj = {"uid":0, "token":"xxxxxxxxxxxxxxxxx","history": "", "prompt": prompt , "max_length": 1024, "top_p": None,"temperature": 0.95, "top_k":1, "repeat_penalty": 1.}
    response = requests.post(url, json=json_obj,  stream = True)
    try:
        pre_msg = ""
        print("stream response:")
        for chunk in response.iter_content(chunk_size=1024*1024):
            msg = chunk.decode(errors='replace')
            if len(msg) > len(pre_msg) and msg[-1] == '\n':
                content = msg[len(pre_msg):]
                pre_msg = msg
            else:
                continue
            print(f"{content}", end="")
            sys.stdout.flush()
        content = msg[len(pre_msg):]
        print(f"{content}", end="")
        print()
    except Exception as ex:
        print(ex)
    
    #batch api
    url = 'http://127.0.0.1:8000/api/batch_chat'
    prompts = ["Hi", "你好", "用emoji表达高兴", "こんにちは"]
    json_obj = {"uid":0, "token":"xxxxxxxxxxxxxxxxx","history": "", "prompts": prompts , "max_length": 100, "top_p": None,"temperature": 0.7, "top_k":1, "repeat_penalty":2.}
    response = requests.post(url, json=json_obj, stream = True)
    print("batch response: {} text:\n{}".format(response, response.text.replace('\\n', '\n')))
