import json
import requests
import sys



if __name__ == '__main__':
    url = 'http://127.0.0.1:8000/api/chat_stream'
    prompt='请用emoji写一首诗赞美世界'
    json_obj = {"uid":0, "token":"xxxxxxxxxxxxxxxxx","history": "", "prompt": prompt , "max_length": None, "top_p": None,"temperature": None}
    response = requests.post(url, json=json_obj,  stream = True)
    try:
        pre_msg = ""
        print("response:")
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
