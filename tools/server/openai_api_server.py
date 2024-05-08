import argparse
import fastapi
import logging
import sys
import uvicorn
from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from protocal.openai_protocol import *
from fastllm_completion import FastLLmCompletion

def parse_args():
    parser = argparse.ArgumentParser(description="OpenAI-compatible API server")
    parser.add_argument("--model_name", type = str, help = "部署的模型名称, 调用api时会进行名称核验", required=True)
    parser.add_argument("--host", type = str, default="0.0.0.0", help = "API Server host")
    parser.add_argument("--port", type = int, default = 8080, help = "API Server Port")
    parser.add_argument('-p', '--path', type = str, required = True, default = '', help = 'fastllm model path(end with .flm)')
    parser.add_argument('-t', '--threads', type=int, default=4,  help='fastllm model 使用的线程数量')
    parser.add_argument('-l', '--low', action='store_true', help='fastllm 是否使用低内存模式')
    return parser.parse_args()


app = fastapi.FastAPI()
# 设置允许的请求来源, 生产环境请做对应变更
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

fastllm_completion:FastLLmCompletion

@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest,
                                 raw_request: Request):
    generator = await fastllm_completion.create_chat_completion(
        request, raw_request)
    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(),
                            status_code=generator.code)
    if request.stream:
        return StreamingResponse(content=generator,
                                 media_type="text/event-stream")
    else:
        assert isinstance(generator, ChatCompletionResponse)
        return JSONResponse(content=generator.model_dump())

def init_logging(log_level = logging.INFO, log_file:str = None):
    logging_format = '%(asctime)s %(process)d %(filename)s[line:%(lineno)d] %(levelname)s: %(message)s'
    root = logging.getLogger()
    root.setLevel(log_level)
    if log_file is not None:
        logging.basicConfig(level=log_level, filemode='a', filename=log_file, format=logging_format)
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(logging.Formatter(logging_format))
    root.addHandler(stdout_handler)

    
if __name__ == "__main__":
    init_logging()
    args = parse_args()
    logging.info(args)
    fastllm_completion = FastLLmCompletion(model_name = args.model_name,
                                           model_path = args.path,
                                           low_mem_mode = args.low,
                                           cpu_thds = args.threads)
    uvicorn.run(app, host=args.host, port=args.port)
