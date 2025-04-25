import argparse
import fastapi
import logging
import sys
import uvicorn
from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from .openai_server.protocal.openai_protocol import *
from .openai_server.fastllm_completion import FastLLmCompletion
from .openai_server.fastllm_embed import FastLLmEmbed
from .openai_server.fastllm_reranker import FastLLmReranker
from .openai_server.fastllm_model import FastLLmModel
from .util import make_normal_parser
from .util import add_server_args
global fastllm_completion
global fastllm_embed
global fastllm_reranker
global fastllm_model

def parse_args():
    parser = make_normal_parser("OpenAI-compatible API server")
    add_server_args(parser)
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
fastllm_embed:FastLLmEmbed
fastllm_model:FastLLmModel

@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest,
                                 raw_request: Request):
    generator = await fastllm_completion.create_chat_completion(
        request, raw_request)
    if isinstance(generator, ErrorResponse):
        return JSONResponse(content = generator.model_dump(),
                            status_code = generator.code)
    if request.stream:
        return StreamingResponse(content = generator[0],
                                 background = generator[1], 
                                 media_type = "text/event-stream")
    else:
        assert isinstance(generator, ChatCompletionResponse)
        return JSONResponse(content = generator.model_dump())

@app.post("/v1/embed")
async def create_embed(request: EmbedRequest,
                       raw_request: Request):
    embedding = fastllm_embed.embedding_sentence(request, raw_request)
    return JSONResponse(embedding)

@app.post("/v1/rerank")
async def create_rerank(request: RerankRequest,
                       raw_request: Request):
    print(request)
    scores = fastllm_reranker.rerank(request, raw_request)    
    return JSONResponse(scores)


@app.get("/v1/models")
async def list_models():
    model_response = fastllm_model.response
    return JSONResponse(content = model_response)

def init_logging(log_level = logging.INFO, log_file:str = None):
    logging_format = '%(asctime)s %(process)d %(filename)s[line:%(lineno)d] %(levelname)s: %(message)s'
    root = logging.getLogger()
    root.setLevel(log_level)
    if log_file is not None:
        logging.basicConfig(level=log_level, filemode='a', filename=log_file, format=logging_format)
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(logging.Formatter(logging_format))
    root.addHandler(stdout_handler)

def fastllm_server(args):
    if args.api_key:
        @app.middleware("http")
        async def authentication(request: Request, call_next):
            print("auth")
            if request.method == "OPTIONS":
                return await call_next(request)
            url_path = request.url.path            
            if not url_path.startswith("/v1"):
                return await call_next(request)
            if request.headers.get("Authorization") != "Bearer " + args.api_key:
                return JSONResponse(content={"error": "Unauthorized"},
                                    status_code=401)
            return await call_next(request)
        
    global fastllm_completion
    global fastllm_embed
    global fastllm_reranker
    global fastllm_model
    init_logging()
    logging.info(args)
    from .util import make_normal_llm_model
    model = make_normal_llm_model(args)
    model.set_verbose(True)
    if (args.model_name is None or args.model_name == ''):
        args.model_name = args.path
        if (args.model_name is None or args.model_name == ''):
            args.model_name = args.model
    fastllm_completion = FastLLmCompletion(model_name = args.model_name, model = model, think = args.think, hide_input = args.hide_input)
    fastllm_embed = FastLLmEmbed(model_name = args.model_name, model = model)
    fastllm_reranker = FastLLmReranker(model_name = args.model_name, model = model)
    fastllm_model = FastLLmModel(model_name = args.model_name)
    uvicorn.run(app, host = args.host, port = args.port)

if __name__ == "__main__":
    args = parse_args()
    fastllm_server(args)
    