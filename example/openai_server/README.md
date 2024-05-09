# OpenAI Compatible API Server
## 介绍
这是一个兼容OpenAI API 的HTTP Server, 支持加载fastllm的flm模型, 暂时不支持https方式访问
> 这个实现参考了vllm v0.4.1 中OpenAI-compatible API server的实现, 在这个基础上进行了简化


## 目前支持的接口
* Open AI 接口官方说明: https://platform.openai.com/docs/api-reference/

| 类型 | 接口名称                | method              | 目前明确不支持的选项| 官方说明和样例                                                 |
| :--- | :--------------------- | :------------------ | :--------------------------------------------------------- |:--------------------------------------------------------- |
| Chat | Create chat completion | v1/chat/completions | (n, presence_penalty, tools, functions, logprobs, seed, logit_bias) | https://platform.openai.com/docs/api-reference/chat/create |


> 考虑到Completions接口已经被标记为Legacy接口，因此不实现该接口

## 依赖
以下依赖在python 3.12.2上没有问题
1. 需要先安装fastllm_pytools工具包
2. 需要安装以下依赖
```bash
cd example/openai_server
pip install -r requirements.txt
```

## 使用方法 && 样例
* server启动命令
```bash
cd example/openai_server
python openai_api_server.py --model_name "model_name" -p "path_to_your_flm_model"
# eg : python openai_api_server.py --model_name "chat-glm2-6b-int4" -p "./chatglm2-6b-int4.flm"
```

* client测试命令
```bash
# client 测试
# 测试命令
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer something" \
  -d '{
    "model": "chat-glm2-6b-int4",
    "messages": [
      {
        "role": "system",
        "content": "You are a helpful assistant."
      },
      {
        "role": "user",
        "content": "Hello!"
  }'] }
# 响应结果
{"id":"fastllm-chat-glm2-6b-int4-e4fd6bea564548f6ae95f6327218616d","object":"chat.completion","created":1715150460,"model":"chat-glm2-6b-int4","choices":[{"index":0,"message":{"role":"assistant","content":" Hello! How can I assist you today?"},"finish_reason":"stop"}],"usage":{"prompt_tokens":0,"total_tokens":0,"completion_tokens":0}}
```


