
from ftllm import llm
import sys
import os
import argparse

def make_normal_parser(des: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description = des)
    parser.add_argument('-p', '--path', type = str, required = True, default = '', help = '模型路径，fastllm模型文件或HF模型文件夹')
    parser.add_argument('-t', '--threads', type = int, default = 4,  help = '线程数量')
    parser.add_argument('-l', '--low', action = 'store_true', help = '是否使用低内存模式')
    parser.add_argument('--dtype', type = str, default = "float16", help = '权重类型（读取HF模型时有效）')
    parser.add_argument('--atype', type = str, default = "float32", help = '推理类型，可使用float32或float16')
    parser.add_argument('--cuda_embedding', action = 'store_true', help = '在cuda上进行embedding')
    parser.add_argument('--device', type = str, help = '使用的设备')
    return parser

def parse_args():
    parser = make_normal_parser("fastllm webui")
    parser.add_argument("--port", type = int, default = 8080, help = "网页端口")
    parser.add_argument("--title", type = str, default = "fastllm webui", help = "页面标题")
    return parser.parse_args()

def make_normal_llm_model(args):
    if (args.device and args.device != ""):
        try:
            import ast
            device_map = ast.literal_eval(args.device)
            if (isinstance(device_map, list) or isinstance(device_map, dict)):
                llm.set_device_map(device_map)
            else:
                llm.set_device_map(args.device)
        except:
            llm.set_device_map(args.device)
    llm.set_cpu_threads(args.threads)
    llm.set_cpu_low_mem(args.low)
    if (args.cuda_embedding):
        llm.set_cuda_embedding(True)
    model = llm.model(args.path, dtype = args.dtype, tokenizer_type = "auto")
    model.set_atype(args.atype)
    return model

args = parse_args()
import streamlit as st
from streamlit_chat import message
st.set_page_config(
    page_title = args.title,
    page_icon = ":robot:"
)

@st.cache_resource
def get_model():
    args = parse_args()
    model = make_normal_llm_model(args)
    return model

if "messages" not in st.session_state:
    st.session_state.messages = []

max_new_tokens = st.sidebar.slider("max_new_tokens", 0, 8192, 512, step = 1)
top_p = st.sidebar.slider("top_p", 0.0, 1.0, 0.8, step = 0.01)
top_k = st.sidebar.slider("top_k", 1, 100, 1, step = 1)
temperature = st.sidebar.slider("temperature", 0.0, 2.0, 1.0, step = 0.01)
repeat_penalty = st.sidebar.slider("repeat_penalty", 1.0, 10.0, 1.0, step = 0.05)

buttonClean = st.sidebar.button("清理会话历史", key="clean")
if buttonClean:
    st.session_state.messages = []
    st.rerun()

for i, (prompt, response) in enumerate(st.session_state.messages):
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        st.markdown(response)

if prompt := st.chat_input("请开始对话"):
    model = get_model()
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        for chunk in model.stream_response(prompt, 
                                           st.session_state.messages, 
                                           max_length = max_new_tokens,
                                           top_k = top_k,
                                           top_p = top_p,
                                           temperature = temperature,
                                           repeat_penalty = repeat_penalty,
                                           one_by_one = True):
            full_response += chunk
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    st.session_state.messages.append((prompt, full_response))
