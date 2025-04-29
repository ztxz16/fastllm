import sys
import os
import argparse
from util import make_normal_parser

def parse_args():
    parser = make_normal_parser("fastllm webui")
    parser.add_argument("--port", type = int, default = 8080, help = "API server port")
    parser.add_argument("--title", type = str, default = "fastllm webui", help = "页面标题")
    parser.add_argument("--max_token", type = int, default = 4096, help = "输出最大token数")
    parser.add_argument("--think", type = str, default = "false", help = "if <think> lost")
    return parser.parse_args()

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
    from util import make_normal_llm_model
    model = make_normal_llm_model(args)
    model.set_verbose(True)
    return model

if "messages" not in st.session_state:
    st.session_state.messages = []
if "images" not in st.session_state:
    st.session_state.images = []

system_prompt = st.sidebar.text_input("system_prompt", "")
max_new_tokens = st.sidebar.slider("max_new_tokens", 0, args.max_token, args.max_token, step = 1)
top_p = st.sidebar.slider("top_p", 0.0, 1.0, 0.8, step = 0.01)
top_k = st.sidebar.slider("top_k", 1, 50, 1, step = 1)
temperature = st.sidebar.slider("temperature", 0.0, 10.0, 1.0, step = 0.1)
repeat_penalty = st.sidebar.slider("repeat_penalty", 1.0, 10.0, 1.0, step = 0.05)

buttonClean = st.sidebar.button("清理会话历史", key="clean")
if buttonClean:
    st.session_state.messages = []
    st.rerun()

# 添加文件输入部件
if (uploaded_file := st.file_uploader("上传图片", type=["jpg", "jpeg", "png"])) is not None:
    from PIL import Image
    import io
    import base64

    image = Image.open(uploaded_file).convert('RGB')
    st.session_state.images = [image]
    
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    # 创建一个包含图片的 HTML 元素，并设置样式
    image_html = f"""
    <div style="display: flex; justify-content: flex-end;">
        <img src="data:image/png;base64,{img_str}" style="max-width: 300px; max-height: 300px;">
    </div>
    """
    
    # 显示图片
    st.markdown(image_html, unsafe_allow_html=True)

model = get_model()

for i, (prompt, response) in enumerate(st.session_state.messages):
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        st.markdown(response)
    
if prompt := st.chat_input("请开始对话"):
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        messages = []
        if system_prompt != "":
            messages.append({"role": "system", "content": system_prompt})
        for his in st.session_state.messages:
            messages.append({"role": "user", "content": his[0]})
            messages.append({"role": "assistant", "content": his[1]})
        messages.append({"role": "user", "content": prompt})

        if (len(st.session_state.images) > 0):
            handle = model.launch_stream_response(messages,
                        max_length = max_new_tokens, do_sample = True,
                        top_p = top_p, top_k = top_k, temperature = temperature,
                        repeat_penalty = repeat_penalty, one_by_one = True, images = st.session_state.images)
            for chunk in model.stream_response_handle(handle):
                full_response += chunk
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        else:
            handle = model.launch_stream_response(messages,
                                           max_length = max_new_tokens,
                                           top_k = top_k,
                                           top_p = top_p,
                                           temperature = temperature,
                                           repeat_penalty = repeat_penalty,
                                           one_by_one = True)
            if ((args.think.lower() != "false")):
                full_response += "<think>"
                message_placeholder.markdown(full_response + "▌")
            for chunk in model.stream_response_handle(handle):
                full_response += chunk
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
    st.session_state.messages.append((prompt, full_response))
