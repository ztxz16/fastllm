import streamlit as st
from streamlit_chat import message
from fastllm_pytools import llm
import sys

st.set_page_config(
    page_title="fastllm web demo",
    page_icon=":robot:"
)

@st.cache_resource
def get_model():
    model = llm.model(sys.argv[1]);
    return model;

def predict(input, history = None):
    model = get_model()
    if history is None:
        history = []
    with container:
        if len(history) > 0:
            for i, (query, response) in enumerate(history):
                message(query, avatar_style = "big-smile", key = str(i) + "_user")
                message(response, avatar_style = "bottts", key = str(i))
        message(input, avatar_style = "big-smile", key = str(len(history)) + "_user")
        st.write("AI正在回复:")
        with st.empty():
            for response in model.stream_response(input, history, one_by_one = False):
                st.write(response)
            history.append((input, response));
    return history

container = st.container()
prompt_text = st.text_area(label="用户命令输入", height = 100, placeholder = "请在这儿输入您的命令")
if 'state' not in st.session_state:
    st.session_state['state'] = []
if st.button("发送", key = "predict"):
    with st.spinner("AI正在思考，请稍等........"):
        st.session_state["state"] = predict(prompt_text, st.session_state["state"])
