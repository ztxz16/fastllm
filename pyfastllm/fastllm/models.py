#!encoding=utf8
import os
import tempfile
from typing import List, Tuple
import re

import pyfastllm
from . import utils
from .utils.quantizer import QuantType


class InferConfig():
    def __init__(self,
                 max_length:int=2048,
                 top_p:float=0.7,
                 temperature:float=0.95,
                 **kwargs) -> None:

        configs = {
            "max_length": max_length,
            "top_p": top_p,
            "temperature": temperature
        }
        configs.update(kwargs)

        self.from_dict(configs)

    def from_dict(self, configs):
        self.configs = configs
        for key, val in configs.items():
            setattr(self, key, val)

    def to_dict(self, ):
        return self.configs
    
    @property
    def flm_config(self, ):
        flm_config = pyfastllm.GenerationConfig()
        for attr, val in self.configs.items():
            setattr(flm_config, attr, val)
        return flm_config


class BaseModel():
    def __init__(self, model_path:str) -> None:
        if model_path.endswith('flm'):
            print("loading model:", pyfastllm.get_llm_type(model_path))
            self.model = pyfastllm.create_llm(model_path)
        elif os.path.isdir(model_path):
            save_path = tempfile.mkstemp()
            utils.convert(model_path, save_path, q_type=QuantType.INT4)
            self.model = pyfastllm.create_llm(save_path)
        else:
            raise NotImplementedError(f"unsupport model type!")
    
    def build_input(self, query, history):
        raise NotImplementedError
    
    def is_stop(self, token_id):
        raise NotImplementedError
    
    def process_response(self, response):
        raise NotImplementedError

    def stream_chat(self,
                    tokenizer=None,
                    query:str='',
                    history=None,
                    max_length:int=2048,
                    top_p:float=0.7,
                    temperature:float=0.95,
                    *args, **kwargs):
        model = self.model
        infer_config = InferConfig(max_length=max_length, top_p=top_p, temperature=temperature, **kwargs)

        if not tokenizer: tokenizer = model.weight.tokenizer
        if not history: history = []
        
        prompt = self.build_input(query,history)
        input_ids = tokenizer.encode(prompt)
        handle = model.launch_response(input_ids, infer_config.flm_config)
        outputs = []
        ret_str = ""
        while len(outputs) < max_length:
            resp_token = model.fetch_response(handle)
            if self.is_stop(resp_token):
                break
            outputs.append(resp_token)
            content = tokenizer.decode(outputs)
            ret_str = self.process_response(content)
            yield ret_str, history + [(query, ret_str)]

    def chat(self,
                tokenizer=None,
                query:str='',
                history=None,
                max_length:int=2048,
                top_p:float=0.7,
                temperature:float=0.95,
                *args, **kwargs):
        model = self.model

        infer_config = InferConfig(max_length=max_length, top_p=top_p, temperature=temperature, **kwargs)

        if not tokenizer: tokenizer = model.weight.tokenizer
        if not history: history = []

        prompt = self.build_input(query, history=history)
        input_ids = tokenizer.encode(prompt)
        handle = model.launch_response(input_ids, infer_config)
        outputs = []
        ret_str = ""
        while len(outputs) < max_length:
            resp_token = model.fetch_response(handle)
            if self.is_stop(resp_token):
                break
            outputs.append(resp_token)
            content = tokenizer.decode(outputs)
            ret_str = self.process_response(content)
        history.append((query, ret_str))
        return ret_str, history


class ChatglmModel(BaseModel):
    def process_response(self, response):
        response = response.strip()
        response = response.replace("[[训练时间]]", "2023年")
        return response
    
    def is_stop(self, token_id):
        return token_id <= 2
    
    def build_input(self, query, history=None):
        if not history: history = []
        prompt = ""

        for i, (old_query, response) in enumerate(history):
            prompt += "[Round {}]\n问：{}\n答：{}\n".format(i, old_query, response)
        prompt += "[Round {}]\n问：{}\n答：".format(len(history), query)
        return prompt

class QwenModel(BaseModel):
    def process_response(self, response):
        return response
    
    def is_stop(self, token_id):
        chat_format = self.model.get("chat_format", "chatml")
        if chat_format == "raw":
            stop_words_ids = [151643]
        elif chat_format == "chatml":
            stop_words_ids = [151645, 151644]
        return token_id in stop_words_ids

    def build_inputs(self, query, history=None):
        prompt = ""
        chat_format = self.model.get("chat_format", "chatml")
        if chat_format == "chatml":
            if history is None: history = []
            prompt = f"{self.model.im_start} system \n {self.model.pre_prompt} + {self.model.im_end}"
            for i, (old_query, response) in enumerate(history):
                prompt += old_query + response
            prompt += f"\n {self.model.im_start + self.model.user_role} \n {query + self.model.im_end} \n  {self.model.im_start + self.model.bot_role} \n"
        elif chat_format == "raw":
            prompt = query
        else:
            raise NotImplementedError(f"Unknown char_format for QWen: {chat_format}")
        return prompt


class BaichuanModel(BaseModel):
    def process_response(self, response):
        return response
    
    def is_stop(self, token_id):
        return token_id == 2
    
    def build_input(self, query, history=None):
        prompt = ""
        round = 0
        # TODO 增加最长截断
        for i, (role, content) in enumerate(history):
            if role == "system" and i == 0:
                prompt += content
            elif role == "user":
                round += 1
                prompt += f"<reserved_102>{content}"
            elif role == "assistant":
                prompt += f"<reserved_103>{content}"
        
        return prompt



class MossModel(BaseModel):
    def process_response(self, response):
        return response
    
    def is_stop(self, token_id):
        return token_id == 106068

    def build_input(self, query, history=None):
        prompt = self.model.pre_prompt
        if not history: history = []

        for i, (old_query, response) in enumerate(history):
            prompt += old_query + response
        
        return prompt + f"{self.model.user_role} {query} {self.model.bot_role}"



class AutoFlmModel:
    def __init__(self) -> None:
        raise NotImplementedError
    
    @classmethod
    def from_pretrained(cls, model_path:str):
        # hf_model
        if os.path.isdir(model_path):
            save_path = tempfile.mkstemp(suffix='flm') 
            utils.convert(model_path, save_path, q_type=QuantType.INT4)
            model_path = save_path
            
        if model_path.endswith('flm'):
            model_type = pyfastllm.get_llm_type(model_path)
        else:
            raise NotImplementedError(f"unsupport model type!")
        
        if model_type == "chatglm":
            model = ChatglmModel(model_path)
        elif model_type == "qwen":
            model = QwenModel(model_path)
        elif model_type == "baichuan":
            model = BaichuanModel(model_path)
        elif model_type == "moss":
            model = MossModel(model_path)
        else:
            raise NotImplementedError(f"unsupport model: {model_type}!")

        return model