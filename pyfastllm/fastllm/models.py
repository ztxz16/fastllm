#!encoding=utf8
import fastllm
import logging
from typing import List, Tuple

try:
    import torch
except Exception as e:
    logging.warn("You must install torch before using this module!")


class ModelConfig():
    def __init__(self, **argv) -> None:
        self._dict = dict(argv)
        self._c_config = None

        for attr, value in argv.items():
            setattr(self, attr, value)
    
    def _to_dict(self, ):
        return self._dict
    
    def to_c_config(self, ):
        attr_map = {
            'max_length': 'output_token_limit',
        }

        if not self._c_config:
            self._c_config = fastllm.GenerationConfig()

            for attr, value in self._dict:
                setattr(self._c_config, attr_map.get(attr) or attr, value)

        return self._c_config

    def __str__(self, ):
        print("ModelConfig: ")
        for key, value in self._dict.items():
            print(f"{key} : {value}")


class baseModel:
    def __init__ (self, config:ModelConfig):
        pass

    def chat(self, ):
        pass

    def get_input_embeddings(self, ):
        pass

    def set_input_embeddings(self, ):
        pass

    def get_prompt(self, ):
        pass

    def forward(self, ):
        pass


def chatBaseModel():
    def get_output_embeddings():
        pass

    def set_output_embeddings():
        pass

    def prepare_inputs_for_generation():
        pass

    def forward():
        pass

    def stream_chat():
        pass

    def stream_generate():
        pass



    def get_prompt(self,
                   query: str,
                   history: List[Tuple[str, str]] = None) -> str:
        if (not(history)):
            history = []
        prompt = ""
        for i, (old_query, response) in enumerate(history):
            prompt = fastllm.make_history_llm_model(self.model, prompt.encode(), i, old_query.encode(), response.encode()).decode()
        prompt = fastllm.make_input_llm_model(self.model, prompt.encode(), len(history), query.encode()).decode()
        return prompt

    def save(self, path : str):
        fastllm.save_llm_model(self.model, path.encode())

    def response(self,
                 query: str,
                 history: List[Tuple[str, str]] = None) -> str:
        prompt = query if self.direct_query else self.get_prompt(query, history)
        ret = fastllm.response_str_llm_model(self.model, prompt.encode()).decode()
        return ret

    def stream_response(self,
                        query: str,
                        history: List[Tuple[str, str]] = None,
                        one_by_one = True):
        prompt = query if self.direct_query else self.get_prompt(query, history)
        handle = fastllm.launch_response_str_llm_model(self.model, prompt.encode())
        res = ""
        ret = b''
        while True:
            ret += fastllm.fetch_response_str_llm_model(self.model, handle)
            cur = ""
            try:
                cur = ret.decode()
                ret = b''
            except:
                pass
            if (cur == "<flmeos>"):
                break
            if one_by_one:
                yield cur
            else:
                res += cur
                yield res

    def chat(self, tokenizer, query: str, history: List[Tuple[str, str]] = None, max_length: int = 8192, num_beams=1,
             do_sample = True, top_p = 0.8, temperature = 0.8, logits_processor = None, **kwargs):
        if (not(history)):
            history = []
        prompt = query if self.direct_query else self.get_prompt(query, history)
        print("prompt", prompt)
        input = tokenizer.encode(prompt)
        handle = fastllm.launch_response_llm_model(self.model, len(input), (ctypes.c_int * len(input))(*input))

        result = []
        while True:
            cur = fastllm.fetch_response_llm_model(self.model, handle)
            if (cur == -1):
                break
            result.append(cur)
        response = tokenizer.decode(result)
        history = history + [(query, response)]
        return response, history

    def stream_chat(self, tokenizer, query: str, history: List[Tuple[str, str]] = None, past_key_values = None,
                    max_length: int = 8192, do_sample = True, top_p = 0.8, temperature = 0.8, logits_processor = None,
                    return_past_key_values = False, **kwargs) -> str:
        if (not(history)):
            history = []
        prompt = query if self.direct_query else self.get_prompt(query, history)
        input = tokenizer.encode(prompt)
        handle = fastllm.launch_response_llm_model(self.model, len(input), (ctypes.c_int * len(input))(*input))
        tokens = []
        while True:
            cur = fastllm.fetch_response_llm_model(self.model, handle)
            if (cur == -1):
                break
            tokens.append(cur)
            response = tokenizer.decode(tokens)
            new_history = history + [(query, response)]
            if return_past_key_values:
                yield response, new_history, None
            else:
                yield response, new_history


