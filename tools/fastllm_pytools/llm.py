import ctypes;
import os;
from typing import Optional, Tuple, Union, List, Callable, Dict, Any;

import platform
if platform.system() == 'Windows':
    fastllm_lib = ctypes.cdll.LoadLibrary(os.path.join(os.path.split(os.path.realpath(__file__))[0], "fastllm_tools.dll"))
else:
    fastllm_lib = ctypes.cdll.LoadLibrary(os.path.join(os.path.split(os.path.realpath(__file__))[0], "libfastllm_tools.so"))

fastllm_lib.create_llm_model.argtypes = [ctypes.c_char_p]
fastllm_lib.create_llm_model.restype = ctypes.c_int

fastllm_lib.launch_response_llm_model.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
                                                  ctypes.c_int, ctypes.c_bool, ctypes.c_float, ctypes.c_int,
                                                  ctypes.c_float, ctypes.c_float]
fastllm_lib.launch_response_llm_model.restype = ctypes.c_int

fastllm_lib.fetch_response_llm_model.argtypes = [ctypes.c_int, ctypes.c_int]
fastllm_lib.fetch_response_llm_model.restype = ctypes.c_int

fastllm_lib.response_str_llm_model.argtypes = [ctypes.c_int, ctypes.c_char_p,
                                               ctypes.c_int, ctypes.c_bool, ctypes.c_float, ctypes.c_int,
                                               ctypes.c_float, ctypes.c_float]
fastllm_lib.response_str_llm_model.restype = ctypes.c_char_p

fastllm_lib.launch_response_str_llm_model.argtype = [ctypes.c_int, ctypes.c_char_p,
                                                     ctypes.c_int, ctypes.c_bool, ctypes.c_float, ctypes.c_int,
                                                     ctypes.c_float, ctypes.c_float]
fastllm_lib.launch_response_str_llm_model.restype = ctypes.c_int

fastllm_lib.fetch_response_str_llm_model.argtypes = [ctypes.c_int, ctypes.c_int]
fastllm_lib.fetch_response_str_llm_model.restype = ctypes.c_char_p

fastllm_lib.make_history_llm_model.argtype = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_char_p]
fastllm_lib.make_history_llm_model.restype = ctypes.c_char_p

fastllm_lib.make_input_llm_model.argtype = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p]
fastllm_lib.make_input_llm_model.restype = ctypes.c_char_p

fastllm_lib.add_tokenizer_word_llm_model.argtype = [ctypes.c_int, ctypes.c_char_p, ctypes.c_float, ctypes.c_int]

fastllm_lib.set_device_map.argtype = [ctypes.c_int, ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]

def set_cpu_threads(threads: int):
    fastllm_lib.set_cpu_threads(threads);

def get_cpu_threads() -> int:
    return fastllm_lib.get_cpu_threads();

def print_ins_info():
    fastllm_lib.print_cpu_ins();

def set_cpu_kvcache(cpu_kvcache):
    fastllm_lib.set_kvcache_in_cpu(ctypes.c_bool(cpu_kvcache));

def get_cpu_kvcache():
    return fastllm_lib.get_kvcache_in_cpu();

def set_cpu_low_mem(low_mem):
    fastllm_lib.set_cpu_low_mem(ctypes.c_bool(low_mem));

def get_cpu_low_mem():
    return fastllm_lib.get_cpu_low_mem();

def set_device_map(device_map):
    devices = [];
    values = [];
    if (isinstance(device_map, str)):
        devices.append(device_map);
        values.append(1);
    elif (isinstance(device_map, list)):
        devices = [str(x) for x in device_map];
        values = [1 for x in device_map];
    elif (isinstance(device_map, dict)):
        devices = [str(x) for x in device_map.keys()];
        values = [int(device_map[x]) for x in device_map.keys()];
    else:
        print("set_device_map error.");
        return;
    device_str = ''.join(devices);
    device_len = [len(x) for x in devices];
    fastllm_lib.set_device_map(len(device_len),
                               (ctypes.c_int * len(device_len))(*device_len),
                               device_str.encode(),
                               (ctypes.c_int * len(values))(*values));
def from_hf(model,
            tokenizer = None,
            dtype = "float16"):
    from fastllm_pytools import hf_model;
    return hf_model.create(model, tokenizer, dtype = dtype);

class model:
    def __init__ (self, path : str,
                  id : int = -99999):
        if (id != -99999):
            self.model = id;
        else:
            self.model = fastllm_lib.create_llm_model(path.encode());
        self.direct_query = False;

    def get_prompt(self,
                   query: str,
                   history: List[Tuple[str, str]] = None) -> str:
        if (not(history)):
            history = [];
        prompt = "";
        for i, (old_query, response) in enumerate(history):
            prompt = fastllm_lib.make_history_llm_model(self.model, prompt.encode(), i, old_query.encode(), response.encode()).decode();
        prompt = fastllm_lib.make_input_llm_model(self.model, prompt.encode(), len(history), query.encode()).decode();
        return prompt;

    def save(self, path : str):
        fastllm_lib.save_llm_model(self.model, path.encode());

    def eval(self):
        pass;

    def response(self,
                 query: str,
                 history: List[Tuple[str, str]] = None,
                 max_length: int = 8192, do_sample = True, top_p = 0.8, top_k = 1, temperature = 1.0, repeat_penalty = 1.0) -> str:
        ret = "";
        for i in self.stream_response(query = query,
                                      history = history,
                                      max_length = max_length,
                                      do_sample = do_sample,
                                      top_p = top_p, top_k = top_k,
                                      temperature = temperature,
                                      repeat_penalty = repeat_penalty,
                                      one_by_one = True):
            ret += i;
        return ret;

    def stream_response(self,
                        query: str,
                        history: List[Tuple[str, str]] = None,
                        max_length: int = 8192, do_sample = True, top_p = 0.8, top_k = 1, temperature = 1.0, repeat_penalty = 1.0,
                        one_by_one = True):
        prompt = query if self.direct_query else self.get_prompt(query, history);
        handle = fastllm_lib.launch_response_str_llm_model(self.model, prompt.encode(),
                                                           ctypes.c_int(max_length), ctypes.c_bool(do_sample), ctypes.c_float(top_p), ctypes.c_int(top_k),
                                                           ctypes.c_float(temperature), ctypes.c_float(repeat_penalty));
        res = "";
        ret = b'';
        fail_cnt = 0;
        while True:
            ret += fastllm_lib.fetch_response_str_llm_model(self.model, handle);
            cur = "";
            try:
                cur = ret.decode();
                ret = b'';
            except:
                fail_cnt += 1;
                if (fail_cnt == 20):
                    break;
                else:
                    continue;
            fail_cnt = 0;
            if (cur == "<flmeos>"):
                break;
            if one_by_one:
                yield cur;
            else:
                res += cur;
                yield res;

    def chat(self, tokenizer, query: str, history: List[Tuple[str, str]] = None, max_length: int = 8192,
             do_sample = True, top_p = 0.8, top_k = 1, temperature = 1.0, repeat_penalty = 1.0, **kwargs):
        if (not(history)):
            history = [];
        prompt = query if self.direct_query else self.get_prompt(query, history);
        input = tokenizer.encode(prompt);
        handle = fastllm_lib.launch_response_llm_model(self.model, len(input), (ctypes.c_int * len(input))(*input),
                                                       max_length, do_sample, top_p, top_k, temperature, repeat_penalty);

        result = [];
        while True:
            cur = fastllm_lib.fetch_response_llm_model(self.model, handle);
            if (cur == -1):
                break;
            result.append(cur);
        response = tokenizer.decode(result);
        history = history + [(query, response)];
        return response, history;

    def stream_chat(self, tokenizer, query: str, history: List[Tuple[str, str]] = None, past_key_values = None,
                    max_length: int = 8192, do_sample = True, top_p = 0.8, top_k = 1, temperature = 1.0, repeat_penalty = 1.0,
                    return_past_key_values = False, **kwargs) -> str:
        if (not(history)):
            history = [];
        prompt = query if self.direct_query else self.get_prompt(query, history);
        input = tokenizer.encode(prompt);
        handle = fastllm_lib.launch_response_llm_model(self.model, len(input), (ctypes.c_int * len(input))(*input),
                                                       max_length, do_sample, top_p, top_k, temperature, repeat_penalty);
        tokens = [];
        while True:
            cur = fastllm_lib.fetch_response_llm_model(self.model, handle);
            if (cur == -1):
                break;
            tokens.append(cur);
            response = tokenizer.decode(tokens);
            new_history = history + [(query, response)];
            if return_past_key_values:
                yield response, new_history, None;
            else:
                yield response, new_history;
