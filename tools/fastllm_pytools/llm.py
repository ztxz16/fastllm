import ctypes
import math
import os
import threading
import asyncio
import copy
import json
import math
from typing import Optional, Tuple, Union, List, Callable, Dict, Any;

try:
    import sentencepiece # 先加载sentencepiece，防止libc冲突
except:
    pass

import platform
if platform.system() == 'Windows':
    fastllm_lib = ctypes.CDLL(os.path.join(os.path.split(os.path.realpath(__file__))[0], "fastllm_tools.dll"), winmode=0)
elif platform.system() == 'Darwin':
    fastllm_lib = ctypes.cdll.LoadLibrary(os.path.join(os.path.split(os.path.realpath(__file__))[0], "libfastllm_tools.dylib"))
else:
    succ = False
    for extraLibName in ["libnuma.so.1"]:
        try:
            ctypes.cdll.LoadLibrary(os.path.join(os.path.split(os.path.realpath(__file__))[0], extraLibName))
            print("Load", extraLibName)
        except:
            continue
    for libname in ["libfastllm_tools.so", "libfastllm_tools-cu11.so", "libfastllm_tools-cpu.so"]:
        try:
            fastllm_lib = ctypes.cdll.LoadLibrary(os.path.join(os.path.split(os.path.realpath(__file__))[0], libname))
            print("Load", libname)
            succ = True
            break
        except OSError:
            continue
        except Exception as e:
            # e.g. Error loading libfastllm_tools.so: /root/anaconda3/envs/ftllm/bin/../lib/libstdc++.so.6: version `GLIBCXX_3.4.32' not found (required by /root/anaconda3/envs/ftllm/lib/python3.12/site-packages/ftllm/libfastllm_tools.so)
            print(f"Error loading {libname}: {e}")
            continue
    if (not(succ)):
        print("Load fastllm failed. (Try update glibc)")
        exit(0)

fastllm_lib.export_llm_model_fromhf.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_bool, ctypes.c_int, ctypes.c_int, ctypes.c_char_p]

fastllm_lib.create_llm_model.argtypes = [ctypes.c_char_p]
fastllm_lib.create_llm_model.restype = ctypes.c_int

fastllm_lib.create_llm_model_fromhf.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_bool, ctypes.c_char_p, ctypes.c_bool, ctypes.c_int, ctypes.c_int, ctypes.c_char_p]
fastllm_lib.create_llm_model_fromhf.restype = ctypes.c_int

fastllm_lib.create_llm_model_fromhf_with_config.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_bool, ctypes.c_char_p]
fastllm_lib.create_llm_model_fromhf_with_config.restype = ctypes.c_int

fastllm_lib.create_llm_tokenizer_fromhf.argtypes = [ctypes.c_char_p]
fastllm_lib.create_llm_tokenizer_fromhf.restype = ctypes.c_int

fastllm_lib.add_eos_token.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int]

fastllm_lib.token_decode.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_char_p]
fastllm_lib.token_decode.restype = ctypes.c_int

fastllm_lib.token_encode_string.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
fastllm_lib.token_encode_string.restype = ctypes.c_int

fastllm_lib.launch_response_llm_model.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
                                                  ctypes.c_int, ctypes.c_int, ctypes.c_bool, ctypes.c_float, ctypes.c_int,
                                                  ctypes.c_float, ctypes.c_float, ctypes.c_bool,
                                                  ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
fastllm_lib.launch_response_llm_model.restype = ctypes.c_int

fastllm_lib.launch_response_llm_model_multimodal.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
                                                            ctypes.c_char_p, ctypes.c_void_p, 
                                                            ctypes.c_int, ctypes.c_int, ctypes.c_bool, ctypes.c_float, ctypes.c_int,
                                                            ctypes.c_float, ctypes.c_float, ctypes.c_bool,
                                                            ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
fastllm_lib.launch_response_llm_model_multimodal.restype = ctypes.c_int


fastllm_lib.add_cache_llm_model.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]

fastllm_lib.fetch_response_llm_model.argtypes = [ctypes.c_int, ctypes.c_int]
fastllm_lib.fetch_response_llm_model.restype = ctypes.c_int

fastllm_lib.fetch_response_logits_llm_model.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_float)]
fastllm_lib.fetch_response_logits_llm_model.restype = ctypes.c_int

fastllm_lib.response_str_llm_model.argtypes = [ctypes.c_int, ctypes.c_char_p,
                                               ctypes.c_int, ctypes.c_bool, ctypes.c_float, ctypes.c_int,
                                               ctypes.c_float, ctypes.c_float, ctypes.c_bool]
fastllm_lib.response_str_llm_model.restype = ctypes.c_char_p

fastllm_lib.launch_response_str_llm_model.argtype = [ctypes.c_int, ctypes.c_char_p,
                                                     ctypes.c_int, ctypes.c_int, ctypes.c_bool, ctypes.c_float, ctypes.c_int,
                                                     ctypes.c_float, ctypes.c_float, ctypes.c_bool,
                                                     ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
fastllm_lib.launch_response_str_llm_model.restype = ctypes.c_int

fastllm_lib.fetch_response_str_llm_model.argtypes = [ctypes.c_int, ctypes.c_int]
fastllm_lib.fetch_response_str_llm_model.restype = ctypes.c_char_p

fastllm_lib.can_fetch_response_llm_model.argtypes = [ctypes.c_int, ctypes.c_int]
fastllm_lib.can_fetch_response_llm_model.restype = ctypes.c_bool

fastllm_lib.abort_response_llm_model.argtypes = [ctypes.c_int, ctypes.c_int]

fastllm_lib.make_history_llm_model.argtype = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_char_p]
fastllm_lib.make_history_llm_model.restype = ctypes.c_char_p

fastllm_lib.make_input_llm_model.argtype = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p]
fastllm_lib.make_input_llm_model.restype = ctypes.c_char_p

fastllm_lib.add_tokenizer_word_llm_model.argtype = [ctypes.c_int, ctypes.c_char_p, ctypes.c_float, ctypes.c_int]

fastllm_lib.set_special_tokens_llm_model.argtype = [ctypes.c_int, ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]

fastllm_lib.set_device_map.argtype = [ctypes.c_int, ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
fastllm_lib.set_moe_device_map.argtype = [ctypes.c_int, ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]

fastllm_lib.apply_chat_template.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
fastllm_lib.apply_chat_template.restype = ctypes.c_char_p

fastllm_lib.set_kv_cache_limit_llm_model.argtypes = [ctypes.c_int, ctypes.c_int64]

fastllm_lib.set_max_batch_llm_model.argtypes = [ctypes.c_int, ctypes.c_int]

fastllm_lib.set_verbose_llm_model.argtypes = [ctypes.c_int, ctypes.c_bool]

fastllm_lib.get_max_input_len_llm_model.argtypes = [ctypes.c_int]
fastllm_lib.get_max_input_len_llm_model.restype = ctypes.c_int

fastllm_lib.get_struct_llm_model.argtypes = [ctypes.c_int]
fastllm_lib.get_struct_llm_model.restype = ctypes.c_char_p

fastllm_lib.embedding_sentence.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_bool, ctypes.POINTER(ctypes.c_int)]
fastllm_lib.embedding_sentence.restype = ctypes.POINTER(ctypes.c_float)

fastllm_lib.embedding_tokens.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.c_bool, ctypes.POINTER(ctypes.c_int)]
fastllm_lib.embedding_tokens.restype = ctypes.POINTER(ctypes.c_float)

fastllm_lib.reranker_compute_score.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
fastllm_lib.reranker_compute_score.restype = ctypes.POINTER(ctypes.c_float)

fastllm_lib.t2s_decode.argtypes = [ctypes.c_char_p,
    ctypes.c_int, ctypes.c_int, ctypes.c_void_p, 
    ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p), 
    ctypes.POINTER(ctypes.c_int64), 
    ctypes.POINTER(ctypes.c_float)]
fastllm_lib.t2s_decode.restype = ctypes.POINTER(ctypes.c_int64)

fastllm_data_type_dict = {
    "": 0,
    "int2g": 11,
    "int4g": 9,
    "int4": 8,
    "int8": 3,
    "float16": 7,
    "fp16": 7,
    "half": 7,
    "float32": 0,
    "fp8": 10,
    "float8": 10,
    "fp8_e4m3": 10,
}

def export_llm_model_fromhf(path : str,
                            output : str,
                            dtype : str = "float16",
                            moe_dtype : str = "",
                            lora: str = "",
                            dtype_config: str = ""):
    if (dtype == "auto"):
        dtype = "float16"

    int4g_groupcnt = 128;
    if ((dtype.startswith("int4g") or dtype.startswith("int2g")) and len(dtype) > 5):
        try:
            int4g_groupcnt = int(dtype[5:])
            dtype = dtype[:5]
        except:
            print("dtype should be like \"int4g256\"")
            exit(0)
    if (dtype not in fastllm_data_type_dict):
        print("dtype should be one of ", list(fastllm_data_type_dict.keys()))
        exit(0)

    moe_int4g_groupcnt = 128
    use_moe_dtype = False
    if (moe_dtype != ""):
        if ((moe_dtype.startswith("int4g") or moe_dtype.startswith("int2g")) and len(moe_dtype) > 5):
            try:
                moe_int4g_groupcnt = int(moe_dtype[5:])
                moe_dtype = moe_dtype[:5]
            except:
                print("dtype should be like \"int4g256\"")
                exit(0)
        if (moe_dtype not in fastllm_data_type_dict):
            print("moe_dtype should be one of ", list(fastllm_data_type_dict.keys()))
            exit(0)
        use_moe_dtype = True
    
    if (dtype_config != "" and os.path.exists(dtype_config)):
        with open(dtype_config, "r", encoding="utf-8") as file:
            dtype_config = file.read()

    fastllm_lib.export_llm_model_fromhf(path.encode(), fastllm_data_type_dict[dtype], int4g_groupcnt, lora.encode(), output.encode(), 
                                        use_moe_dtype, fastllm_data_type_dict[moe_dtype], moe_int4g_groupcnt, dtype_config.encode());

def softmax(a):
    max_value = a[0]
    for i in a:
        max_value = max(max_value, i)
    sum = 0.0
    for i in range(len(a)):
        a[i] = math.exp(a[i] - max_value)
        sum += a[i]
    for i in range(len(a)):
        a[i] /= sum

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

def set_cpu_historycache(cpu_historycache):
    fastllm_lib.set_historycache_in_cpu(ctypes.c_bool(cpu_historycache));

def get_cpu_historycache():
    return fastllm_lib.get_historycache_in_cpu();

def set_cuda_embedding(cuda_embedding):
    fastllm_lib.set_cuda_embedding(ctypes.c_bool(cuda_embedding));

def set_cuda_shared_expert(cuda_shared_expert):
    fastllm_lib.set_cuda_shared_expert(ctypes.c_bool(cuda_shared_expert));

def set_cpu_low_mem(low_mem):
    fastllm_lib.set_cpu_low_mem(ctypes.c_bool(low_mem));

def get_cpu_low_mem():
    return fastllm_lib.get_cpu_low_mem();

def set_device_map(device_map, is_moe = False):
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
    if (is_moe):
        fastllm_lib.set_moe_device_map(len(device_len),
                               (ctypes.c_int * len(device_len))(*device_len),
                               device_str.encode(),
                               (ctypes.c_int * len(values))(*values));
    else:
        fastllm_lib.set_device_map(len(device_len),
                               (ctypes.c_int * len(device_len))(*device_len),
                               device_str.encode(),
                               (ctypes.c_int * len(values))(*values));

def t2s_decode(safetensors_path, xy_pos, k_cache, v_cache, y, pe):
    bsz = xy_pos.shape[0]
    src_len = v_cache[0].shape[1]
    assert bsz == 1
    assert xy_pos.shape[1] == 1 and xy_pos.shape[2] == 512
    assert len(k_cache) == 24 and len(v_cache) == 24
    assert k_cache[0].shape[0] == bsz and k_cache[0].shape[1] == src_len and k_cache[0].shape[2] == 512
    assert y.shape[0] == bsz
    assert pe.shape[0] == 1 and pe.shape[1] == 4000 and pe.shape[2] == 512 

    y = y.cpu()
    pe_list = pe.view(4000 * 512).tolist();
    pe_p = (ctypes.c_float * len(pe_list))(*pe_list)
    kk = []
    k_cache_ct = []
    vv = []
    v_cache_ct = []
    for i in range(len(k_cache)):
        kk.append(k_cache[i].contiguous())
        vv.append(v_cache[i].contiguous())
        k_cache_ct.append(ctypes.cast(kk[-1].data_ptr(), ctypes.c_void_p))
        v_cache_ct.append(ctypes.cast(vv[-1].data_ptr(), ctypes.c_void_p))

    ret_c = fastllm_lib.t2s_decode(
        safetensors_path.encode(),
        src_len, y.shape[1], 
        ctypes.cast(xy_pos.data_ptr(), ctypes.c_void_p),
        (ctypes.c_void_p * len(k_cache_ct))(*k_cache_ct),
        (ctypes.c_void_p * len(v_cache_ct))(*v_cache_ct),
        ctypes.cast(y.data_ptr(), ctypes.POINTER(ctypes.c_int64)),
        pe_p)
    num = ret_c[0];
    ret = []
    for i in range(num):
        ret.append(ret_c[i+1]);
    return ret

def from_hf(model,
            tokenizer = None,
            pre_prompt = None,
            user_role = None,
            bot_role = None,
            history_sep = None,
            dtype = "float16"):
    from ftllm import hf_model;
    return hf_model.create(model, tokenizer, pre_prompt = pre_prompt, user_role = user_role,
                           bot_role = bot_role, history_sep = history_sep, dtype = dtype);

class tokenizer:
    def __init__ (self, path : str,
                  id : int = -99999,
                  system_prompt : str = ""):
        self.systemp_prompt = system_prompt
        if (id != -99999):
            self.model = id
        else:
            if os.path.isfile(path):
                self.model = fastllm_lib.create_llm_tokenizer(path.encode());
            elif os.path.isdir(path):
                self.model = fastllm_lib.create_llm_tokenizer_fromhf(path.encode());
            else:
                print("path error: ", path);
                exit(0)
        self.thread_local_obj = threading.local()
        self.tokenizer_decode_token_cache = None
    
    def apply_chat_template(
        self,
        conversation: Union[List[Dict[str, str]], List[List[Dict[str, str]]], "Conversation"],
        chat_template: Optional[str] = None,
        add_generation_prompt: bool = False,
        tokenize: bool = True,
        #padding: bool = False,
        #truncation: bool = False,
        #max_length: Optional[int] = None,
        #return_tensors: Optional[Union[str, TensorType]] = None,
        #return_dict: bool = False,
        #tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Union[str, List[int], List[str], List[List[int]]]:
        if isinstance(conversation, (list, tuple)) and (
            isinstance(conversation[0], (list, tuple)) or hasattr(conversation[0], "messages")
        ):
            conversations = conversation
            is_batched = True
        else:
            conversations = [conversation]
            is_batched = False
        strs = []        
        for conversation in conversations:
            messages = []
            for it in conversation:
                if it["role"] == "system":
                    messages += ["system", it["content"]]
            for it in conversation:
                if it["role"] != "system":
                    messages += [it["role"], it["content"]]
            poss = []
            lens = []
            all = b''
            for i in range(len(messages)):
                messages[i] = messages[i].encode()
                all += messages[i]
                poss.append(0 if i == 0 else poss[-1] + lens[-1])
                lens.append(len(messages[i]))
            strs.append(fastllm_lib.apply_chat_template(self.model, all, len(messages), (ctypes.c_int * len(poss))(*poss), (ctypes.c_int * len(lens))(*lens)).decode())
        if (is_batched):
            return strs
        else:
            return strs[0]
        
    def encode(
        self,
        text: str,
        #text_pair: Optional[Union[TextInput, PreTokenizedInput, EncodedInput]] = None,
        #add_special_tokens: bool = True,
        #padding: Union[bool, str, PaddingStrategy] = False,
        #truncation: Union[bool, str, TruncationStrategy] = None,
        #max_length: Optional[int] = None,
        #stride: int = 0,
        #return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs,
    ) -> List[int]:
        content = text
        output_buffer_init_len = 1024
        if "tokenizer_encode_string__output_buffer" not in dir(self.thread_local_obj) or self.thread_local_obj.tokenizer_encode_string__output_buffer is None:
            self.thread_local_obj.tokenizer_encode_string__output_buffer = (ctypes.c_int * output_buffer_init_len)()

        buffer = self.thread_local_obj.tokenizer_encode_string__output_buffer
        buffer_len = len(buffer)
        result_len = fastllm_lib.token_encode_string(self.model, content.encode(), buffer_len, buffer)
        if result_len > buffer_len:
            if result_len > 10240:
                # 要处理的数据过长，使用一次性的buffer
                temp_buffer = (ctypes.c_int * result_len)()
                ret = fastllm_lib.token_encode_string(self.model, content.encode(), result_len, temp_buffer)
                return [i for i in temp_buffer]
            else:
                # 扩展buffer大小
                new_buffer_len = round(math.ceil(result_len / 1024.0)) * 1024
                buffer = (ctypes.c_int * new_buffer_len)()
                self.thread_local_obj.tokenizer_encode_string__output_buffer = buffer
                result_len = fastllm_lib.token_encode_string(self.model, content.encode(), new_buffer_len, buffer)

        return [buffer[i] for i in range(result_len)]

# copy from transformers
class GenerationConfig:
    def __init__(self, **kwargs):
        # Parameters that control the length of the output
        self.max_length = kwargs.pop("max_length", 20)
        self.max_new_tokens = kwargs.pop("max_new_tokens", 256)
        self.min_length = kwargs.pop("min_length", 0)
        self.min_new_tokens = kwargs.pop("min_new_tokens", None)
        self.early_stopping = kwargs.pop("early_stopping", False)
        self.max_time = kwargs.pop("max_time", None)
        self.stop_strings = kwargs.pop("stop_strings", None)

        # Parameters that control the generation strategy used
        self.do_sample = kwargs.pop("do_sample", False)
        self.num_beams = kwargs.pop("num_beams", 1)
        self.num_beam_groups = kwargs.pop("num_beam_groups", 1)
        self.penalty_alpha = kwargs.pop("penalty_alpha", None)
        self.use_cache = kwargs.pop("use_cache", True)

        # Parameters for manipulation of the model output logits
        self.temperature = kwargs.pop("temperature", 1.0)
        self.top_k = kwargs.pop("top_k", 1)
        self.top_p = kwargs.pop("top_p", 1.0)
        self.min_p = kwargs.pop("min_p", None)
        self.typical_p = kwargs.pop("typical_p", 1.0)
        self.epsilon_cutoff = kwargs.pop("epsilon_cutoff", 0.0)
        self.eta_cutoff = kwargs.pop("eta_cutoff", 0.0)
        self.diversity_penalty = kwargs.pop("diversity_penalty", 0.0)
        self.repetition_penalty = kwargs.pop("repetition_penalty", 1.0)
        self.encoder_repetition_penalty = kwargs.pop("encoder_repetition_penalty", 1.0)
        self.length_penalty = kwargs.pop("length_penalty", 1.0)
        self.no_repeat_ngram_size = kwargs.pop("no_repeat_ngram_size", 0)
        self.bad_words_ids = kwargs.pop("bad_words_ids", None)
        self.force_words_ids = kwargs.pop("force_words_ids", None)
        self.renormalize_logits = kwargs.pop("renormalize_logits", False)
        self.constraints = kwargs.pop("constraints", None)
        self.forced_bos_token_id = kwargs.pop("forced_bos_token_id", None)
        self.forced_eos_token_id = kwargs.pop("forced_eos_token_id", None)
        self.remove_invalid_values = kwargs.pop("remove_invalid_values", False)
        self.exponential_decay_length_penalty = kwargs.pop("exponential_decay_length_penalty", None)
        self.suppress_tokens = kwargs.pop("suppress_tokens", None)
        self.begin_suppress_tokens = kwargs.pop("begin_suppress_tokens", None)
        self.forced_decoder_ids = kwargs.pop("forced_decoder_ids", None)
        self.sequence_bias = kwargs.pop("sequence_bias", None)
        self.token_healing = kwargs.pop("token_healing", False)
        self.guidance_scale = kwargs.pop("guidance_scale", None)
        self.low_memory = kwargs.pop("low_memory", None)
        #watermarking_config = kwargs.pop("watermarking_config", None)
        #if watermarking_config is None:
        #    self.watermarking_config = None
        #elif isinstance(watermarking_config, WatermarkingConfig):
        #    self.watermarking_config = watermarking_config
        #else:
        #    self.watermarking_config = WatermarkingConfig.from_dict(watermarking_config)

        # Parameters that define the output variables of `generate`
        self.num_return_sequences = kwargs.pop("num_return_sequences", 1)
        self.output_attentions = kwargs.pop("output_attentions", False)
        self.output_hidden_states = kwargs.pop("output_hidden_states", False)
        self.output_scores = kwargs.pop("output_scores", False)
        self.output_logits = kwargs.pop("output_logits", None)
        self.return_dict_in_generate = kwargs.pop("return_dict_in_generate", False)

        # Special tokens that can be used at generation time
        self.pad_token_id = kwargs.pop("pad_token_id", None)
        self.bos_token_id = kwargs.pop("bos_token_id", None)
        self.eos_token_id = kwargs.pop("eos_token_id", None)

        # Generation parameters exclusive to encoder-decoder models
        self.encoder_no_repeat_ngram_size = kwargs.pop("encoder_no_repeat_ngram_size", 0)
        self.decoder_start_token_id = kwargs.pop("decoder_start_token_id", None)

        # Assistant generation
        self.num_assistant_tokens = kwargs.pop("num_assistant_tokens", 5)
        self.num_assistant_tokens_schedule = kwargs.pop("num_assistant_tokens_schedule", "heuristic")

        # Cache implementation
        self.cache_implementation = kwargs.pop("cache_implementation", None)
        self.cache_config = kwargs.pop("cache_config", None)
        #if self.cache_implementation is not None:
        #    cache_config_class = NEEDS_CACHE_CONFIG[self.cache_implementation]
        #    if self.cache_config is None:
        #        self.cache_config = cache_config_class()
        #    elif isinstance(self.cache_config, dict):
        #        self.cache_config = cache_config_class.from_dict(self.cache_config)
        self.return_legacy_cache = kwargs.pop("return_legacy_cache", True)

        # Prompt lookup decoding
        self.prompt_lookup_num_tokens = kwargs.pop("prompt_lookup_num_tokens", None)
        self.max_matching_ngram_size = kwargs.pop("max_matching_ngram_size", None)

        # Wild card
        self.generation_kwargs = kwargs.pop("generation_kwargs", {})

        # The remaining attributes do not parametrize `.generate()`, but are informative and/or used by the hub
        # interface.
        self._from_model_config = kwargs.pop("_from_model_config", False)
        self._commit_hash = kwargs.pop("_commit_hash", None)
        self.transformers_version = kwargs.pop("transformers_version", "fastllm")

        # Additional attributes without default values
        if not self._from_model_config:
            # we don't want to copy values from the model config if we're initializing a `GenerationConfig` from a
            # model's default configuration file
            for key, value in kwargs.items():
                try:
                    setattr(self, key, value)
                except AttributeError as err:                    
                    raise err
        # Validate the values of the attributes
        # self.validate(is_init = True)

class TokenizerCache:
    def __init__ (self):
        from collections import deque
        self.caches = deque(maxlen = 100)
    
    def add(self, prompt, tokens):
        # print("add cache", prompt[:100], tokens[:100])
        self.caches.append([prompt, tokens])
    
    def prompt_can_match(self, prompt, cur_len, s):
        if (cur_len + len(s) > len(prompt)):
            return False
        for i in range(len(s)):
            if (prompt[cur_len + i] != s[i]):
                return False
        return True
    
    def tokenize_with_cache(self, tokenizer, prompt: str):
        max_len = 0
        use_cahce_prompt = ""
        use_cache_tokens = []
        for it in self.caches:
            cur_tokens = []
            cur_prompt = ""
            cur_len = 0
            for i in range(len(it[0])):    
                if (self.prompt_can_match(prompt, cur_len, it[0][i])):
                    cur_prompt += it[0][i]
                    cur_len += len(it[0][i])
                    cur_tokens += it[1][i]
                else:
                    break
            if cur_len > max_len:
                max_len = cur_len
                use_cahce_prompt = cur_prompt
                import copy
                use_cache_tokens = copy.deepcopy(cur_tokens)
        #print("use_cahce_prompt", use_cahce_prompt[:100])
        #print("use_cache_tokens", use_cache_tokens[:100])

        if (max_len > 0):
            #print("real prompt", prompt)
            #print("decode", tokenizer.decode(use_cache_tokens + tokenizer.encode(prompt[max_len : ], add_special_tokens = False)))
            return use_cache_tokens + tokenizer.encode(prompt[max_len : ], add_special_tokens = False)
        else:
            return tokenizer.encode(prompt)

class model:
    def __init__ (self, path : str,
                  id : int = -99999,
                  dtype : str = "float16",
                  moe_dtype : str = "",
                  system_prompt : str = "",
                  eos_token: List[str] = [],
                  tokenizer_type = "auto", 
                  model_json: str = "", 
                  graph: type = None, 
                  lora: str = "", 
                  dtype_config: str = ""):
        if (graph != None):
            current_graph = graph()
            if (os.path.isdir(path) and os.path.isfile(os.path.join(path, "config.json"))):
                if (os.path.isfile(os.path.join(path, "config.json"))):
                    current_graph.config = json.load(open(os.path.join(path, "config.json"), "r"))
                if (os.path.isfile(os.path.join(path, "tokenizer_config.json"))):
                    current_graph.tokenizer_config = json.load(open(os.path.join(path, "tokenizer_config.json"), "r"))
                if (os.path.isfile(os.path.join(path, "generation_config.json"))):
                    current_graph.generation_config = json.load(open(os.path.join(path, "generation_config.json"), "r"))
                current_graph.build()
                model_json = str(current_graph)
            else:
                print("When using a custom Graph, only model folders in HF format can be read.")
                exit(0)    

        int4g_groupcnt = 128;
        if ((dtype.startswith("int4g") or dtype.startswith("int2g")) and len(dtype) > 5):
            try:
                int4g_groupcnt = int(dtype[5:])
                dtype = dtype[:5]
            except:
                print("dtype should be like \"int4g256\"")
                exit(0)    
        if (dtype not in fastllm_data_type_dict):
            print("dtype should be one of ", list(fastllm_data_type_dict.keys()))
            exit(0)

        moe_int4g_groupcnt = 128
        use_moe_dtype = False
        if (moe_dtype != ""):
            if ((moe_dtype.startswith("int4g") or moe_dtype.startswith("int2g")) and len(moe_dtype) > 5):
                try:
                    moe_int4g_groupcnt = int(moe_dtype[5:])
                    moe_dtype = moe_dtype[:5]
                except:
                    print("dtype should be like \"int4g256\"")
                    exit(0)
            if (moe_dtype not in fastllm_data_type_dict):
                print("moe_dtype should be one of ", list(fastllm_data_type_dict.keys()))
                exit(0)
            use_moe_dtype = True
        
        self.save_history = False
        self.tokenizer_cache = TokenizerCache()
        self.current_tokenizer_cache = dict()
        
        self.hf_tokenizer = None
        self.enable_thinking = True

        if (id != -99999):
            self.model = id;
        else:
            if os.path.isfile(path):
                self.model = fastllm_lib.create_llm_model(path.encode());
            elif os.path.isdir(path):
                if tokenizer_type != "fastllm":
                    try:
                        import logging
                        # 1. 保存当前的日志级别
                        original_level = logging.root.manager.disable
                        # 2. 完全禁止所有 logging 输出
                        logging.disable(logging.CRITICAL)  # 禁用所有日志（包括 ERROR, WARNING, INFO, DEBUG）
                        from transformers import AutoTokenizer
                        logging.disable(original_level)  # 恢复原来的日志级别
                        self.hf_tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code = True)
                    except:
                        self.hf_tokenizer = None
                        print("Load AutoTokenizer failed. (you can try install transformers)")
                        print("Try load fastllm tokenizer.")
                if model_json != "":
                    self.model = fastllm_lib.create_llm_model_fromhf_with_config(path.encode(), fastllm_data_type_dict[dtype], int4g_groupcnt, 
                                                                 ctypes.c_bool(self.hf_tokenizer != None), model_json.encode());
                else:
                    self.model = fastllm_lib.create_llm_model_fromhf(path.encode(), fastllm_data_type_dict[dtype], int4g_groupcnt, 
                                                                 ctypes.c_bool(self.hf_tokenizer != None), lora.encode(), 
                                                                 use_moe_dtype, fastllm_data_type_dict[moe_dtype], moe_int4g_groupcnt, dtype_config.encode());
                if (os.path.isfile(os.path.join(path, "config.json"))):
                    self.config = json.load(open(os.path.join(path, "config.json"), "r"))
            else:
                print("path error: ", path);
                exit(0)

        self.direct_query = False;
        self.system_prompt = system_prompt;
        self.eos_token = [] + eos_token
        for token in self.eos_token:
            fastllm_lib.add_eos_token(self.model, token.encode(), len(token.encode()));

        # 为了减少重复申请释放buffer对象而使用的线程局部存储区对象池
        self.thread_local_obj = threading.local()
        #self.thread_local_obj.tokenizer_encode_string__output_buffer = None
        #self.thread_local_obj.tokenizer_decode_token__output_buffer = None

        # tokenizer_decode_token 输出结果的静态缓存，手工触发构建
        # 由于token数量有限且不太多，所以缓存该结果来减少调用较为适合。
        # 不做成自动缓存是为了避免在多线程调用的时候对缓存dict加锁，同时也为不同场景提供选择空间
        self.tokenizer_decode_token_cache = None
    
    def apply_chat_template(
        self,
        conversation: List[Dict[str, str]],
        chat_template: Optional[str] = None,
        add_generation_prompt: bool = False,
        **kwargs,
    ) -> str:
        messages = []
        for it in conversation:
            if it["role"] == "system":
                messages += ["system", it["content"]]
        for it in conversation:
            if it["role"] != "system":
                messages += [it["role"], it["content"]]
        poss = []
        lens = []
        all = b''
        for i in range(len(messages)):
            messages[i] = messages[i].encode()
            all += messages[i]
            poss.append(0 if i == 0 else poss[-1] + lens[-1])
            lens.append(len(messages[i]))
        str = fastllm_lib.apply_chat_template(self.model, all, len(messages), (ctypes.c_int * len(poss))(*poss), (ctypes.c_int * len(lens))(*lens)).decode()
        return str

    def generate(
        self,
        inputs,
        #generation_config = None,
        #logits_processor: Optional[LogitsProcessorList] = None,
        #stopping_criteria: Optional[StoppingCriteriaList] = None,
        #prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
        #synced_gpus: Optional[bool] = None,
        #assistant_model: Optional["PreTrainedModel"] = None,
        #streamer: Optional["BaseStreamer"] = None,
        #negative_prompt_ids: Optional[torch.Tensor] = None,
        #negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        **kwargs) : #-> Union[GenerateOutput, torch.LongTensor]:
        if (str(type(inputs)).find("torch.Tensor") != -1):
            inputs = inputs.tolist()
        config = GenerationConfig(**kwargs)        
        stop_token_len, stop_token_list = self.stop_token_ctypes(None)
        handles = []
        for i in range(len(inputs)):
            handles.append(fastllm_lib.launch_response_llm_model(self.model, len(inputs[i]),
                                                       (ctypes.c_int * len(inputs[i]))(*inputs[i]),
                                                       ctypes.c_int(config.max_new_tokens), ctypes.c_int(0), ctypes.c_bool(config.do_sample), 
                                                       ctypes.c_float(config.top_p), ctypes.c_int(config.top_k),
                                                       ctypes.c_float(config.temperature), ctypes.c_float(config.repetition_penalty), ctypes.c_bool(False),
                                                       stop_token_len, stop_token_list))
        outputs = inputs
        for i in range(len(inputs)):
            while True:
                cur_token = fastllm_lib.fetch_response_llm_model(self.model, handles[i])
                if cur_token <= -1:
                    break
                outputs[i].append(cur_token)
        return outputs
        
    def get_prompt(self,
                   query: str,
                   history: List[Tuple[str, str]] = None) -> str:
        if (not(history)):
            history = [];
        messages = []

        if (self.hf_tokenizer != None and hasattr(self.hf_tokenizer, "chat_template") and self.hf_tokenizer.chat_template != ""):
            if (self.system_prompt != ""):
                messages.append({"role": "system", "content": self.system_prompt})
            for his in history:
                messages.append({"role": "user", "content": his[0]})
                messages.append({"role": "assistant", "content": his[1]})
            messages.append({"role": "user", "content": query})
            return self.hf_tokenizer.apply_chat_template(messages, tokenize = False, enable_thinking = self.enable_thinking, add_generation_prompt = True)
        else:
            if (self.system_prompt != ""):
                messages += ["system", self.system_prompt]
            for his in history:
                messages += ["user", his[0], "assistant", his[1]]
            messages += ["user", query]
            poss = []
            lens = []
            all = b''

            for i in range(len(messages)):
                messages[i] = messages[i].encode()
                all += messages[i]
                poss.append(0 if i == 0 else poss[-1] + lens[-1])
                lens.append(len(messages[i]))
            return fastllm_lib.apply_chat_template(self.model, all, len(messages), (ctypes.c_int * len(poss))(*poss), (ctypes.c_int * len(lens))(*lens)).decode()

    def save(self, path : str):
        fastllm_lib.save_llm_model(self.model, path.encode());

    def eval(self):
        return self;

    def build_tokenizer_decode_token_cache(self):
        if self.tokenizer_decode_token_cache is not None:
            return

        cache_dict = dict()
        vocab_size = fastllm_lib.get_tokenizer_vocab_size(self.model)
        for token_id in range(vocab_size):
            cache_dict[token_id] = self.tokenizer_decode_token(token_id)

        self.tokenizer_decode_token_cache = cache_dict
    
    def tokenizer_encode_string(self, content: str) -> List[int]:
        output_buffer_init_len = 1024
        if "tokenizer_encode_string__output_buffer" not in dir(self.thread_local_obj) or self.thread_local_obj.tokenizer_encode_string__output_buffer is None:
            self.thread_local_obj.tokenizer_encode_string__output_buffer = (ctypes.c_int * output_buffer_init_len)()

        buffer = self.thread_local_obj.tokenizer_encode_string__output_buffer
        buffer_len = len(buffer)
        result_len = fastllm_lib.token_encode_string(self.model, content.encode(), buffer_len, buffer)
        if result_len > buffer_len:
            if result_len > 10240:
                # 要处理的数据过长，使用一次性的buffer
                temp_buffer = (ctypes.c_int * result_len)()
                ret = fastllm_lib.token_encode_string(self.model, content.encode(), result_len, temp_buffer)
                return [i for i in temp_buffer]
            else:
                # 扩展buffer大小
                new_buffer_len = round(math.ceil(result_len / 1024.0)) * 1024
                buffer = (ctypes.c_int * new_buffer_len)()
                self.thread_local_obj.tokenizer_encode_string__output_buffer = buffer
                result_len = fastllm_lib.token_encode_string(self.model, content.encode(), new_buffer_len, buffer)

        return [buffer[i] for i in range(result_len)]
    
    def encode(self, content: str) -> List[int]:
        return self.tokenizer_encode_string(content)
    
    def tokenizer_decode_token(self, token_id: int) -> bytes:
        if self.tokenizer_decode_token_cache is not None:
            cache_result = self.tokenizer_decode_token_cache.get(token_id)
            if cache_result is not None:
                return cache_result

        output_buffer_init_len = 256
        if "tokenizer_decode_token__output_buffer" not in dir(self.thread_local_obj) or self.thread_local_obj.tokenizer_decode_token__output_buffer is None:
            self.thread_local_obj.tokenizer_decode_token__output_buffer = ctypes.create_string_buffer(output_buffer_init_len)

        buffer = self.thread_local_obj.tokenizer_decode_token__output_buffer
        ret = fastllm_lib.token_decode(self.model, token_id, len(buffer), buffer)
        if ret > 0:
            # buffer长度不够，扩展buffer大小
            new_buffer_len = round(math.ceil(ret / 16.0)) * 16
            buffer = ctypes.create_string_buffer(new_buffer_len)
            self.thread_local_obj.tokenizer_decode_token__output_buffer = buffer
            ret = fastllm_lib.token_decode(self.model, token_id, len(buffer), buffer)
            assert ret == 0

        buffer_bytes = buffer.raw
        result_len = len(buffer_bytes)
        for i in range(len(buffer_bytes)):
            if buffer_bytes[i] == 0:
                result_len = i
                break
        return buffer_bytes[:result_len]

    def stop_token_ctypes(self, stop_token_ids):
        if stop_token_ids is None:
            return 0, None
        else:
            return ctypes.c_int(len(stop_token_ids)), (ctypes.c_int * len(stop_token_ids))(*stop_token_ids)
    
    def trans_conversation(self, conversation: List[Dict[str, str]]) -> List[Dict[str, str]]:
        if (self.get_struct() in ["minimax"]):
            for i in range(len(conversation)):
                if ("content" in conversation[i] and isinstance(conversation[i]["content"], str)):
                    conversation[i]["content"] = [{"type": "text", "text": conversation[i]["content"]}]
        return conversation

    def get_input_token_len(self, conversation: List[Dict[str, str]], add_generation_prompt = True) -> int:
        if (self.hf_tokenizer != None and hasattr(self.hf_tokenizer, "chat_template") and self.hf_tokenizer.chat_template != ""):
            prompt = self.hf_tokenizer.apply_chat_template(self.trans_conversation(conversation), add_generation_prompt = add_generation_prompt, tokenize = False, enable_thinking = self.enable_thinking)
            return len(self.hf_tokenizer.encode(prompt))
        else:
            prompt = self.apply_chat_template(conversation)
            return len(self.encode(prompt))

    def token_healing(self, 
                      prompt: str):
        if (self.hf_tokenizer == None):
            raise("Error: generate_text_token_healing need use transformers tokenizer")
        tokenizer = self.hf_tokenizer
        vocab_size = len(tokenizer.get_vocab())
        if (not hasattr(self, "vocab_cache")):
            self.vocab_cache = []
            self.max_vocab_len = 0
            for i in range(vocab_size):
                self.vocab_cache.append(tokenizer.decode([i]))
                self.max_vocab_len = max(self.max_vocab_len, len(self.vocab_cache[-1]))
        cur_set = set(range(vocab_size))
        logits = list(range(vocab_size))
        array = (ctypes.c_float * (vocab_size * 4))(*logits)

        common_token = tokenizer.encode(prompt)
        for i in range(1, len(prompt) - 1):
            cur_prompt = prompt[:-i]
            right_prompt = prompt[len(cur_prompt):]
            if right_prompt != "":
                for id in list(cur_set):
                    if self.vocab_cache[id].find(right_prompt) == -1:
                        cur_set.remove(id)
            if (len(cur_set) == 0):
                break
            cur_token = tokenizer.encode(cur_prompt)
            for l in range(len(common_token)):
                if (l >= len(cur_token) or cur_token[l] != common_token[l]):
                    common_token = common_token[:l]
                    break

        last_ret = ""
        max_prob = -1e5
        cur_set = set(range(vocab_size))
        for i in range(len(prompt) - 1):
            cur_prompt = prompt[:-i] if i > 0 else prompt
            right_prompt = prompt[len(cur_prompt):]
            if right_prompt != "":
                for id in list(cur_set):
                    if self.vocab_cache[id].find(right_prompt) == -1:
                        cur_set.remove(id)
            if (len(cur_set) == 0):
                break

            cur_prob = 0.0
            real_input = tokenizer.encode(cur_prompt)
            for idx in range(len(common_token), len(real_input) + 1):
                input = real_input[ : idx]
                stop_token_len, stop_token_list = self.stop_token_ctypes(None)
                handle = fastllm_lib.launch_response_llm_model(self.model, len(input), (ctypes.c_int * len(input))(*input),
                                                            1, 0, False, 1, 1, 1, 1, True, stop_token_len, stop_token_list)
                ret = fastllm_lib.fetch_response_logits_llm_model(self.model, handle, array)
                out = list(array)[:vocab_size]
                softmax(out)
                if (idx < len(real_input)):
                    cur_prob += math.log(out[real_input[idx]])
                while True:
                    if (ret <= -1):
                        break
                    ret = fastllm_lib.fetch_response_logits_llm_model(self.model, handle, array)
            max_id = -1
            for i in cur_set:
                if max_id == -1 or out[i] > out[max_id]:
                    if (self.vocab_cache[i].startswith(right_prompt)):
                        max_id = i
            if max_id != -1:
                input.append(max_id)
                cur_prob += math.log(out[max_id])
                if cur_prob > max_prob:
                    max_prob = max_prob
                    last_ret = tokenizer.decode(input)
                #print(math.exp(cur_prob))
                #print(tokenizer.decode(input))
        return last_ret
        
    def response_logits(self,
                        query: str,
                        history: List[Tuple[str, str]] = None,
                        tokenizer = None,
                        stop_token_ids: List[int] = None,
                        ) -> str:
        prompt = query if self.direct_query else self.get_prompt(query, history);
        stop_token_len, stop_token_list = self.stop_token_ctypes(stop_token_ids)
        if (tokenizer == None and self.hf_tokenizer != None):
            tokenizer = self.hf_tokenizer
        if (tokenizer == None):
            vocab_size = fastllm_lib.get_tokenizer_vocab_size(self.model)
            handle = fastllm_lib.launch_response_str_llm_model(self.model, prompt.encode(),
                                                           ctypes.c_int(1), ctypes.c_int(0), ctypes.c_bool(False), ctypes.c_float(1), ctypes.c_int(1),
                                                           ctypes.c_float(1), ctypes.c_float(1), ctypes.c_bool(True),
                                                           stop_token_len, stop_token_list)
        else:
            input = tokenizer.encode(prompt)
            handle = fastllm_lib.launch_response_llm_model(self.model, len(input), (ctypes.c_int * len(input))(*input),
                                                           1, False, 1, 1, 1, 1, True, stop_token_len, stop_token_list)
            vocab_size = len(tokenizer.get_vocab())
        logits = list(range(vocab_size))
        array = (ctypes.c_float * (vocab_size * 4))(*logits)
        ret = fastllm_lib.fetch_response_logits_llm_model(self.model, handle, array)
        out = list(array)[:vocab_size]
        while (ret != -1):
            ret = fastllm_lib.fetch_response_logits_llm_model(self.model, handle, array)
        return out

    def response(self,
                 query: str,
                 history: List[Tuple[str, str]] = None,
                 max_length: int = 8192, do_sample = True, top_p = 0.8, top_k = 1, temperature = 1.0, repeat_penalty = 1.0,
                 stop_token_ids: List[int] = None) -> str:
        ret = "";
        for i in self.stream_response(query = query,
                                      history = history,
                                      max_length = max_length,
                                      do_sample = do_sample,
                                      top_p = top_p, top_k = top_k,
                                      temperature = temperature,
                                      repeat_penalty = repeat_penalty,
                                      one_by_one = True,
                                      stop_token_ids = stop_token_ids):
            ret += i;
        return ret;

    def stream_response(self,
                        query: Union[str, List[Dict[str, str]]],
                        history: List[Tuple[str, str]] = None,
                        max_length: int = 8192, do_sample = True, top_p = 0.8, top_k = 1, temperature = 1.0, repeat_penalty = 1.0,
                        one_by_one = True, stop_token_ids: List[int] = None, add_generation_prompt = True):
        conversation = None
        if (isinstance(query, List)):
            conversation = query
        if (self.hf_tokenizer != None and hasattr(self.hf_tokenizer, "chat_template") and self.hf_tokenizer.chat_template != ""):
            tokenizer = self.hf_tokenizer
            type = None
            if (hasattr(tokenizer, "name") 
                and tokenizer.name == "GLMTokenizer" 
                and hasattr(tokenizer, "build_chat_input")):
                type = "ChatGLM3"
            if (not(history)):
                history = [];
            if (type == "ChatGLM3"):
                input = tokenizer.build_chat_input(query, history=history)["input_ids"].reshape(-1).tolist()
            else:
                prompt = ""
                if (conversation != None and len(conversation) != 0):
                    prompt = tokenizer.apply_chat_template(self.trans_conversation(conversation), add_generation_prompt = add_generation_prompt, tokenize = False, enable_thinking = self.enable_thinking)
                    #input = tokenizer.apply_chat_template(self.trans_conversation(conversation), add_generation_prompt = add_generation_prompt, tokenize = True)
                else:
                    prompt = query if self.direct_query else self.get_prompt(query, history)
                input = tokenizer.encode(prompt)
                #print("prompt", prompt)

            stop_token_len, stop_token_list = self.stop_token_ctypes(stop_token_ids)
            handle = fastllm_lib.launch_response_llm_model(self.model, len(input), (ctypes.c_int * len(input))(*input),
                                                        max_length, 0, do_sample, top_p, top_k, temperature, repeat_penalty,
                                                        False, stop_token_len, stop_token_list)
            tokens = [];
            while True:
                if not(fastllm_lib.can_fetch_response_llm_model(self.model, handle)):
                    continue
                cur = fastllm_lib.fetch_response_llm_model(self.model, handle)
                if (cur <= -1):
                    break
                tokens.append(cur)
                ret = tokenizer.decode(tokens)
                if (ret.encode().find(b'\xef\xbf\xbd') == -1):
                    tokens.clear()
                    yield ret
                else:
                    yield ""
            if len(tokens) > 0:
                yield tokenizer.decode(tokens)
        else:
            prompt = ""
            if (conversation != None and len(conversation) != 0):
                prompt = self.apply_chat_template(conversation)
            else:
                prompt = query if self.direct_query else self.get_prompt(query, history)
            stop_token_len, stop_token_list = self.stop_token_ctypes(stop_token_ids);
            handle = fastllm_lib.launch_response_str_llm_model(self.model, prompt.encode(),
                                                            ctypes.c_int(max_length), ctypes.c_int(0), ctypes.c_bool(do_sample), ctypes.c_float(top_p), ctypes.c_int(top_k),
                                                            ctypes.c_float(temperature), ctypes.c_float(repeat_penalty), ctypes.c_bool(False),
                                                            stop_token_len, stop_token_list);
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
    
    def add_cache(self,
                        prompt: str):
        if (self.hf_tokenizer != None):
            tokenizer = self.hf_tokenizer
            input = tokenizer.encode(prompt);
            fastllm_lib.add_cache_llm_model(self.model, len(input), (ctypes.c_int * len(input))(*input));
        else:
            print("add_cache failed: need hf_tokenizer.")
            exit(0)
    
    def launch_stream_response(self,
                        query: Union[str, List[Dict[str, str]]],
                        history: List[Tuple[str, str]] = None,
                        max_length: int = 8192, min_length: int = 0, do_sample = True, 
                        top_p = 0.8, top_k = 1, temperature = 1.0, repeat_penalty = 1.0,
                        one_by_one = True, stop_token_ids: List[int] = None, add_generation_prompt = True, 
                        images: List = None):
        conversation = None
        if (isinstance(query, List)):
            conversation = query
            conversation = self.trans_conversation(conversation)
        if (images != None):
            architecture = ""
            try:
                architecture = self.config["architectures"][0]
            except:
                print("Error: can't detect architectures for this model.")
                exit(0)
            if (architecture == "CogVLMForCausalLM"):
                image_channels = int(self.config["vision_config"]["in_channels"])
                image_size = int(self.config["vision_config"]["image_size"])
                configs = {
                    "image_channels": image_channels,  
                    "image_height": image_size,
                    "image_width": image_size
                }
                des = json.dumps(configs)
                from torchvision import transforms
                transform = transforms.Compose(
                    [
                        transforms.Resize(
                            (image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC
                        ),
                        transforms.ToTensor(),
                        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
                    ]
                )
                image = transform(images[0]).reshape([-1]).tolist()
            else:
                print("Error: can't support architectures: " + architecture)
                exit(0)

            # 有图片输入，多模态模型
            tokenizer = self.hf_tokenizer
            prompt = ""
            if (conversation != None and len(conversation) != 0):
                prompt = tokenizer.apply_chat_template(self.trans_conversation(conversation), add_generation_prompt = add_generation_prompt, tokenize = False, enable_thinking = self.enable_thinking)
            else:
                prompt = query if self.direct_query else self.get_prompt(query, history)
            input = tokenizer.encode(prompt)
            stop_token_len, stop_token_list = self.stop_token_ctypes(stop_token_ids)
            handle = fastllm_lib.launch_response_llm_model_multimodal(self.model, len(input), (ctypes.c_int * len(input))(*input),
                                                        des.encode(), (ctypes.c_float * len(image))(*image),
                                                        max_length, min_length, do_sample, top_p, top_k, temperature, repeat_penalty,
                                                        False, stop_token_len, stop_token_list)
            return handle

        if (self.hf_tokenizer != None and hasattr(self.hf_tokenizer, "chat_template") and self.hf_tokenizer.chat_template != ""):
            tokenizer = self.hf_tokenizer
            type = None
            if (hasattr(tokenizer, "name") 
                and tokenizer.name == "GLMTokenizer" 
                and hasattr(tokenizer, "build_chat_input")):
                type = "ChatGLM3"
            if (not(history)):
                history = [];
            if (type == "ChatGLM3"):
                input = tokenizer.build_chat_input(query, history=history)["input_ids"].reshape(-1).tolist()
            else:
                prompt = ""
                if (conversation != None and len(conversation) != 0):
                    prompt = tokenizer.apply_chat_template(self.trans_conversation(conversation), add_generation_prompt = add_generation_prompt, tokenize = False, enable_thinking = self.enable_thinking)
                else:
                    prompt = query if self.direct_query else self.get_prompt(query, history)
                if (self.save_history):
                    input = self.tokenizer_cache.tokenize_with_cache(tokenizer, prompt)
                else:
                    input = tokenizer.encode(prompt)
                #print("prompt", prompt[:100])
                #print("input", input[:100])

            stop_token_len, stop_token_list = self.stop_token_ctypes(stop_token_ids)
            handle = fastllm_lib.launch_response_llm_model(self.model, len(input), (ctypes.c_int * len(input))(*input),
                                                        max_length, min_length, do_sample, top_p, top_k, temperature, repeat_penalty,
                                                        False, stop_token_len, stop_token_list)
            if (self.save_history):
                self.current_tokenizer_cache[handle] = [[prompt], [input]]
            return handle
        else:
            prompt = ""
            if (conversation != None and len(conversation) != 0):
                prompt = self.apply_chat_template(conversation)
            else:
                prompt = query if self.direct_query else self.get_prompt(query, history)
            stop_token_len, stop_token_list = self.stop_token_ctypes(stop_token_ids)
            handle = fastllm_lib.launch_response_str_llm_model(self.model, prompt.encode(),
                                                            ctypes.c_int(max_length), ctypes.c_int(min_length), ctypes.c_bool(do_sample), ctypes.c_float(top_p), ctypes.c_int(top_k),
                                                            ctypes.c_float(temperature), ctypes.c_float(repeat_penalty), ctypes.c_bool(False),
                                                            stop_token_len, stop_token_list)
            return handle
    
    def abort_handle(self, handle):
        # print("into force abort")
        if (self.save_history):
            try:
                cur = self.current_tokenizer_cache.pop(handle)
                self.tokenizer_cache.add(cur[0], cur[1])
            except:
                pass
        fastllm_lib.abort_response_llm_model(self.model, handle)
    
    def stream_response_handle(self, handle):
        if (self.hf_tokenizer != None and hasattr(self.hf_tokenizer, "chat_template") and self.hf_tokenizer.chat_template != ""):
            tokenizer = self.hf_tokenizer
            tokens = []
            while True:
                if not(fastllm_lib.can_fetch_response_llm_model(self.model, handle)):
                    continue
                cur = fastllm_lib.fetch_response_llm_model(self.model, handle)
                if (cur <= -1):
                    if (cur == -2):
                        yield "prompt too long"
                    break
                tokens.append(cur)
                ret = tokenizer.decode(tokens)
                if (ret.encode().find(b'\xef\xbf\xbd') == -1):
                    tokens.clear()
                    yield ret
                else:
                    yield ""
            if len(tokens) > 0:
                yield tokenizer.decode(tokens)
        else:
            res = ""
            ret = b''
            fail_cnt = 0
            while True:
                if not(fastllm_lib.can_fetch_response_llm_model(self.model, handle)):
                    continue
                ret += fastllm_lib.fetch_response_str_llm_model(self.model, handle)
                cur = ""
                try:
                    cur = ret.decode()
                    ret = b''
                except:
                    fail_cnt += 1
                    if (fail_cnt == 20):
                        break
                    else:
                        continue
                fail_cnt = 0
                if (cur == "<flmeos>"):
                    break
                yield cur

    async def stream_response_handle_async(self, handle):
        if (self.hf_tokenizer != None and hasattr(self.hf_tokenizer, "chat_template") and self.hf_tokenizer.chat_template != ""):
            tokenizer = self.hf_tokenizer
            tokens = []
            while True:
                if not(fastllm_lib.can_fetch_response_llm_model(self.model, handle)):
                    await asyncio.sleep(0)
                    continue
                cur = fastllm_lib.fetch_response_llm_model(self.model, handle)
                if (cur <= -1):
                    if (cur == -2):
                        yield "prompt too long"
                    if (self.save_history):
                        try:
                            cur = self.current_tokenizer_cache.pop(handle)
                            self.tokenizer_cache.add(cur[0], cur[1])
                        except:
                            pass
                    break
                tokens.append(cur)
                ret = tokenizer.decode(tokens)
                if (ret.encode().find(b'\xef\xbf\xbd') == -1):
                    if (self.save_history and handle in self.current_tokenizer_cache):
                        self.current_tokenizer_cache[handle][0].append(ret)
                        self.current_tokenizer_cache[handle][1].append([] + tokens)
                    tokens.clear()
                    yield ret
                else:
                    yield ""
            if len(tokens) > 0:
                if (self.save_history and handle in self.current_tokenizer_cache):
                    self.current_tokenizer_cache[handle][0].append(tokenizer.decode(tokens))
                    self.current_tokenizer_cache[handle][1].append([] + tokens)
                yield tokenizer.decode(tokens)
        else:
            res = ""
            ret = b''
            fail_cnt = 0
            while True:
                if not(fastllm_lib.can_fetch_response_llm_model(self.model, handle)):
                    await asyncio.sleep(0)
                    continue
                ret += fastllm_lib.fetch_response_str_llm_model(self.model, handle)
                cur = ""
                try:
                    cur = ret.decode()
                    ret = b''
                except:
                    fail_cnt += 1
                    if (fail_cnt == 20):
                        break
                    else:
                        continue
                fail_cnt = 0
                if (cur == "<flmeos>"):
                    break
                yield cur
                    
    async def stream_response_async(self,
                        query: Union[str, List[Dict[str, str]]],
                        history: List[Tuple[str, str]] = None,
                        max_length: int = 8192, do_sample = True, top_p = 0.8, top_k = 1, temperature = 1.0, repeat_penalty = 1.0,
                        one_by_one = True, stop_token_ids: List[int] = None, add_generation_prompt = True):
        handle = self.launch_stream_response(query, history, max_length, 0, do_sample, top_p, top_k, temperature, 
                                             repeat_penalty, one_by_one, stop_token_ids, add_generation_prompt)
        async for ret in self.stream_response_handle_async(handle):
            yield ret

    def stream_response_raw(self,
                            input_tokens: List[int],
                            max_length: int = 8192, do_sample = True, top_p = 0.8, top_k = 1, temperature = 1.0, repeat_penalty = 1.0,
                            one_by_one = True,
                            stop_token_ids: List[int] = None
                            ):
        stop_token_len, stop_token_list = self.stop_token_ctypes(stop_token_ids)
        handle = fastllm_lib.launch_response_llm_model(self.model, len(input_tokens),
                                                       (ctypes.c_int * len(input_tokens))(*input_tokens),
                                                       ctypes.c_int(max_length), ctypes.c_int(0), ctypes.c_bool(do_sample), ctypes.c_float(top_p), ctypes.c_int(top_k),
                                                       ctypes.c_float(temperature), ctypes.c_float(repeat_penalty), ctypes.c_bool(False),
                                                       stop_token_len, stop_token_list)

        # 可能遇到长尾char需要多个token才能够生成，所以只返回bytes，string.decode策略交给外部
        # 方便统计输出token数量，和控制不完整utf8时候解码的逻辑

        total_bytes = b''
        while True:
            cur_token = fastllm_lib.fetch_response_llm_model(self.model, handle)
            if cur_token <= -1:
                break

            cur_bytes = self.tokenizer_decode_token(cur_token)

            if one_by_one:
                yield cur_bytes
            else:
                total_bytes += cur_bytes
                yield total_bytes

    def chat(self, tokenizer, query: str, history: List[Tuple[str, str]] = None, max_length: int = 8192,
             do_sample = True, top_p = 0.8, top_k = 1, temperature = 1.0, repeat_penalty = 1.0, stop_token_ids: List[int] = None, **kwargs):
        if (not(history)):
            history = [];
        prompt = query if self.direct_query else self.get_prompt(query, history);
        input = tokenizer.encode(prompt);
        stop_token_len, stop_token_list = self.stop_token_ctypes(stop_token_ids)
        handle = fastllm_lib.launch_response_llm_model(self.model, len(input), (ctypes.c_int * len(input))(*input),
                                                       max_length, 0, do_sample, top_p, top_k, temperature, repeat_penalty,
                                                       False, stop_token_len, stop_token_list);

        result = [];
        while True:
            cur = fastllm_lib.fetch_response_llm_model(self.model, handle);
            if (cur <= -1):
                break;
            result.append(cur);
        response = tokenizer.decode(result);
        history = history + [(query, response)];
        return response, history
    
    def stream_chat(self, tokenizer, query: str, history: List[Tuple[str, str]] = None, past_key_values = None,
                    max_length: int = 8192, do_sample = True, top_p = 0.8, top_k = 1, temperature = 1.0, repeat_penalty = 1.0,
                    return_past_key_values = False, stop_token_ids: List[int] = None, **kwargs) -> str:
        type = None
        if (hasattr(tokenizer, "name") 
            and tokenizer.name == "GLMTokenizer" 
            and hasattr(tokenizer, "build_chat_input")):
            type = "ChatGLM3"

        if (not(history)):
            history = [];
        
        if (type == "ChatGLM3"):
            input = tokenizer.build_chat_input(query, history=history)["input_ids"].reshape(-1).tolist()
        else:
            prompt = query if self.direct_query else self.get_prompt(query, history);
            input = tokenizer.encode(prompt);

        stop_token_len, stop_token_list = self.stop_token_ctypes(stop_token_ids)
        handle = fastllm_lib.launch_response_llm_model(self.model, len(input), (ctypes.c_int * len(input))(*input),
                                                       max_length, 0, do_sample, top_p, top_k, temperature, repeat_penalty,
                                                       False, stop_token_len, stop_token_list);
        tokens = [];
        while True:
            cur = fastllm_lib.fetch_response_llm_model(self.model, handle);
            if (cur <= -1):
                break;
            tokens.append(cur);
            response = tokenizer.decode(tokens);
            new_history = history + [(query, response)];
            if (type == "ChatGLM3"):
                new_history = history
                new_history.append({"role": "user", "content": query})
                new_history.append({"role": "assistant", "metadata": "", "content": response})
            if return_past_key_values:
                yield response, new_history, None;
            else:
                yield response, new_history;

    def set_adapter(self, name: str):
        fastllm_lib.set_adapter(self.model, str(name).encode())
    
    def disable_adapter(self):
        fastllm_lib.disable_adapter(self.model)
    
    def release_memory(self):
        fastllm_lib.release_memory(self.model)
    
    def set_save_history(self, save: bool):
        self.save_history = True
        fastllm_lib.set_save_history(self.model, save)

    def set_atype(self, atype: str):
        fastllm_lib.set_model_atype(self.model, str(atype).encode())

    def set_moe_experts(self, experts: int):
        fastllm_lib.set_moe_experts(self.model, experts)
    
    def set_cuda_shared_expert(self, cuda_shared_expert: bool):
        fastllm_lib.set_cuda_shared_expert(cuda_shared_expert)

    def set_kv_cache_limit(self, limit: str):
        limit_bytes = 0
        try:
            if (limit.endswith('k') or limit.endswith('K')):
                limit_bytes = float(limit[:-1]) * 1e3
            elif (limit.endswith('m') or limit.endswith('M')):
                limit_bytes = float(limit[:-1]) * 1e6
            elif (limit.endswith('g') or limit.endswith('G')):
                limit_bytes = float(limit[:-1]) * 1e9
            else:
                limit_bytes = float(limit[:-1])
        except:
            print('set_kv_cache_limit error, param should be like "10k" or "10m" or "1g"')
            exit(0)
        limit_bytes = int(limit_bytes)
        fastllm_lib.set_kv_cache_limit_llm_model(self.model, ctypes.c_int64(limit_bytes))
    
    def set_max_batch(self, batch: int):
        fastllm_lib.set_max_batch_llm_model(self.model, batch)
    
    def set_verbose(self, verbose: int):
        fastllm_lib.set_verbose_llm_model(self.model, verbose)
    
    def get_max_input_len(self):
        return fastllm_lib.get_max_input_len_llm_model(self.model)

    def get_struct(self):
        return fastllm_lib.get_struct_llm_model(self.model).decode()

    def embedding_sentence(self, input: str, normalize = True):
        embedding_len = ctypes.c_int(0)
        if (self.hf_tokenizer != None):
            input_ids = self.hf_tokenizer(input,  padding = True, truncation = True)['input_ids']
            embedding_c_float = fastllm_lib.embedding_tokens(self.model, len(input_ids), (ctypes.c_int * len(input_ids))(*input_ids), normalize, embedding_len)
        else:
            embedding_c_float = fastllm_lib.embedding_sentence(self.model, input.encode(), normalize, embedding_len)
        embedding = []
        for i in range(embedding_len.value):
            embedding.append(embedding_c_float[i])
            #print("{:.7f}".format(embedding[i]), end=" ")
        return embedding
    
    def reranker_compute_score(self, pairs: List):
        batch = len(pairs)
        seq_lens = []
        tokens = []
        for i in range(batch):
            input_ids = self.hf_tokenizer(pairs[i : i + 1], padding = True, truncation = True)['input_ids'][0]
            seq_lens.append(len(input_ids))
            tokens += input_ids
        ret_c = fastllm_lib.reranker_compute_score(self.model, batch, (ctypes.c_int * len(seq_lens))(*seq_lens), (ctypes.c_int * len(tokens))(*tokens))
        ret = []
        for i in range(batch):
            ret.append(ret_c[i])
        return ret

def GraphNode(name: str,
              type: str = "data",
              value = None):
    dict = {"name": name, "type": type}
    if value:
        dict["value"] = value
    return dict

class GraphWeight:
    def __init__ (self, pre = ""):
        self.pre = pre

    def __getitem__ (self, index):
        return GraphWeight(self.pre + str(index))

    def to_json(self):
        return GraphNode(self.pre, type = "weight")
    
class GraphData:
    def __init__ (self, pre = ""):
        self.pre = pre

    def __getitem__ (self, index):
        return GraphData(self.pre + str(index))

    def to_json(self):
        return GraphNode(self.pre)

def FloatGraphNode(v):
    if (isinstance(v, int) or isinstance(v, str)):
        v = float(v)
    if (isinstance(v, float)):
        return GraphNode("", "constant.float", v)
    if (isinstance(v, Dict)):
        v["type"] = "config.float"
    return v
    
def IntGraphNode(v):
    if (isinstance(v, float)):
        v = int(v)
    if (isinstance(v, int)):
        return GraphNode("", "constant.int", v)
    if (isinstance(v, Dict)):
        v["type"] = "config.int"
    return v

class ComputeGraph:
    def __init__ (self):
        self.weight = GraphWeight()
        self.data = GraphData()
        self.graph = []
    
    def __str__(self):
        output = {"graph": self.graph}
        if (hasattr(self, "config")):
            output["config"] = self.config
        if (hasattr(self, "tokenizer_config")):
            output["tokenizer_config"] = self.tokenizer_config
        if (hasattr(self, "generation_config")):
            output["generation_config"] = self.generation_config
        return json.dumps(output, indent = 4, default = lambda x: x.to_json())
    
    def Print(self, input):
        self.graph.append({"type": "Print", 
                           "nodes": {"input": input}})
    
    def Exit(self):
        self.graph.append({"type": "Exit"})
    
    def AddTo(self, input0, input1, alpha = 1.0):
        self.graph.append({"type": "AddTo", 
                           "nodes": {"input0": input0, "input1": input1, "alpha": FloatGraphNode(alpha)}})
    
    def DataTypeAs(self, input, input1):
        self.graph.append({"type": "DataTypeAs", 
                           "nodes": {"input": input, "input1": input1}})
        
    def Embedding(self, input, weight, output):
        self.graph.append({"type": "Embedding", 
                           "nodes": {"input": input, "weight": weight, "output": output}})
    
    def ExpandHead(self, input, headDim):
        self.graph.append({"type": "ExpandHeads", 
                           "nodes": {"input": input, "headDim": IntGraphNode(headDim)}})
    
    def FusedAttention(self, q, k, v, curk, curv, original, mask, output, seqLens, 
                       scale, maskType = 0, unitLen = 128):
        self.graph.append({"type": "FusedAttention", 
                           "nodes": {"q": q, "k": k, "v": v, "curk": curk, "curv": curv, 
                                    "original": original, "mask": mask, "output": output, "seqLens": seqLens, 
                                     "scale": FloatGraphNode(scale), 
                                     "maskType": IntGraphNode(maskType), "unitLen": IntGraphNode(unitLen)}})
        
    def Linear(self, input, weight, bias, output):
        self.graph.append({"type": "Linear", 
                           "nodes": {"input": input, "weight": weight, "bias": bias, "output": output}})
    
    def LlamaRotatePosition2D(self, input, positionIds, sin, cos, rotaryDim):
        self.graph.append({"type": "LlamaRotatePosition2D", 
                           "nodes": {"input": input, "positionIds": positionIds, "sin": sin, "cos": cos, "rotaryDim": IntGraphNode(rotaryDim)}})
    
    def MulTo(self, input0, input1):
        self.graph.append({"type": "MulTo", 
                           "nodes": {"input0": input0, "input1": input1}})
        
    def RMSNorm(self, input, weight, eps, output):
        self.graph.append({"type": "RMSNorm", 
                           "nodes": {"input": input, "weight": weight, "eps": FloatGraphNode(eps), "output": output}})
    
    def Silu(self, input, output):
        self.graph.append({"type": "Silu", 
                           "nodes": {"input": input, "output": output}})
    
    def SplitLastTokenStates(self, input, seqLens, output):
        self.graph.append({"type": "SplitLastTokenStates", 
                           "nodes": {"input": input, "output": output, "seqLens": seqLens}})
