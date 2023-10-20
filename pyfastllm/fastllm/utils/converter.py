import struct

from typing import Any
import numpy as np
import torch

from .writer import Writer
from .quantizer import QuantType

class BaseConverter():
    def __init__(self, model, tokenizer, q_type=0) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.q_type = q_type
    
    def get_model_info(self):
        model_info = self.model.config.__dict__
        if self.model.generation_config is not None:
            model_info.update(self.model.generation_config.__dict__)
        model_info["tokenizer_use_score"] = "1"
        return model_info

    def get_vocab(self, ):
        raise NotImplementedError

    def get_weights(self):
        state_dict = self.model.state_dict()
        if hasattr(self.model, "peft_config"):
            state_dict.keys = [key.replace('base_model.model.', '') for key in state_dict]

        state_dict = {key: val.numpy().astype(np.float32) for key, val in state_dict.items()}
        for name, m in self.model.named_modules():
            if isinstance(m, torch.nn.Linear): 
                if self.q_type == QuantType.FP16:
                    state_dict[name+".weight.fp16"] = state_dict[name+".weight"].astype(np.float16)
                    state_dict.pop(name+".weight")
                elif self.q_type == QuantType.INT8:
                    state_dict[name+".weight.int8"] = state_dict[name+".weight"]
                    state_dict.pop(name+".weight")
                elif self.q_type == QuantType.INT4:
                    state_dict[name+".weight.int4"] = state_dict[name+".weight"]
                    state_dict.pop(name+".weight")

        return state_dict

    def convert_model_info(self, wt:Writer):
        model_info = self.get_model_info()
        model_info = {
            str(key): str(val)
            for key, val in model_info.items()
        }
        wt.write(model_info)

    def convert_tokenizer(self, wt:Writer):
        vocab = self.get_vocab()
        vocab_len = len(vocab)
        wt.write(int(vocab_len))
        for i, key in enumerate(vocab):
            # wt.write(len(key))
            # for c in key: wt.write(int(c))
            wt.write(key)
            wt.write(int(i))
            wt.write(float(vocab[key]))

    def convert_weights(self, wt:Writer):
        state_dict = self.get_weights()
        wt.write(len(state_dict))
        tot = 0
        for name, tensor in state_dict.items():
            print(f"{name} : {tensor.shape}")
            if name.endswith("int4") or name.endswith("int8") or name.endswith("fp16"):
                wt.write(str(name[:-5]))
                wt.write_tensor(tensor, self.q_type)
            else:
                wt.write(str(name))
                wt.write(tensor)

            print("output (", tot, "/", len(state_dict), end = " )\r")
            tot += 1
        print("\nfinish.")
    
    def forward(self, wt:Writer, *args: Any, **kwds: Any) -> Any:
        self.convert_model_info(wt)
        self.convert_tokenizer(wt)
        self.convert_weights(wt)

    def __call__(self, wt:Writer, *args: Any, **kwds: Any) -> Any:
        return self.forward(wt, *args, **kwds)
    
    def dump(self, outpath:str):
        wt = Writer(outpath=outpath)
        # version id
        wt.write(int(2))
        self.forward(wt=wt)


class ChatglmConverter(BaseConverter):
    def get_vocab(self):
        tokenizer = self.tokenizer.tokenizer
        piece_size = tokenizer.sp_model.piece_size()

        vocab = {
            tokenizer.sp_model.id_to_piece(i).encode(): float(tokenizer.sp_model.get_score(i)) for i in range(piece_size)
        }
        return vocab
 
    
class BaichuanConverter(BaseConverter):
    def get_model_info(self, ):
        model_info = super().get_model_info()
        if hasattr(self.model, "model") and hasattr(self.model.model, "get_alibi_mask"):
            model_info.update({
                "use_alibi": "1",
                "pre_prompt": "",
                "user_role": "<FLM_FIX_TOKEN_" + str(self.model.generation_config.user_token_id) + "> ",
                "bot_role": "<FLM_FIX_TOKEN_" + str(self.model.generation_config.assistant_token_id) + ">",
                "history_sep": ""
            })
        return model_info
    
    def get_vocab(self,):
        vocab = self.tokenizer.get_vocab()

        vocab = {
            key.encode(): vocab[key] for key in vocab
        }
        return vocab
    

class QwenConverter(BaseConverter):
    def get_model_info(self,):
        model_info = super().get_model_info()
        if model_info["chat_format"] == "chatml":
            model_info.update({
                "im_end_id": self.tokenizer.im_end_id, 
                "im_start_id": self.tokenizer.im_start_id
            })
        
        return model_info 
    
    def get_vocab(self, ):
        vocab = self.tokenizer.get_vocab()
        vocab = {
            key: 1.0 for key in vocab.keys()
        }
        return vocab


class MossConverter(BaseConverter):
    def get_vocab(self, ):
        tokenizer = self.tokenizer.tokenizer
        vocab = tokenizer.get_vocab()
        vocab = {
            [tokenizer.byte_decoder.get(c, ord(c)) for c in v]: 1.0
            for v in vocab
        }
        
        return vocab
