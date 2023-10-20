from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer

from .quantizer import QuantType
from .converter import ChatglmConverter, BaichuanConverter, QwenConverter, MossConverter

def convert(hf_model_name_or_path:str, save_path:str, q_type=QuantType.INT4):
    config = AutoConfig.from_pretrained(hf_model_name_or_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(hf_model_name_or_path, trust_remote_code=True)

    if "Baichuan" in config.architectures:
        model = AutoModelForCausalLM.from_pretrained(hf_model_name_or_path, trust_remote_code=True).cpu().eval()
        converter = BaichuanConverter(model=model, tokenizer=tokenizer, q_type=q_type)
    elif "ChatGLM" in config.architectures:
        model = AutoModel.from_pretrained(hf_model_name_or_path, trust_remote_code=True).cpu().eval()
        converter = ChatglmConverter(model=model, tokenizer=tokenizer, q_type=q_type)
    elif "Qwen" in config.architectures:
        model = AutoModelForCausalLM.from_pretrained(hf_model_name_or_path, trust_remote_code=True, fp16=True).cpu().eval()
        converter = QwenConverter(model=model, tokenizer=tokenizer, q_type=q_type)
    elif "Moss" in config.architectures:
        model = AutoModelForCausalLM.from_pretrained(hf_model_name_or_path, trust_remote_code=True).cpu().eval()
        converter = MossConverter(model=model, tokenizer=tokenizer, q_type=q_type)
    else:
        raise NotImplementedError(f"Unsupport model: {config.architectures}")
    
    converter.dump(save_path)
