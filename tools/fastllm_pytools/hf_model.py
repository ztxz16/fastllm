from fastllm_pytools import llm;
import torch;
import ctypes;
import numpy as np;

fastllm_data_type_dict = {
    "int4": 8,
    "int8": 3,
    "float16": 7
}
fastllm_weight_type_dict = {
    "linear": 1,
    "embedding": 2,
    "QuantizedLinear": 111
}

def create(model,
           tokenizer = None,
           pre_prompt = None,
           user_role = None,
           bot_role = None,
           history_sep = None,
           dtype = "float16"):
    if (dtype not in fastllm_data_type_dict):
        print("dtype should in ", list(fastllm_data_type_dict.keys()));
        exit(0);

    # 0.1 model info
    modelInfo = model.config.__dict__
    if (pre_prompt):
        modelInfo["pre_prompt"] = pre_prompt;
    if (user_role):
        modelInfo["user_role"] = user_role;
    if (bot_role):
        modelInfo["bot_role"] = bot_role;
    if (history_sep):
        modelInfo["history_sep"] = history_sep;
    if (modelInfo["model_type"] == "baichuan" and hasattr(model, "model") and hasattr(model.model, "get_alibi_mask")):
        # Baichuan 2代
        modelInfo["use_alibi"] = "1";
        modelInfo["pre_prompt"] = "";
        modelInfo["user_role"] = tokenizer.decode([model.generation_config.user_token_id]);
        modelInfo["bot_role"] = tokenizer.decode([model.generation_config.assistant_token_id]);
        modelInfo["history_sep"] = "";

    weight_type_dict = {};
    module_dict = {};
    weight_bits = {};
    for key, m in model.named_modules():
        if (str(type(m)).find("QuantizedLinear") != -1):
            weight_type_dict[key + ".weight"] = "QuantizedLinear";
            weight_bits[key + ".weight"] = m.weight_bit_width;
        if (isinstance(m, torch.nn.Linear)):
            weight_type_dict[key + ".weight"] = "linear";
            module_dict[key + ".weight"] = m;
        if (isinstance(m, torch.nn.Embedding)):
            weight_type_dict[key] = "embedding";

    model = model.cpu();
    dict = model.state_dict();
    model_type = model.config.__dict__["model_type"];
    model = llm.fastllm_lib.create_empty_llm_model(model_type.encode());
    for it in modelInfo.keys():
        llm.fastllm_lib.add_dict_llm_model(model, str(it).encode(), str(modelInfo[it]).encode());

    # 1. vocab
    if (tokenizer):
        if (hasattr(tokenizer, "sp_model")):
            piece_size = tokenizer.sp_model.piece_size();
            for i in range(piece_size):
                llm.fastllm_lib.add_tokenizer_word_llm_model(model, tokenizer.sp_model.id_to_piece(i).encode(), i);
        else:
            vocab = tokenizer.get_vocab();
            for v in vocab.keys():
                llm.fastllm_lib.add_tokenizer_word_llm_model(model, v.encode(), vocab[v]);
    tot = 0;
    for key in dict:
        ori_data_type = 0;
        ori_np_data_type = np.float32;
        cur_weight_type = 0;
        if (key in weight_type_dict and weight_type_dict[key] in fastllm_weight_type_dict):
            cur_weight_type = fastllm_weight_type_dict[weight_type_dict[key]];
        to_data_type = 0;

        if (cur_weight_type == 1):
            to_data_type = fastllm_data_type_dict[dtype];
            if (to_data_type == 7):
                ori_data_type = 7;
                ori_np_data_type = np.float16;
        elif (cur_weight_type == 2):
            # TODO bfloat
            to_data_type = 0;

        if (cur_weight_type == 111):
            llm.fastllm_lib.add_qlinear_weight_llm_model(model, key.encode(),
                                                 len(dict[key].shape),
                                                 (ctypes.c_int * len(dict[key].shape))(*list(dict[key].shape)),
                                                 weight_bits[key],
                                                 dict[key + "_scale"].numpy().astype(np.float32).ctypes.data_as(ctypes.c_void_p),
                                                 dict[key].numpy().ctypes.data_as(ctypes.c_void_p));
        else:
            llm.fastllm_lib.add_weight_llm_model(model, key.encode(),
                                             len(dict[key].shape),
                                             (ctypes.c_int * len(dict[key].shape))(*list(dict[key].shape)),
                                             to_data_type, cur_weight_type, ori_data_type,
                                             dict[key].numpy().astype(ori_np_data_type).ctypes.data_as(ctypes.c_void_p));
        tot += 1;
        print("convert (", tot, "/", len(dict), end = " )\r");

    print("");
    llm.fastllm_lib.init_params_llm_model(model);
    llm.fastllm_lib.warmup_llm_model(model);
    ret = llm.model("", id = model);
    return ret;

