#include <cstdio>
#include <ctime>
#ifdef USE_CUDA
#include "fastllm-cuda.cuh"
#endif

float exponential_rand() {
	float val = ((float)rand())/RAND_MAX;
	return -log(val);
}

extern "C" {
    DLL_EXPORT long long* t2s_decode(char *model_path, int src_len, int y_len, 
    	void *xy_pos_pt, void **k_cache_pt_list, void **v_cache_pt_list, long long* y_pt, float *pe_list) {
    	using namespace fastllm;
        static bool init_done = false;
        static std::unique_ptr<basellm> model;
        if (!init_done) {
        	printf("fastllm: load t2s_model's safetensors from path: %s\n", model_path);
            model =  CreateLLMModelFromHF(model_path, DataType::FLOAT32, -1, true, "", "", true);
            init_done = true;
        }

        const int num_blocks = 24;
        const int num_heads = 16;
        const int bsz = 1;
        const int len = 1;
        const int dim = 512;
        const float repetition_penalty = 1.35;
        const int top_k_num = 20;
        const int eos_token = 1024;
        WeightMap &weight = model->weight;

		Data xy_pos(DataType::FLOAT16, {bsz, len, dim}, DataDevice::CUDA, (void *)xy_pos_pt);
		Data x(xy_pos);
        std::vector<Data> past_keys, past_values;
        past_keys.reserve(num_blocks);
        past_values.reserve(num_blocks);
        for (int i = 0; i < num_blocks; i++) {
        	past_keys.push_back(Data(DataType::FLOAT16, {bsz, src_len, dim}, DataDevice::CUDA, k_cache_pt_list[i]));
        	past_values.push_back(Data(DataType::FLOAT16, {bsz, src_len, dim}, DataDevice::CUDA, v_cache_pt_list[i]));
        }
        std::vector<long long> y_vec(y_pt, y_pt + y_len);
        y_len -= 1;

        for (int i = 0; i < num_blocks; i++) {
        	past_keys[i].Reshape({bsz, src_len, num_heads, -1});
        	PermuteSelf(past_keys[i], {0, 2, 1, 3});
        	past_values[i].Reshape({bsz, src_len, num_heads, -1});
        	PermuteSelf(past_values[i], {0, 2, 1, 3});
        	past_keys[i].Reshape({bsz * num_heads, src_len, -1});
        	past_values[i].Reshape({bsz * num_heads, src_len, -1});
        	past_keys[i].Expansion({bsz * num_heads, src_len + 1510, past_keys[i].dims[2]});
        	past_values[i].Expansion({bsz * num_heads, src_len + 1510, past_values[i].dims[2]});
        }

        Data qkv0, q0, k0, v0;
        Data k1, v1, attn_weight, attn_out, attn_linear;
        Data x_norm1, x_mlp1, x_relu, x_mlp2, x_norm2;
        Data xy_dec, logits1, logits2, topk;
        std::string predict_weight_name = "model.ar_predict_layer.weight";
        std::string emb_weight_name = "model.ar_audio_embedding.word_embeddings.weight"; // [1025, 512]
        std::string ad_alpha_weight_name = "model.ar_audio_position.alpha"; // [1, 1] = 3.480469
        const float position_alpha = 3.480469;
        Data &emb_weight = weight[emb_weight_name];
        emb_weight.ToDevice(DataDevice::CPU);

        for (int idx = 1; idx < 1500; idx ++) {
	        for(int bk= 0; bk < num_blocks; bk++) {
	        	std::string qkv_weight_name   = "model.h.layers." + std::to_string(bk) + ".self_attn.in_proj_weight";
	        	std::string qkv_bias_name     = "model.h.layers." + std::to_string(bk) + ".self_attn.in_proj_bias";
	        	std::string out_weight_name   = "model.h.layers." + std::to_string(bk) + ".self_attn.out_proj.weight";
	        	std::string out_bias_name     = "model.h.layers." + std::to_string(bk) + ".self_attn.out_proj.bias";
	        	std::string norm1_weight_name = "model.h.layers." + std::to_string(bk) + ".norm1.weight";
	        	std::string norm1_bias_name   = "model.h.layers." + std::to_string(bk) + ".norm1.bias";
	        	std::string norm2_weight_name = "model.h.layers." + std::to_string(bk) + ".norm2.weight";
	        	std::string norm2_bias_name   = "model.h.layers." + std::to_string(bk) + ".norm2.bias";
	        	std::string ln1_weight_name   = "model.h.layers." + std::to_string(bk) + ".linear1.weight";
	        	std::string ln1_bias_name     = "model.h.layers." + std::to_string(bk) + ".linear1.bias";
	        	std::string ln2_weight_name   = "model.h.layers." + std::to_string(bk) + ".linear2.weight";
	        	std::string ln2_bias_name     = "model.h.layers." + std::to_string(bk) + ".linear2.bias";

	    		ToDataType(x, FLOAT16);
	    		ToDataType(weight[qkv_weight_name], DataType::FLOAT16);
	        	Linear(x, weight[qkv_weight_name], weight[qkv_bias_name], qkv0); // [1, 1, 1536]
	        	Split(qkv0, 2, 0,     dim,   q0); // [1, 1, 512]
	        	Split(qkv0, 2, dim,   dim*2, k0);
	        	Split(qkv0, 2, dim*2, dim*3, v0);

	        	Data &k_cache = past_keys[bk], &v_cache = past_values[bk]; // [1, src_len + idx, 512]
	        	k0.Reshape({bsz, k0.dims[1], num_heads, -1});
	        	v0.Reshape({bsz, v0.dims[1], num_heads, -1});
	        	PermuteSelf(k0, {0, 2, 1, 3});
	        	PermuteSelf(v0, {0, 2, 1, 3});
	        	k0.Reshape({bsz * num_heads, k0.dims[2], k0.dims[3]});
	        	v0.Reshape({bsz * num_heads, v0.dims[2], v0.dims[3]});
	        	int kv_len = k_cache.dims[1] + k0.dims[1];
	        	CatDirect(k_cache, k0, 1);
	        	CatDirect(v_cache, v0, 1);

	        	int q_len = q0.dims[1];
	        	q0.Reshape({bsz, q_len, num_heads, -1});
	        	PermuteSelf(q0, {0, 2, 1, 3});
	        	q0.Reshape({bsz * num_heads, q_len, -1});
	        	Attention(q0, k_cache, v_cache, Data(), attn_out, 1, 1.0 / sqrt(q0.dims[2]), 2);
				
	        	attn_out.Reshape({bsz, num_heads, q_len, -1}); // [1, 16, 1, 32]
	        	PermuteSelf(attn_out, {0, 2, 1, 3});
	        	attn_out.Reshape({bsz, q_len, -1});

	        	ToDataType(weight[out_weight_name], DataType::FLOAT16);
	        	Linear(attn_out, weight[out_weight_name], weight[out_bias_name], attn_linear); // [1, 1, 512]

	        	AddTo(x, attn_linear);
	        	ToDataType(x, DataType::FLOAT32); // LayerNorm 在 F16 下会出错，另外对 F16 似乎并没有优化
	        	LayerNorm(x, weight[norm1_weight_name], weight[norm1_bias_name], 2, x_norm1);

	        	Linear(x_norm1, weight[ln1_weight_name], weight[ln1_bias_name], x_mlp1);
	        	Relu(x_mlp1, x_relu);

	        	Linear(x_relu, weight[ln2_weight_name], weight[ln2_bias_name], x_mlp2);
	        	AddTo(x_norm1, x_mlp2);
	        	LayerNorm(x_norm1, weight[norm2_weight_name], weight[norm2_bias_name], 2, x_norm2); // [1, 1, 512]
	        	x.CopyFrom(x_norm2);
	    	}

	    	Split(x, 1, x.dims[1]-1, x.dims[1], xy_dec);
	    	xy_dec.Reshape({bsz, xy_dec.dims[2]});
	    	Linear(xy_dec, weight[predict_weight_name], Data(), logits1); // [1, 1025]
	    	Data *lgt;
	    	if (idx < 11) {
	    		Split(logits1, 1, 0, logits1.dims[1]-1, logits2);
	    		lgt = &logits2;
	    	}
	    	else
	    		lgt = &logits1;
	    	lgt->ToDevice(DataDevice::CPU);
	    	ToDataType(*lgt, DataType::FLOAT32);
	    	float *lgt_float = (float*)(lgt->cpuData);
	    	for (int i = 0; i < y_len + idx; i++) 
	    		if (idx >= 11 || y_vec[i] != eos_token) {
		    		float &val = lgt_float[y_vec[i]];
		    		val = val < 0 ? val * repetition_penalty : val / repetition_penalty;
		    	}
		    TopK(*lgt, topk, top_k_num);
	  		int idx_next = 0, max_val = -99999999;
	    	topk.ToDevice(DataDevice::CPU);
	    	for (int i = 0; i < top_k_num; i++) {
	    		float val = ((float*)topk.cpuData)[i*2 + 1] / exponential_rand();
	    		if (val > max_val) {
	    			max_val = val;
	    			idx_next = (int)(((float*)topk.cpuData)[i*2] + 0.5);
	    		}
	    	}
        	y_vec.push_back(idx_next);
	  		if (idx_next == eos_token)
	  			break;

	  		x.ToDevice(DataDevice::CPU);
	  		ToDataType(x, DataType::FLOAT32);
			for (int i = 0; i < dim; i++)
				((float*)(x.cpuData))[i] = ((float*)(emb_weight.cpuData))[idx_next * dim + i] 
					+ position_alpha * pe_list[(y_len + idx) * dim + i];
    	}

    	long long *res = new long long[y_vec.size() + 1];
    	res[0] = y_vec.size();
    	memcpy(res + 1, y_vec.data(), y_vec.size() * sizeof(long long));
        return res;
    }
}