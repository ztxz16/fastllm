//
//  LLMChat.h
//  LLMChat
//
//  Created by 胡其斌 on 2023/5/18.
//

#ifdef __cplusplus
extern "C" {
#endif
typedef void(* RuntimeResult)(int index,const char* content);

int initGptConf(int model,const char* modelPath,int threads);
int chat(const char* prompt, RuntimeResult chatCallback);
void uninitLLM();

#ifdef __cplusplus
}
#endif
