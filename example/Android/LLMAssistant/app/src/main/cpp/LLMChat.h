//
//  LLMChat.h
//  LLMChat
//
//  Created by 胡其斌 on 2023/5/18.
//

#ifdef __cplusplus
extern "C" {
#endif

#include <android/log.h>
#define  LOG_Debug(...)  __android_log_print(ANDROID_LOG_DEBUG, "Assistant", __VA_ARGS__)

typedef void(* RuntimeResultMobile)(int index,const char* content);

std::string initGptConf(const char* modelPath,int threads);
int chat(const char* prompt, RuntimeResultMobile chatCallback);
void uninitLLM();

#ifdef __cplusplus
}
#endif
