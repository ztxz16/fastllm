#include <jni.h>
#include <string>
#include <android/log.h>

#include "LLMChat.h"
#define  LOG_Debug(...)  __android_log_print(ANDROID_LOG_DEBUG, "Assistant", __VA_ARGS__)

JavaVM *g_javaVM = NULL;
jobject g_obj;

void initGvm(JNIEnv *env,jobject thiz) {
    if(g_javaVM == NULL) {
        env->GetJavaVM(&g_javaVM);
        g_obj = env->NewGlobalRef(thiz);
    }
}

void chatCb(int index,const char* content) {
    JNIEnv *env = NULL;
    int mNeedDetach = 0;
    //获取当前native线程是否有没有被附加到jvm环境中
    int getEnvStat = g_javaVM->GetEnv((void **)&env,JNI_VERSION_1_6);
    if (getEnvStat == JNI_EDETACHED) {
        //如果没有， 主动附加到jvm环境中，获取到env
        if (g_javaVM->AttachCurrentThread( &env, NULL) != 0) {
            LOG_Debug("Unable to AttachCurrentThread");
            return;
        }
        mNeedDetach = 1;
    }
    //通过全局变量g_obj 获取到要回调的类
    jclass javaClass = env->GetObjectClass(g_obj);//env->FindClass("com/doujiao/core/AssistantCore");//
    if (javaClass == 0) {
        LOG_Debug("Unable to find class");
        if(mNeedDetach) {
            g_javaVM->DetachCurrentThread();
        }
        return;
    }
    jmethodID jgetDBpathMethod = env->GetMethodID(javaClass, "reportChat", "(Ljava/lang/String;I)V");
    if (jgetDBpathMethod == NULL) {
        LOG_Debug("Unable to find method:jgetDBpathMethod");
        return;
    }
    jstring chat = env->NewStringUTF(content);
//    jobject  obj = env->NewObject(javaClass,jgetDBpathMethod);
    env->CallVoidMethod(g_obj, jgetDBpathMethod,chat,index);
    env->DeleteLocalRef(javaClass);
    //释放当前线程
    if(mNeedDetach) {
        g_javaVM->DetachCurrentThread();
    }
    env = NULL;
}

extern "C" JNIEXPORT jint JNICALL
Java_com_doujiao_core_AssistantCore_initLLMConfig(
        JNIEnv* env,
        jobject obj,
        jint modeltype,
        jstring modelpath,
        jint threads) {
    initGvm(env,obj);
    const char *path = env->GetStringUTFChars(modelpath, NULL);
    int ret =  initGptConf(modeltype,path,threads);
    env->ReleaseStringUTFChars( modelpath, path);
    return ret;
}

extern "C" JNIEXPORT jint JNICALL
Java_com_doujiao_core_AssistantCore_chat(
        JNIEnv* env,
        jobject obj,
        jstring prompt) {
    initGvm(env,obj);
    const char *question = env->GetStringUTFChars(prompt, NULL);
    chat(question,[](int index,const char* content){
        chatCb(index,content);
    });
//    chatCb(1,"content");
    return 0;
}

extern "C" JNIEXPORT void JNICALL
Java_com_doujiao_core_AssistantCore_uninitLLM(
        JNIEnv* env,
        jobject /* this */) {
    uninitLLM();
}