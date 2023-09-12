#include <jni.h>
#include <string>
#include "LLMChat.h"


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
    jobject bb = env->NewDirectByteBuffer((void *) content, strlen(content));
    jclass cls_Charset = env->FindClass("java/nio/charset/Charset");
    jmethodID mid_Charset_forName = env->GetStaticMethodID(cls_Charset, "forName", "(Ljava/lang/String;)Ljava/nio/charset/Charset;");
    jobject charset = env->CallStaticObjectMethod(cls_Charset, mid_Charset_forName, env->NewStringUTF("UTF-8"));

    jmethodID mid_Charset_decode = env->GetMethodID(cls_Charset, "decode", "(Ljava/nio/ByteBuffer;)Ljava/nio/CharBuffer;");
    jobject cb = env->CallObjectMethod(charset, mid_Charset_decode, bb);
    env->DeleteLocalRef(bb);

    jclass cls_CharBuffer = env->FindClass("java/nio/CharBuffer");
    jmethodID mid_CharBuffer_toString = env->GetMethodID(cls_CharBuffer, "toString", "()Ljava/lang/String;");
    jstring str = static_cast<jstring>(env->CallObjectMethod(cb, mid_CharBuffer_toString));
    env->CallVoidMethod(g_obj, jgetDBpathMethod,str,index);
    env->DeleteLocalRef(javaClass);
    //释放当前线程
    if(mNeedDetach) {
        g_javaVM->DetachCurrentThread();
    }
    env = NULL;
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_doujiao_core_AssistantCore_initLLMConfig(
        JNIEnv* env,
        jobject obj,
        jstring modelpath,
        jint threads) {
    initGvm(env,obj);
    const char *path = env->GetStringUTFChars(modelpath, NULL);
    std::string ret =  initGptConf(path,threads);
    LOG_Debug("@@@initLLMConfig:%s",ret.c_str());
    env->ReleaseStringUTFChars( modelpath, path);
    return env->NewStringUTF(ret.c_str());
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