#include <jni.h>
#include <string>
#include <chrono>
#include <android/log.h>
#include "model.h"
#include "utils.h"
#include "fastllm.h"

#define TAG "FastllmJNI"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

static std::unique_ptr<fastllm::basellm> g_model = nullptr;
static fastllm::ChatMessages g_messages;

static JavaVM *g_jvm = nullptr;

jint JNI_OnLoad(JavaVM *vm, void *) {
    g_jvm = vm;
    return JNI_VERSION_1_6;
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_fastllm_app_FastllmJNI_initModel(JNIEnv *env, jclass, jstring jpath, jint threads) {
    const char *path = env->GetStringUTFChars(jpath, nullptr);
    LOGI("Loading model: %s threads=%d", path, threads);

    fastllm::SetThreads(threads);
    g_model.reset();
    g_messages.clear();

    std::string p(path);
    env->ReleaseStringUTFChars(jpath, path);

    try {
        bool isHFDir = fastllm::FileExists(p + "/config.json") || fastllm::FileExists(p + "config.json");
        if (isHFDir) {
            LOGI("Detected HF directory, using CreateLLMModelFromHF");
            g_model = fastllm::CreateLLMModelFromHF(p, fastllm::DataType::FLOAT16);
        } else {
            g_model = fastllm::CreateLLMModelFromFile(p);
        }
    } catch (const std::exception &e) {
        LOGE("Model load exception: %s", e.what());
        return env->NewStringUTF("");
    } catch (...) {
        LOGE("Model load unknown exception");
        return env->NewStringUTF("");
    }

    if (g_model) {
        g_model->SetSaveHistoryChat(true);
        LOGI("Model loaded: %s", g_model->model_type.c_str());
        return env->NewStringUTF(g_model->model_type.c_str());
    }
    LOGE("Model load failed");
    return env->NewStringUTF("");
}

extern "C" JNIEXPORT void JNICALL
Java_com_fastllm_app_FastllmJNI_chat(JNIEnv *env, jclass, jstring jinput,
                                      jfloat topP, jint topK, jfloat temperature,
                                      jfloat repeatPenalty, jobject callback) {
    if (!g_model) return;

    const char *s = env->GetStringUTFChars(jinput, nullptr);
    std::string input(s);
    env->ReleaseStringUTFChars(jinput, s);

    g_messages.push_back({"user", input});

    jclass cbClass = env->GetObjectClass(callback);
    jmethodID onToken = env->GetMethodID(cbClass, "onToken", "(ILjava/lang/String;)V");
    jobject gCallback = env->NewGlobalRef(callback);

    fastllm::GenerationConfig config;
    config.top_p = topP;
    config.top_k = topK;
    config.temperature = temperature;
    config.repeat_penalty = repeatPenalty;

    fastllm::ClearProfiler();
    auto chatStartTime = std::chrono::system_clock::now();
    auto lastTokenTime = chatStartTime;
    int tokenCount = 0;

    std::string ret = g_model->Response(
        g_model->ApplyChatTemplate(g_messages),
        [gCallback, onToken, &chatStartTime, &lastTokenTime, &tokenCount](int index, const char *content) {
            JNIEnv *e = nullptr;
            bool detach = false;
            if (g_jvm->GetEnv((void**)&e, JNI_VERSION_1_6) == JNI_EDETACHED) {
                g_jvm->AttachCurrentThread(&e, nullptr);
                detach = true;
            }

            auto now = std::chrono::system_clock::now();

            if (index == 0) {
                float prefillMs = std::chrono::duration<float, std::milli>(now - chatStartTime).count();
                LOGI("[Token #%d] Prefill (first token): %.1f ms", tokenCount, prefillMs);
                fastllm::PrintProfiler();
                fastllm::ClearProfiler();
                lastTokenTime = now;
            } else if (index > 0) {
                float tokenMs = std::chrono::duration<float, std::milli>(now - lastTokenTime).count();
                float tokenTps = tokenMs > 0 ? 1000.0f / tokenMs : 0;
                float totalSec = std::chrono::duration<float>(now - chatStartTime).count();
                float avgTps = totalSec > 0 ? tokenCount / totalSec : 0;
                LOGI("[Token #%d] %.1f ms (cur: %.2f tps, avg: %.2f tps)", tokenCount, tokenMs, tokenTps, avgTps);
                fastllm::PrintProfiler();
                fastllm::ClearProfiler();
                lastTokenTime = now;
            }

            if (index >= 0) tokenCount++;

            if (e) {
                jstring jc = e->NewStringUTF(content);
                e->CallVoidMethod(gCallback, onToken, index, jc);
                e->DeleteLocalRef(jc);
            }
            if (detach) g_jvm->DetachCurrentThread();
        },
        config);

    auto chatEndTime = std::chrono::system_clock::now();
    float totalSec = std::chrono::duration<float>(chatEndTime - chatStartTime).count();
    LOGI("[Summary] %d tokens in %.2f s (avg %.2f tokens/s)", tokenCount, totalSec,
         totalSec > 0 ? tokenCount / totalSec : 0);

    g_messages.push_back({"assistant", ret});
    env->DeleteGlobalRef(gCallback);
}

extern "C" JNIEXPORT void JNICALL
Java_com_fastllm_app_FastllmJNI_resetChat(JNIEnv *, jclass) {
    g_messages.clear();
}

extern "C" JNIEXPORT void JNICALL
Java_com_fastllm_app_FastllmJNI_releaseModel(JNIEnv *, jclass) {
    g_messages.clear();
    g_model.reset();
}
