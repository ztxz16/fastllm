package com.doujiao.core;

import android.support.annotation.Keep;
import android.util.Log;

public class AssistantCore {

    private static AssistantCore instance = null;
    private static runtimeResult mRuntimeRes = null;

    static {
        System.loadLibrary("assistant");
    }

    /*静态对象*/
    public static AssistantCore getInstance(){
        if(instance == null){
            instance = new AssistantCore();
        }

        return instance;
    }

    public String initLLM(String path,runtimeResult callback) {
        mRuntimeRes = callback;
        return initLLMConfig(path,8);
    }

    @Keep
    public void reportChat(String content,int index) {
        Log.d("@@@","recv:"+content+",index:"+index);
        if (mRuntimeRes != null) {
            mRuntimeRes.callbackResult(index,content);
        }
    }

    public interface runtimeResult {
        void callbackResult(int index,String content);
    }

    private native String initLLMConfig(String path,int threads);
    public native int chat(String prompt);
    public native int uninitLLM();
}
