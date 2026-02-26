package com.fastllm.chat;

public class FastllmJNI {
    static { System.loadLibrary("fastllm"); }

    public interface ChatCallback {
        void onToken(int index, String content);
    }

    public static native String initModel(String path, int threads);
    public static native void chat(String input, ChatCallback callback);
    public static native void resetChat();
    public static native void releaseModel();
}
