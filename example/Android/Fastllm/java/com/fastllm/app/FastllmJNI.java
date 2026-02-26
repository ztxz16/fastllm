package com.fastllm.app;

public class FastllmJNI {
    static { System.loadLibrary("fastllm"); }

    public interface ChatCallback {
        void onToken(int index, String content);
    }

    public static native String initModel(String path, int threads);
    public static native void chat(String input, float topP, int topK,
                                   float temperature, float repeatPenalty,
                                   ChatCallback callback);
    public static native void resetChat();
    public static native void releaseModel();
}
