package com.doujiao.xiaozhihuiassistant.utils;

import android.content.Context;
import android.content.SharedPreferences;

public class PrefUtil {
    private static final String SF_NAME = "com.doujiao.llm.config";
    private static final String MOLE_PATH = "llm_path";
    private static SharedPreferences mPref;

    public static void initPref(Context context) {
        if (mPref == null) {
            mPref = context.getSharedPreferences(SF_NAME, Context.MODE_PRIVATE);
        }
    }

    public static void setModelPath(String path) {
        if (mPref != null) {
            mPref.edit().putString(MOLE_PATH,path).apply();
        }
    }

    public static String getModelPath() {
        if (mPref != null) {
            return mPref.getString(MOLE_PATH,"");
        }
        return "";
    }
}
