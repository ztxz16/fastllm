package com.fastllm.app;

import android.content.Context;
import android.content.SharedPreferences;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

public class ModelStorage {

    private static final String PREF_NAME = "fastllm_models";
    private static final String KEY_MODEL_LIST = "model_list";
    private static final String SEP = "\n";
    private static final String FIELD_SEP = "\t";

    private final SharedPreferences prefs;

    public ModelStorage(Context context) {
        prefs = context.getSharedPreferences(PREF_NAME, Context.MODE_PRIVATE);
    }

    public static class ModelInfo {
        public String name;
        public String path;
        public long sizeBytes;
        public long addTime;

        public ModelInfo(String name, String path, long sizeBytes, long addTime) {
            this.name = name;
            this.path = path;
            this.sizeBytes = sizeBytes;
            this.addTime = addTime;
        }

        public String getSizeText() {
            if (sizeBytes < 1024) return sizeBytes + " B";
            if (sizeBytes < 1024 * 1024) return String.format("%.1f KB", sizeBytes / 1024.0);
            if (sizeBytes < 1024L * 1024 * 1024) return String.format("%.1f MB", sizeBytes / (1024.0 * 1024));
            return String.format("%.2f GB", sizeBytes / (1024.0 * 1024 * 1024));
        }

        String serialize() {
            return name + FIELD_SEP + path + FIELD_SEP + sizeBytes + FIELD_SEP + addTime;
        }

        static ModelInfo deserialize(String s) {
            String[] parts = s.split(FIELD_SEP);
            if (parts.length < 4) return null;
            try {
                return new ModelInfo(parts[0], parts[1], Long.parseLong(parts[2]), Long.parseLong(parts[3]));
            } catch (Exception e) {
                return null;
            }
        }
    }

    public List<ModelInfo> getModels() {
        List<ModelInfo> list = new ArrayList<>();
        String raw = prefs.getString(KEY_MODEL_LIST, "");
        if (raw.isEmpty()) return list;
        for (String line : raw.split(SEP)) {
            if (line.isEmpty()) continue;
            ModelInfo info = ModelInfo.deserialize(line);
            if (info != null) list.add(info);
        }
        return list;
    }

    public void addModel(ModelInfo model) {
        List<ModelInfo> models = getModels();
        for (ModelInfo m : models) {
            if (m.path.equals(model.path)) return;
        }
        models.add(model);
        save(models);
    }

    public void removeModel(String path) {
        List<ModelInfo> models = getModels();
        models.removeIf(m -> m.path.equals(path));
        save(models);
    }

    private void save(List<ModelInfo> models) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < models.size(); i++) {
            if (i > 0) sb.append(SEP);
            sb.append(models.get(i).serialize());
        }
        prefs.edit().putString(KEY_MODEL_LIST, sb.toString()).apply();
    }

    public static long computeDirSize(File dir) {
        long size = 0;
        if (dir.isFile()) return dir.length();
        File[] files = dir.listFiles();
        if (files != null) {
            for (File f : files) {
                size += f.isDirectory() ? computeDirSize(f) : f.length();
            }
        }
        return size;
    }
}
