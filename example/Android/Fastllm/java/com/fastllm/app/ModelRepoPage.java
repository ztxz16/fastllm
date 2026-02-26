package com.fastllm.app;

import android.app.AlertDialog;
import android.content.Intent;
import android.graphics.Color;
import android.graphics.drawable.GradientDrawable;
import android.net.Uri;
import android.os.Build;
import android.os.Environment;
import android.util.TypedValue;
import android.view.Gravity;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;

import java.io.File;
import java.util.List;

public class ModelRepoPage {

    private final MainActivity activity;
    private final ModelStorage storage;
    private final LinearLayout modelListContainer;
    private final View rootView;

    public ModelRepoPage(MainActivity activity) {
        this.activity = activity;
        this.storage = new ModelStorage(activity);

        LinearLayout root = new LinearLayout(activity);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setBackgroundColor(0xFFF5F5F5);

        ScrollView scrollView = new ScrollView(activity);
        scrollView.setFillViewport(true);

        LinearLayout inner = new LinearLayout(activity);
        inner.setOrientation(LinearLayout.VERTICAL);
        inner.setPadding(dp(12), dp(12), dp(12), dp(12));

        // Header
        TextView header = new TextView(activity);
        header.setText("模型仓库");
        header.setTextSize(TypedValue.COMPLEX_UNIT_SP, 18);
        header.setTextColor(0xFF333333);
        header.setTypeface(null, android.graphics.Typeface.BOLD);
        header.setPadding(dp(4), 0, 0, dp(8));
        inner.addView(header, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT));

        modelListContainer = new LinearLayout(activity);
        modelListContainer.setOrientation(LinearLayout.VERTICAL);
        inner.addView(modelListContainer, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT));

        scrollView.addView(inner, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT));

        root.addView(scrollView, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, 0, 1));

        // Add model button
        Button addBtn = new Button(activity);
        addBtn.setText("+ 添加模型");
        addBtn.setTextColor(Color.WHITE);
        addBtn.setTextSize(TypedValue.COMPLEX_UNIT_SP, 16);
        addBtn.setAllCaps(false);

        GradientDrawable btnBg = new GradientDrawable();
        btnBg.setCornerRadius(dp(8));
        btnBg.setColor(0xFF1976D2);
        addBtn.setBackground(btnBg);

        LinearLayout.LayoutParams btnLp = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, dp(48));
        btnLp.setMargins(dp(12), dp(8), dp(12), dp(12));
        root.addView(addBtn, btnLp);

        addBtn.setOnClickListener(v -> showAddModelDialog());

        this.rootView = root;
        refreshList();
    }

    View getView() {
        return rootView;
    }

    void refreshList() {
        modelListContainer.removeAllViews();
        List<ModelStorage.ModelInfo> models = storage.getModels();

        if (models.isEmpty()) {
            TextView empty = new TextView(activity);
            empty.setText("暂无模型，请点击下方按钮添加");
            empty.setTextColor(0xFF999999);
            empty.setTextSize(TypedValue.COMPLEX_UNIT_SP, 14);
            empty.setGravity(Gravity.CENTER);
            empty.setPadding(0, dp(60), 0, dp(60));
            modelListContainer.addView(empty, new LinearLayout.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT));
            return;
        }

        for (ModelStorage.ModelInfo model : models) {
            modelListContainer.addView(createModelCard(model));
        }
    }

    private View createModelCard(ModelStorage.ModelInfo model) {
        LinearLayout card = new LinearLayout(activity);
        card.setOrientation(LinearLayout.VERTICAL);
        card.setPadding(dp(14), dp(12), dp(14), dp(12));

        GradientDrawable cardBg = new GradientDrawable();
        cardBg.setCornerRadius(dp(8));
        cardBg.setColor(Color.WHITE);
        card.setBackground(cardBg);
        card.setElevation(dp(2));

        LinearLayout.LayoutParams cardLp = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT);
        cardLp.bottomMargin = dp(8);
        card.setLayoutParams(cardLp);

        // Model name
        TextView nameView = new TextView(activity);
        nameView.setText(model.name);
        nameView.setTextSize(TypedValue.COMPLEX_UNIT_SP, 16);
        nameView.setTextColor(0xFF333333);
        nameView.setTypeface(null, android.graphics.Typeface.BOLD);
        card.addView(nameView, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT));

        // Model path
        TextView pathView = new TextView(activity);
        pathView.setText(model.path);
        pathView.setTextSize(TypedValue.COMPLEX_UNIT_SP, 12);
        pathView.setTextColor(0xFF888888);
        pathView.setSingleLine(true);
        LinearLayout.LayoutParams pathLp = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT);
        pathLp.topMargin = dp(4);
        card.addView(pathView, pathLp);

        // Bottom row: size + delete
        LinearLayout bottomRow = new LinearLayout(activity);
        bottomRow.setOrientation(LinearLayout.HORIZONTAL);
        bottomRow.setGravity(Gravity.CENTER_VERTICAL);
        LinearLayout.LayoutParams bottomLp = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT);
        bottomLp.topMargin = dp(6);

        TextView sizeView = new TextView(activity);
        sizeView.setText(model.getSizeText());
        sizeView.setTextSize(TypedValue.COMPLEX_UNIT_SP, 13);
        sizeView.setTextColor(0xFF666666);
        bottomRow.addView(sizeView, new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1));

        TextView deleteBtn = new TextView(activity);
        deleteBtn.setText("删除");
        deleteBtn.setTextSize(TypedValue.COMPLEX_UNIT_SP, 13);
        deleteBtn.setTextColor(0xFFE53935);
        deleteBtn.setPadding(dp(12), dp(4), dp(12), dp(4));
        deleteBtn.setOnClickListener(v -> {
            new AlertDialog.Builder(activity)
                    .setTitle("确认删除")
                    .setMessage("确定要从列表中移除此模型吗？\n（不会删除本地文件）")
                    .setPositiveButton("删除", (d, w) -> {
                        storage.removeModel(model.path);
                        refreshList();
                        activity.notifyModelsChanged();
                    })
                    .setNegativeButton("取消", null)
                    .show();
        });
        bottomRow.addView(deleteBtn, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT));

        card.addView(bottomRow, bottomLp);

        return card;
    }

    private void showAddModelDialog() {
        new AlertDialog.Builder(activity)
                .setTitle("添加模型")
                .setItems(new String[]{"添加本地模型", "从模型市场下载"}, (dialog, which) -> {
                    if (which == 0) {
                        startFolderPicker();
                    } else {
                        Intent intent = new Intent(activity, ModelMarketActivity.class);
                        activity.startActivityForResult(intent, MainActivity.REQ_MARKET);
                    }
                })
                .show();
    }

    private void startFolderPicker() {
        if (Build.VERSION.SDK_INT >= 30 && !Environment.isExternalStorageManager()) {
            activity.requestStoragePermission();
            return;
        }
        openFolderPicker();
    }

    void onPermissionResult() {
        openFolderPicker();
    }

    private void openFolderPicker() {
        Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT_TREE);
        activity.startActivityForResult(intent, MainActivity.REQ_FOLDER);
    }

    void onFolderSelected(Uri treeUri) {
        String folderPath = getFolderPathFromUri(treeUri);
        if (folderPath == null) {
            activity.toast("无法获取文件夹路径");
            return;
        }
        File folder = new File(folderPath);
        if (!folder.exists()) {
            activity.toast("文件夹不存在: " + folderPath);
            return;
        }

        String name = folder.getName();
        long size = ModelStorage.computeDirSize(folder);

        ModelStorage.ModelInfo model = new ModelStorage.ModelInfo(
                name, folderPath, size, System.currentTimeMillis());
        storage.addModel(model);
        refreshList();
        activity.notifyModelsChanged();
        activity.toast("模型已添加: " + name);
    }

    private String getFolderPathFromUri(Uri uri) {
        String docId = null;
        try {
            if (Build.VERSION.SDK_INT >= 21) {
                docId = android.provider.DocumentsContract.getTreeDocumentId(uri);
            }
        } catch (Exception e) {
            return null;
        }
        if (docId == null) return null;
        if (docId.startsWith("primary:")) {
            return Environment.getExternalStorageDirectory().getAbsolutePath()
                    + "/" + docId.substring("primary:".length());
        }
        String[] parts = docId.split(":");
        if (parts.length == 2) {
            File[] externals = activity.getExternalFilesDirs(null);
            if (externals != null) {
                for (File ext : externals) {
                    if (ext == null) continue;
                    String extPath = ext.getAbsolutePath();
                    int idx = extPath.indexOf("/Android/");
                    if (idx > 0) {
                        String storagePath = extPath.substring(0, idx);
                        File candidate = new File(storagePath + "/" + parts[1]);
                        if (candidate.exists()) return candidate.getAbsolutePath();
                    }
                }
            }
        }
        return null;
    }

    private int dp(int v) {
        return activity.dp(v);
    }
}
