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
        root.setBackgroundColor(MainActivity.COLOR_BG);

        ScrollView scrollView = new ScrollView(activity);
        scrollView.setFillViewport(true);

        LinearLayout inner = new LinearLayout(activity);
        inner.setOrientation(LinearLayout.VERTICAL);
        inner.setPadding(dp(16), dp(16), dp(16), dp(16));

        LinearLayout headerRow = new LinearLayout(activity);
        headerRow.setOrientation(LinearLayout.HORIZONTAL);
        headerRow.setGravity(Gravity.CENTER_VERTICAL);
        headerRow.setPadding(dp(2), 0, 0, dp(12));

        TextView headerIcon = new TextView(activity);
        headerIcon.setText("📦");
        headerIcon.setTextSize(TypedValue.COMPLEX_UNIT_SP, 22);
        headerRow.addView(headerIcon, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT));

        LinearLayout headerTextGroup = new LinearLayout(activity);
        headerTextGroup.setOrientation(LinearLayout.VERTICAL);
        LinearLayout.LayoutParams htgLp = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT);
        htgLp.leftMargin = dp(10);

        TextView header = new TextView(activity);
        header.setText("模型仓库");
        header.setTextSize(TypedValue.COMPLEX_UNIT_SP, 20);
        header.setTextColor(MainActivity.COLOR_TEXT_PRIMARY);
        header.setTypeface(null, android.graphics.Typeface.BOLD);
        headerTextGroup.addView(header);

        TextView headerSub = new TextView(activity);
        headerSub.setText("管理你的本地模型");
        headerSub.setTextSize(TypedValue.COMPLEX_UNIT_SP, 13);
        headerSub.setTextColor(MainActivity.COLOR_TEXT_SECONDARY);
        headerTextGroup.addView(headerSub);

        headerRow.addView(headerTextGroup, htgLp);
        inner.addView(headerRow, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT));

        modelListContainer = new LinearLayout(activity);
        modelListContainer.setOrientation(LinearLayout.VERTICAL);
        inner.addView(modelListContainer, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT));

        scrollView.addView(inner, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT));

        root.addView(scrollView, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, 0, 1));

        LinearLayout btnContainer = new LinearLayout(activity);
        btnContainer.setOrientation(LinearLayout.VERTICAL);
        btnContainer.setBackgroundColor(MainActivity.COLOR_BG);
        btnContainer.setPadding(dp(16), dp(8), dp(16), dp(16));

        TextView addBtn = new TextView(activity);
        addBtn.setText("＋  添加模型");
        addBtn.setTextColor(Color.WHITE);
        addBtn.setTextSize(TypedValue.COMPLEX_UNIT_SP, 16);
        addBtn.setTypeface(null, android.graphics.Typeface.BOLD);
        addBtn.setGravity(Gravity.CENTER);
        addBtn.setPadding(0, dp(14), 0, dp(14));

        GradientDrawable btnBg = new GradientDrawable(
                GradientDrawable.Orientation.LEFT_RIGHT,
                new int[]{MainActivity.COLOR_GRADIENT_START, MainActivity.COLOR_GRADIENT_END});
        btnBg.setCornerRadius(dp(14));
        addBtn.setBackground(btnBg);
        addBtn.setElevation(dp(4));

        addBtn.setOnClickListener(v -> showAddModelDialog());
        btnContainer.addView(addBtn, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT));

        root.addView(btnContainer, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT));

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
            LinearLayout emptyLayout = new LinearLayout(activity);
            emptyLayout.setOrientation(LinearLayout.VERTICAL);
            emptyLayout.setGravity(Gravity.CENTER);
            emptyLayout.setPadding(0, dp(60), 0, dp(60));

            TextView emptyIcon = new TextView(activity);
            emptyIcon.setText("📭");
            emptyIcon.setTextSize(TypedValue.COMPLEX_UNIT_SP, 48);
            emptyIcon.setGravity(Gravity.CENTER);
            emptyLayout.addView(emptyIcon);

            TextView emptyTitle = new TextView(activity);
            emptyTitle.setText("暂无模型");
            emptyTitle.setTextColor(MainActivity.COLOR_TEXT_PRIMARY);
            emptyTitle.setTextSize(TypedValue.COMPLEX_UNIT_SP, 17);
            emptyTitle.setTypeface(null, android.graphics.Typeface.BOLD);
            emptyTitle.setGravity(Gravity.CENTER);
            LinearLayout.LayoutParams etLp = new LinearLayout.LayoutParams(
                    ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT);
            etLp.topMargin = dp(12);
            emptyLayout.addView(emptyTitle, etLp);

            TextView emptyDesc = new TextView(activity);
            emptyDesc.setText("点击下方按钮添加本地模型或从模型市场下载");
            emptyDesc.setTextColor(MainActivity.COLOR_TEXT_SECONDARY);
            emptyDesc.setTextSize(TypedValue.COMPLEX_UNIT_SP, 14);
            emptyDesc.setGravity(Gravity.CENTER);
            LinearLayout.LayoutParams edLp = new LinearLayout.LayoutParams(
                    ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT);
            edLp.topMargin = dp(6);
            emptyLayout.addView(emptyDesc, edLp);

            modelListContainer.addView(emptyLayout, new LinearLayout.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT));
            return;
        }

        for (ModelStorage.ModelInfo model : models) {
            modelListContainer.addView(createModelCard(model));
        }
    }

    private View createModelCard(ModelStorage.ModelInfo model) {
        LinearLayout card = new LinearLayout(activity);
        card.setOrientation(LinearLayout.HORIZONTAL);
        card.setPadding(dp(14), dp(14), dp(14), dp(14));

        GradientDrawable cardBg = new GradientDrawable();
        cardBg.setCornerRadius(dp(14));
        cardBg.setColor(MainActivity.COLOR_SURFACE);
        card.setBackground(cardBg);
        card.setElevation(dp(2));

        LinearLayout.LayoutParams cardLp = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT);
        cardLp.bottomMargin = dp(10);
        card.setLayoutParams(cardLp);

        TextView modelIcon = new TextView(activity);
        modelIcon.setText("🤖");
        modelIcon.setTextSize(TypedValue.COMPLEX_UNIT_SP, 28);
        modelIcon.setGravity(Gravity.CENTER);
        LinearLayout.LayoutParams iconLp = new LinearLayout.LayoutParams(dp(44), dp(44));
        iconLp.gravity = Gravity.CENTER_VERTICAL;

        GradientDrawable iconBg = new GradientDrawable();
        iconBg.setCornerRadius(dp(12));
        iconBg.setColor(0xFFF0EFFF);
        modelIcon.setBackground(iconBg);

        card.addView(modelIcon, iconLp);

        LinearLayout infoCol = new LinearLayout(activity);
        infoCol.setOrientation(LinearLayout.VERTICAL);
        LinearLayout.LayoutParams infoLp = new LinearLayout.LayoutParams(
                0, ViewGroup.LayoutParams.WRAP_CONTENT, 1);
        infoLp.leftMargin = dp(12);
        infoLp.gravity = Gravity.CENTER_VERTICAL;

        TextView nameView = new TextView(activity);
        nameView.setText(model.name);
        nameView.setTextSize(TypedValue.COMPLEX_UNIT_SP, 15);
        nameView.setTextColor(MainActivity.COLOR_TEXT_PRIMARY);
        nameView.setTypeface(null, android.graphics.Typeface.BOLD);
        nameView.setSingleLine(true);
        infoCol.addView(nameView, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT));

        TextView pathView = new TextView(activity);
        pathView.setText(model.path);
        pathView.setTextSize(TypedValue.COMPLEX_UNIT_SP, 11);
        pathView.setTextColor(MainActivity.COLOR_TEXT_SECONDARY);
        pathView.setSingleLine(true);
        LinearLayout.LayoutParams pathLp = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT);
        pathLp.topMargin = dp(2);
        infoCol.addView(pathView, pathLp);

        LinearLayout metaRow = new LinearLayout(activity);
        metaRow.setOrientation(LinearLayout.HORIZONTAL);
        metaRow.setGravity(Gravity.CENTER_VERTICAL);
        LinearLayout.LayoutParams metaLp = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT);
        metaLp.topMargin = dp(6);

        TextView sizeBadge = new TextView(activity);
        sizeBadge.setText("💾 " + model.getSizeText());
        sizeBadge.setTextSize(TypedValue.COMPLEX_UNIT_SP, 11);
        sizeBadge.setTextColor(MainActivity.COLOR_PRIMARY);
        GradientDrawable sizeBadgeBg = new GradientDrawable();
        sizeBadgeBg.setCornerRadius(dp(8));
        sizeBadgeBg.setColor(0xFFF0EFFF);
        sizeBadge.setBackground(sizeBadgeBg);
        sizeBadge.setPadding(dp(8), dp(3), dp(8), dp(3));
        metaRow.addView(sizeBadge);

        infoCol.addView(metaRow, metaLp);
        card.addView(infoCol, infoLp);

        TextView deleteBtn = new TextView(activity);
        deleteBtn.setText("🗑");
        deleteBtn.setTextSize(TypedValue.COMPLEX_UNIT_SP, 18);
        deleteBtn.setGravity(Gravity.CENTER);
        LinearLayout.LayoutParams delLp = new LinearLayout.LayoutParams(dp(38), dp(38));
        delLp.gravity = Gravity.CENTER_VERTICAL;

        GradientDrawable delBg = new GradientDrawable();
        delBg.setCornerRadius(dp(10));
        delBg.setColor(0xFFFFF0F0);
        deleteBtn.setBackground(delBg);

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
        card.addView(deleteBtn, delLp);

        return card;
    }

    private void showAddModelDialog() {
        new AlertDialog.Builder(activity)
                .setTitle("添加模型")
                .setItems(new String[]{"📁 添加本地模型", "🛒 从模型市场下载"}, (dialog, which) -> {
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
