package com.fastllm.app;

import android.app.Activity;
import android.content.Intent;
import android.graphics.Color;
import android.graphics.drawable.GradientDrawable;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.provider.Settings;
import android.util.TypedValue;
import android.view.Gravity;
import android.view.View;
import android.view.ViewGroup;
import android.widget.FrameLayout;
import android.widget.LinearLayout;
import android.widget.TextView;
import android.widget.Toast;

import java.util.ArrayList;
import java.util.List;

public class MainActivity extends Activity {

    static final int REQ_PERM = 1001;
    static final int REQ_FOLDER = 1002;
    static final int REQ_MARKET = 1003;

    private FrameLayout contentContainer;
    private LinearLayout tabBar;

    private final List<TabEntry> tabs = new ArrayList<>();
    private int currentTabIndex = -1;

    private ModelRepoPage modelRepoPage;
    private ChatPage chatPage;

    private static class TabEntry {
        String label;
        View page;
        TextView tabView;
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setBackgroundColor(0xFFF5F5F5);

        // Top bar
        LinearLayout topBar = new LinearLayout(this);
        topBar.setOrientation(LinearLayout.HORIZONTAL);
        topBar.setBackgroundColor(0xFF1976D2);
        topBar.setPadding(dp(16), dp(10), dp(16), dp(10));
        topBar.setGravity(Gravity.CENTER_VERTICAL);

        TextView title = new TextView(this);
        title.setText("Fastllm");
        title.setTextColor(Color.WHITE);
        title.setTextSize(TypedValue.COMPLEX_UNIT_SP, 20);
        title.setTypeface(null, android.graphics.Typeface.BOLD);
        topBar.addView(title, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT));

        root.addView(topBar, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT));

        contentContainer = new FrameLayout(this);
        root.addView(contentContainer, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, 0, 1));

        // Bottom tab bar
        View divider = new View(this);
        divider.setBackgroundColor(0xFFDDDDDD);
        root.addView(divider, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, 1));

        tabBar = new LinearLayout(this);
        tabBar.setOrientation(LinearLayout.HORIZONTAL);
        tabBar.setBackgroundColor(Color.WHITE);
        tabBar.setGravity(Gravity.CENTER);
        root.addView(tabBar, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, dp(52)));

        setContentView(root);

        modelRepoPage = new ModelRepoPage(this);
        chatPage = new ChatPage(this);

        addTab("模型仓库", modelRepoPage.getView());
        addTab("本地对话", chatPage.getView());

        selectTab(0);
    }

    private void addTab(String label, View page) {
        TabEntry entry = new TabEntry();
        entry.label = label;
        entry.page = page;

        TextView tv = new TextView(this);
        tv.setText(label);
        tv.setTextSize(TypedValue.COMPLEX_UNIT_SP, 15);
        tv.setGravity(Gravity.CENTER);
        tv.setPadding(dp(16), dp(10), dp(16), dp(10));

        int index = tabs.size();
        tv.setOnClickListener(v -> selectTab(index));

        entry.tabView = tv;
        tabs.add(entry);

        LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.MATCH_PARENT, 1);
        tabBar.addView(tv, lp);

        page.setVisibility(View.GONE);
        contentContainer.addView(page, new FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT));
    }

    private void selectTab(int index) {
        if (index == currentTabIndex) return;
        currentTabIndex = index;

        for (int i = 0; i < tabs.size(); i++) {
            TabEntry entry = tabs.get(i);
            boolean selected = (i == index);
            entry.page.setVisibility(selected ? View.VISIBLE : View.GONE);
            entry.tabView.setTextColor(selected ? 0xFF1976D2 : 0xFF888888);
            entry.tabView.setTypeface(null, selected ?
                    android.graphics.Typeface.BOLD : android.graphics.Typeface.NORMAL);

            if (selected) {
                GradientDrawable indicator = new GradientDrawable();
                indicator.setColor(Color.WHITE);
                entry.tabView.setBackground(indicator);
            } else {
                entry.tabView.setBackground(null);
            }
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        chatPage.onDestroy();
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if (requestCode == REQ_PERM) {
            modelRepoPage.onPermissionResult();
            return;
        }
        if (requestCode == REQ_FOLDER && resultCode == RESULT_OK && data != null && data.getData() != null) {
            modelRepoPage.onFolderSelected(data.getData());
            return;
        }
        if (requestCode == REQ_MARKET && resultCode == RESULT_OK) {
            modelRepoPage.refreshList();
            return;
        }
    }

    void requestStoragePermission() {
        if (Build.VERSION.SDK_INT >= 30 && !Environment.isExternalStorageManager()) {
            try {
                Intent i = new Intent(Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION,
                        Uri.parse("package:" + getPackageName()));
                startActivityForResult(i, REQ_PERM);
            } catch (Exception e) {
                Intent i = new Intent(Settings.ACTION_MANAGE_ALL_FILES_ACCESS_PERMISSION);
                startActivityForResult(i, REQ_PERM);
            }
            return;
        }
        modelRepoPage.onPermissionResult();
    }

    void notifyModelsChanged() {
        chatPage.refreshModelList();
    }

    int dp(int v) {
        return (int) (v * getResources().getDisplayMetrics().density);
    }

    void toast(String s) {
        Toast.makeText(this, s, Toast.LENGTH_SHORT).show();
    }
}
