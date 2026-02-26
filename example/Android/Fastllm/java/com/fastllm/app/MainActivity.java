package com.fastllm.app;

import android.app.Activity;
import android.content.Intent;
import android.graphics.Color;
import android.graphics.LinearGradient;
import android.graphics.Shader;
import android.graphics.drawable.GradientDrawable;
import android.graphics.drawable.PaintDrawable;
import android.graphics.drawable.ShapeDrawable;
import android.graphics.drawable.shapes.RectShape;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.provider.Settings;
import android.util.TypedValue;
import android.view.Gravity;
import android.view.View;
import android.view.ViewGroup;
import android.view.Window;
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

    static final int COLOR_PRIMARY = 0xFF6C63FF;
    static final int COLOR_PRIMARY_DARK = 0xFF5A52E0;
    static final int COLOR_ACCENT = 0xFF00BFA5;
    static final int COLOR_BG = 0xFFF8F9FD;
    static final int COLOR_SURFACE = 0xFFFFFFFF;
    static final int COLOR_TEXT_PRIMARY = 0xFF1A1A2E;
    static final int COLOR_TEXT_SECONDARY = 0xFF6E7191;
    static final int COLOR_DIVIDER = 0xFFEEF0F8;
    static final int COLOR_GRADIENT_START = 0xFF6C63FF;
    static final int COLOR_GRADIENT_END = 0xFF4834DF;
    static final int COLOR_USER_BUBBLE = 0xFF6C63FF;
    static final int COLOR_BOT_BUBBLE = 0xFFFFFFFF;
    static final int COLOR_SUCCESS = 0xFF00BFA5;
    static final int COLOR_DANGER = 0xFFFF5252;
    static final int COLOR_TAB_INACTIVE = 0xFFADB5BD;

    private FrameLayout contentContainer;
    private LinearLayout tabBar;

    private final List<TabEntry> tabs = new ArrayList<>();
    private int currentTabIndex = -1;

    private ModelRepoPage modelRepoPage;
    private ChatPage chatPage;

    private static class TabEntry {
        String label;
        String icon;
        View page;
        TextView tabView;
        TextView iconView;
        View indicator;
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        if (Build.VERSION.SDK_INT >= 21) {
            Window window = getWindow();
            window.setStatusBarColor(COLOR_GRADIENT_END);
        }

        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setBackgroundColor(COLOR_BG);

        LinearLayout topBar = new LinearLayout(this);
        topBar.setOrientation(LinearLayout.HORIZONTAL);
        topBar.setPadding(dp(20), dp(14), dp(20), dp(14));
        topBar.setGravity(Gravity.CENTER_VERTICAL);

        GradientDrawable topBarBg = new GradientDrawable(
                GradientDrawable.Orientation.TL_BR,
                new int[]{COLOR_GRADIENT_START, COLOR_GRADIENT_END});
        topBar.setBackground(topBarBg);
        topBar.setElevation(dp(4));

        TextView title = new TextView(this);
        title.setText("⚡ Fastllm");
        title.setTextColor(Color.WHITE);
        title.setTextSize(TypedValue.COMPLEX_UNIT_SP, 22);
        title.setTypeface(null, android.graphics.Typeface.BOLD);
        title.setLetterSpacing(0.02f);
        topBar.addView(title, new LinearLayout.LayoutParams(
                0, ViewGroup.LayoutParams.WRAP_CONTENT, 1));

        TextView versionBadge = new TextView(this);
        versionBadge.setText("v1.0");
        versionBadge.setTextColor(0xBBFFFFFF);
        versionBadge.setTextSize(TypedValue.COMPLEX_UNIT_SP, 12);
        GradientDrawable badgeBg = new GradientDrawable();
        badgeBg.setCornerRadius(dp(10));
        badgeBg.setColor(0x33FFFFFF);
        versionBadge.setBackground(badgeBg);
        versionBadge.setPadding(dp(10), dp(3), dp(10), dp(3));
        topBar.addView(versionBadge, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT));

        root.addView(topBar, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT));

        contentContainer = new FrameLayout(this);
        root.addView(contentContainer, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, 0, 1));

        View divider = new View(this);
        divider.setBackgroundColor(COLOR_DIVIDER);
        root.addView(divider, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, 1));

        tabBar = new LinearLayout(this);
        tabBar.setOrientation(LinearLayout.HORIZONTAL);
        tabBar.setBackgroundColor(COLOR_SURFACE);
        tabBar.setGravity(Gravity.CENTER);
        tabBar.setElevation(dp(8));
        root.addView(tabBar, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, dp(60)));

        setContentView(root);

        modelRepoPage = new ModelRepoPage(this);
        chatPage = new ChatPage(this);

        addTab("📦", "模型仓库", modelRepoPage.getView());
        addTab("💬", "本地对话", chatPage.getView());

        selectTab(0);
    }

    private void addTab(String icon, String label, View page) {
        TabEntry entry = new TabEntry();
        entry.label = label;
        entry.icon = icon;
        entry.page = page;

        LinearLayout tabItem = new LinearLayout(this);
        tabItem.setOrientation(LinearLayout.VERTICAL);
        tabItem.setGravity(Gravity.CENTER);
        tabItem.setPadding(dp(16), dp(6), dp(16), dp(4));

        View indicator = new View(this);
        GradientDrawable indicatorBg = new GradientDrawable();
        indicatorBg.setCornerRadius(dp(2));
        indicator.setBackground(indicatorBg);
        tabItem.addView(indicator, new LinearLayout.LayoutParams(dp(20), dp(3)));

        TextView iconView = new TextView(this);
        iconView.setText(icon);
        iconView.setTextSize(TypedValue.COMPLEX_UNIT_SP, 20);
        iconView.setGravity(Gravity.CENTER);
        LinearLayout.LayoutParams iconLp = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT);
        iconLp.topMargin = dp(2);
        tabItem.addView(iconView, iconLp);

        TextView tv = new TextView(this);
        tv.setText(label);
        tv.setTextSize(TypedValue.COMPLEX_UNIT_SP, 11);
        tv.setGravity(Gravity.CENTER);
        LinearLayout.LayoutParams tvLp = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT);
        tvLp.topMargin = dp(1);
        tabItem.addView(tv, tvLp);

        int index = tabs.size();
        tabItem.setOnClickListener(v -> selectTab(index));

        entry.tabView = tv;
        entry.iconView = iconView;
        entry.indicator = indicator;
        tabs.add(entry);

        LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.MATCH_PARENT, 1);
        tabBar.addView(tabItem, lp);

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
            entry.tabView.setTextColor(selected ? COLOR_PRIMARY : COLOR_TAB_INACTIVE);
            entry.tabView.setTypeface(null, selected ?
                    android.graphics.Typeface.BOLD : android.graphics.Typeface.NORMAL);

            GradientDrawable indicatorBg = (GradientDrawable) entry.indicator.getBackground();
            indicatorBg.setColor(selected ? COLOR_PRIMARY : Color.TRANSPARENT);
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
