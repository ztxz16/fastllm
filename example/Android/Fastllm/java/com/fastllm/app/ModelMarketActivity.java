package com.fastllm.app;

import android.app.Activity;
import android.app.AlertDialog;
import android.graphics.Color;
import android.graphics.drawable.GradientDrawable;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.util.TypedValue;
import android.view.Gravity;
import android.view.View;
import android.view.ViewGroup;
import android.view.Window;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import android.widget.ScrollView;
import android.widget.TextView;
import android.widget.Toast;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.URL;

public class ModelMarketActivity extends Activity {

    private final Handler handler = new Handler(Looper.getMainLooper());
    private volatile boolean downloading = false;
    private volatile boolean cancelDownload = false;

    private static class MarketModel {
        String name;
        String description;
        String sizeText;
        String downloadUrl;
        String fileName;

        MarketModel(String name, String description, String sizeText,
                    String downloadUrl, String fileName) {
            this.name = name;
            this.description = description;
            this.sizeText = sizeText;
            this.downloadUrl = downloadUrl;
            this.fileName = fileName;
        }
    }

    private static final MarketModel[] MODELS = {
        new MarketModel(
            "Qwen2.5-0.5B-Instruct (Q4_K_M)",
            "通义千问2.5 0.5B，超轻量级模型，适合手机端快速体验",
            "~400MB",
            "https://modelscope.cn/models/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/master/qwen2.5-0.5b-instruct-q4_k_m.gguf",
            "qwen2.5-0.5b-instruct-q4_k_m.gguf"
        ),
        new MarketModel(
            "Qwen2.5-1.5B-Instruct (Q4_K_M)",
            "通义千问2.5 1.5B，轻量级模型，手机端流畅运行",
            "~1.0GB",
            "https://modelscope.cn/models/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/master/qwen2.5-1.5b-instruct-q4_k_m.gguf",
            "qwen2.5-1.5b-instruct-q4_k_m.gguf"
        ),
        new MarketModel(
            "Qwen2.5-3B-Instruct (Q4_K_M)",
            "通义千问2.5 3B，平衡性能与质量，推荐手机端使用",
            "~2.0GB",
            "https://modelscope.cn/models/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/master/qwen2.5-3b-instruct-q4_k_m.gguf",
            "qwen2.5-3b-instruct-q4_k_m.gguf"
        ),
        new MarketModel(
            "Qwen3-0.6B (Q4_K_M)",
            "通义千问3 0.6B，最新一代超轻量模型",
            "~500MB",
            "https://modelscope.cn/models/Qwen/Qwen3-0.6B-GGUF/resolve/master/qwen3-0.6b-q4_k_m.gguf",
            "qwen3-0.6b-q4_k_m.gguf"
        ),
        new MarketModel(
            "Qwen3-1.7B (Q4_K_M)",
            "通义千问3 1.7B，新一代轻量模型",
            "~1.2GB",
            "https://modelscope.cn/models/Qwen/Qwen3-1.7B-GGUF/resolve/master/qwen3-1.7b-q4_k_m.gguf",
            "qwen3-1.7b-q4_k_m.gguf"
        ),
        new MarketModel(
            "Qwen3-4B (Q4_K_M)",
            "通义千问3 4B，高质量模型，需较大内存",
            "~2.6GB",
            "https://modelscope.cn/models/Qwen/Qwen3-4B-GGUF/resolve/master/qwen3-4b-q4_k_m.gguf",
            "qwen3-4b-q4_k_m.gguf"
        ),
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        if (Build.VERSION.SDK_INT >= 21) {
            Window window = getWindow();
            window.setStatusBarColor(0xFF4834DF);
        }

        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setBackgroundColor(0xFFF8F9FD);

        LinearLayout topBar = new LinearLayout(this);
        topBar.setOrientation(LinearLayout.HORIZONTAL);
        topBar.setPadding(dp(6), dp(12), dp(20), dp(12));
        topBar.setGravity(Gravity.CENTER_VERTICAL);

        GradientDrawable topBarBg = new GradientDrawable(
                GradientDrawable.Orientation.TL_BR,
                new int[]{0xFF6C63FF, 0xFF4834DF});
        topBar.setBackground(topBarBg);
        topBar.setElevation(dp(4));

        TextView backBtn = new TextView(this);
        backBtn.setText("←");
        backBtn.setTextColor(Color.WHITE);
        backBtn.setTextSize(TypedValue.COMPLEX_UNIT_SP, 22);
        backBtn.setPadding(dp(12), dp(4), dp(12), dp(4));
        backBtn.setOnClickListener(v -> {
            if (downloading) {
                toast("请等待下载完成或取消下载");
                return;
            }
            finish();
        });
        topBar.addView(backBtn, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT));

        LinearLayout titleGroup = new LinearLayout(this);
        titleGroup.setOrientation(LinearLayout.VERTICAL);

        TextView title = new TextView(this);
        title.setText("🛒 模型市场");
        title.setTextColor(Color.WHITE);
        title.setTextSize(TypedValue.COMPLEX_UNIT_SP, 20);
        title.setTypeface(null, android.graphics.Typeface.BOLD);
        title.setLetterSpacing(0.02f);
        titleGroup.addView(title);

        TextView titleSub = new TextView(this);
        titleSub.setText("来自 ModelScope 的精选模型");
        titleSub.setTextColor(0xBBFFFFFF);
        titleSub.setTextSize(TypedValue.COMPLEX_UNIT_SP, 12);
        titleGroup.addView(titleSub);

        topBar.addView(titleGroup, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT));

        root.addView(topBar, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT));

        LinearLayout infoBanner = new LinearLayout(this);
        infoBanner.setOrientation(LinearLayout.HORIZONTAL);
        infoBanner.setGravity(Gravity.CENTER_VERTICAL);
        infoBanner.setPadding(dp(16), dp(12), dp(16), dp(12));

        GradientDrawable bannerBg = new GradientDrawable();
        bannerBg.setCornerRadius(dp(12));
        bannerBg.setColor(0xFFF0EFFF);
        infoBanner.setBackground(bannerBg);

        TextView infoIcon = new TextView(this);
        infoIcon.setText("ℹ️");
        infoIcon.setTextSize(TypedValue.COMPLEX_UNIT_SP, 16);
        infoBanner.addView(infoIcon);

        TextView infoText = new TextView(this);
        infoText.setText("选择模型后将自动下载到本地存储");
        infoText.setTextColor(0xFF6C63FF);
        infoText.setTextSize(TypedValue.COMPLEX_UNIT_SP, 13);
        LinearLayout.LayoutParams infoTextLp = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT);
        infoTextLp.leftMargin = dp(8);
        infoBanner.addView(infoText, infoTextLp);

        LinearLayout.LayoutParams bannerLp = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT);
        bannerLp.setMargins(dp(16), dp(12), dp(16), dp(4));
        root.addView(infoBanner, bannerLp);

        ScrollView scrollView = new ScrollView(this);
        LinearLayout listContainer = new LinearLayout(this);
        listContainer.setOrientation(LinearLayout.VERTICAL);
        listContainer.setPadding(dp(16), dp(8), dp(16), dp(16));

        for (MarketModel model : MODELS) {
            listContainer.addView(createModelCard(model));
        }

        scrollView.addView(listContainer, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT));
        root.addView(scrollView, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, 0, 1));

        setContentView(root);
    }

    private View createModelCard(MarketModel model) {
        LinearLayout card = new LinearLayout(this);
        card.setOrientation(LinearLayout.HORIZONTAL);
        card.setPadding(dp(14), dp(14), dp(14), dp(14));

        GradientDrawable cardBg = new GradientDrawable();
        cardBg.setCornerRadius(dp(14));
        cardBg.setColor(Color.WHITE);
        card.setBackground(cardBg);
        card.setElevation(dp(2));

        LinearLayout.LayoutParams cardLp = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT);
        cardLp.bottomMargin = dp(10);
        card.setLayoutParams(cardLp);

        TextView modelIcon = new TextView(this);
        modelIcon.setText("🧠");
        modelIcon.setTextSize(TypedValue.COMPLEX_UNIT_SP, 26);
        modelIcon.setGravity(Gravity.CENTER);
        LinearLayout.LayoutParams iconLp = new LinearLayout.LayoutParams(dp(48), dp(48));
        iconLp.gravity = Gravity.TOP;

        GradientDrawable iconBg = new GradientDrawable();
        iconBg.setCornerRadius(dp(12));
        iconBg.setColor(0xFFF0EFFF);
        modelIcon.setBackground(iconBg);

        card.addView(modelIcon, iconLp);

        LinearLayout infoCol = new LinearLayout(this);
        infoCol.setOrientation(LinearLayout.VERTICAL);
        LinearLayout.LayoutParams infoLp = new LinearLayout.LayoutParams(
                0, ViewGroup.LayoutParams.WRAP_CONTENT, 1);
        infoLp.leftMargin = dp(12);

        TextView nameView = new TextView(this);
        nameView.setText(model.name);
        nameView.setTextSize(TypedValue.COMPLEX_UNIT_SP, 15);
        nameView.setTextColor(0xFF1A1A2E);
        nameView.setTypeface(null, android.graphics.Typeface.BOLD);
        infoCol.addView(nameView);

        TextView descView = new TextView(this);
        descView.setText(model.description);
        descView.setTextSize(TypedValue.COMPLEX_UNIT_SP, 13);
        descView.setTextColor(0xFF6E7191);
        LinearLayout.LayoutParams descLp = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT);
        descLp.topMargin = dp(4);
        infoCol.addView(descView, descLp);

        LinearLayout bottomRow = new LinearLayout(this);
        bottomRow.setOrientation(LinearLayout.HORIZONTAL);
        bottomRow.setGravity(Gravity.CENTER_VERTICAL);
        LinearLayout.LayoutParams bottomLp = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT);
        bottomLp.topMargin = dp(10);

        TextView sizeBadge = new TextView(this);
        sizeBadge.setText("💾 " + model.sizeText);
        sizeBadge.setTextSize(TypedValue.COMPLEX_UNIT_SP, 11);
        sizeBadge.setTextColor(0xFF6C63FF);
        GradientDrawable sizeBadgeBg = new GradientDrawable();
        sizeBadgeBg.setCornerRadius(dp(8));
        sizeBadgeBg.setColor(0xFFF0EFFF);
        sizeBadge.setBackground(sizeBadgeBg);
        sizeBadge.setPadding(dp(8), dp(3), dp(8), dp(3));
        bottomRow.addView(sizeBadge);

        View spacer = new View(this);
        bottomRow.addView(spacer, new LinearLayout.LayoutParams(0, 0, 1));

        TextView dlBtn = new TextView(this);
        dlBtn.setText("⬇ 下载");
        dlBtn.setTextColor(Color.WHITE);
        dlBtn.setTextSize(TypedValue.COMPLEX_UNIT_SP, 13);
        dlBtn.setTypeface(null, android.graphics.Typeface.BOLD);
        dlBtn.setGravity(Gravity.CENTER);
        dlBtn.setPadding(dp(18), dp(8), dp(18), dp(8));

        GradientDrawable btnBg = new GradientDrawable(
                GradientDrawable.Orientation.LEFT_RIGHT,
                new int[]{0xFF00BFA5, 0xFF009688});
        btnBg.setCornerRadius(dp(10));
        dlBtn.setBackground(btnBg);

        dlBtn.setOnClickListener(v -> {
            if (downloading) {
                toast("已有下载任务进行中");
                return;
            }
            confirmDownload(model);
        });

        bottomRow.addView(dlBtn, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT));

        infoCol.addView(bottomRow, bottomLp);
        card.addView(infoCol, infoLp);

        return card;
    }

    private void confirmDownload(MarketModel model) {
        File dir = getExternalFilesDir("models");
        if (dir == null) {
            toast("无法访问存储");
            return;
        }
        File target = new File(dir, model.fileName);
        if (target.exists()) {
            new AlertDialog.Builder(this)
                    .setTitle("文件已存在")
                    .setMessage(model.fileName + " 已存在，是否重新下载？")
                    .setPositiveButton("重新下载", (d, w) -> startDownload(model, target))
                    .setNeutralButton("直接添加", (d, w) -> {
                        addToStorage(model, target);
                        toast("模型已添加");
                    })
                    .setNegativeButton("取消", null)
                    .show();
            return;
        }
        startDownload(model, target);
    }

    private void startDownload(MarketModel model, File target) {
        downloading = true;
        cancelDownload = false;

        AlertDialog.Builder builder = new AlertDialog.Builder(this);
        builder.setCancelable(false);

        LinearLayout dlLayout = new LinearLayout(this);
        dlLayout.setOrientation(LinearLayout.VERTICAL);
        dlLayout.setPadding(dp(24), dp(20), dp(24), dp(12));

        TextView dlTitle = new TextView(this);
        dlTitle.setText("⬇ 下载中");
        dlTitle.setTextSize(TypedValue.COMPLEX_UNIT_SP, 18);
        dlTitle.setTextColor(0xFF1A1A2E);
        dlTitle.setTypeface(null, android.graphics.Typeface.BOLD);
        dlLayout.addView(dlTitle);

        TextView dlModel = new TextView(this);
        dlModel.setText(model.name);
        dlModel.setTextSize(TypedValue.COMPLEX_UNIT_SP, 14);
        dlModel.setTextColor(0xFF6E7191);
        LinearLayout.LayoutParams dlModelLp = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT);
        dlModelLp.topMargin = dp(4);
        dlLayout.addView(dlModel, dlModelLp);

        ProgressBar progressBar = new ProgressBar(this, null,
                android.R.attr.progressBarStyleHorizontal);
        progressBar.setMax(100);
        progressBar.setProgress(0);
        LinearLayout.LayoutParams pbLp = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT);
        pbLp.topMargin = dp(16);
        dlLayout.addView(progressBar, pbLp);

        TextView progressText = new TextView(this);
        progressText.setText("准备下载...");
        progressText.setTextSize(TypedValue.COMPLEX_UNIT_SP, 13);
        progressText.setTextColor(0xFF6E7191);
        progressText.setGravity(Gravity.CENTER);
        LinearLayout.LayoutParams ptLp = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT);
        ptLp.topMargin = dp(8);
        dlLayout.addView(progressText, ptLp);

        TextView percentText = new TextView(this);
        percentText.setText("0%");
        percentText.setTextSize(TypedValue.COMPLEX_UNIT_SP, 28);
        percentText.setTextColor(0xFF6C63FF);
        percentText.setTypeface(null, android.graphics.Typeface.BOLD);
        percentText.setGravity(Gravity.CENTER);
        LinearLayout.LayoutParams pctLp = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT);
        pctLp.topMargin = dp(8);
        dlLayout.addView(percentText, pctLp);

        builder.setView(dlLayout);
        builder.setNegativeButton("取消下载", (d, w) -> cancelDownload = true);

        AlertDialog dlg = builder.create();
        dlg.show();

        new Thread(() -> {
            File tmpFile = new File(target.getAbsolutePath() + ".tmp");
            boolean success = false;
            try {
                URL url = new URL(model.downloadUrl);
                HttpURLConnection conn = (HttpURLConnection) url.openConnection();
                conn.setConnectTimeout(30000);
                conn.setReadTimeout(30000);
                conn.setRequestProperty("User-Agent", "Fastllm-Android/1.0");
                conn.setInstanceFollowRedirects(true);

                int responseCode = conn.getResponseCode();
                if (responseCode == HttpURLConnection.HTTP_MOVED_TEMP
                        || responseCode == HttpURLConnection.HTTP_MOVED_PERM
                        || responseCode == 307 || responseCode == 308) {
                    String newUrl = conn.getHeaderField("Location");
                    conn.disconnect();
                    conn = (HttpURLConnection) new URL(newUrl).openConnection();
                    conn.setConnectTimeout(30000);
                    conn.setReadTimeout(30000);
                    conn.setRequestProperty("User-Agent", "Fastllm-Android/1.0");
                }

                long total = conn.getContentLengthLong();
                InputStream is = conn.getInputStream();
                FileOutputStream fos = new FileOutputStream(tmpFile);

                byte[] buf = new byte[65536];
                long downloaded = 0;
                int len;
                long lastUpdate = 0;

                while ((len = is.read(buf)) != -1) {
                    if (cancelDownload) break;
                    fos.write(buf, 0, len);
                    downloaded += len;

                    long now = System.currentTimeMillis();
                    if (now - lastUpdate > 300) {
                        lastUpdate = now;
                        long dl = downloaded;
                        int pct = total > 0 ? (int) (dl * 100 / total) : 0;
                        handler.post(() -> {
                            progressBar.setProgress(pct);
                            String dlText = formatSize(dl);
                            String totalText = total > 0 ? formatSize(total) : "未知";
                            progressText.setText(dlText + " / " + totalText);
                            percentText.setText(pct + "%");
                        });
                    }
                }

                fos.close();
                is.close();
                conn.disconnect();

                if (!cancelDownload) {
                    tmpFile.renameTo(target);
                    success = true;
                }
            } catch (Exception e) {
                e.printStackTrace();
            }

            if (!success && !cancelDownload) {
                tmpFile.delete();
            }
            if (cancelDownload) {
                tmpFile.delete();
            }

            boolean ok = success;
            handler.post(() -> {
                downloading = false;
                dlg.dismiss();
                if (ok) {
                    addToStorage(model, target);
                    toast("✅ 下载完成: " + model.name);
                    setResult(RESULT_OK);
                } else if (cancelDownload) {
                    toast("下载已取消");
                } else {
                    toast("❌ 下载失败，请检查网络后重试");
                }
            });
        }).start();
    }

    private void addToStorage(MarketModel model, File file) {
        ModelStorage storage = new ModelStorage(this);
        ModelStorage.ModelInfo info = new ModelStorage.ModelInfo(
                model.name, file.getAbsolutePath(), file.length(),
                System.currentTimeMillis());
        storage.addModel(info);
        setResult(RESULT_OK);
    }

    private String formatSize(long bytes) {
        if (bytes < 1024) return bytes + " B";
        if (bytes < 1024 * 1024) return String.format("%.1f KB", bytes / 1024.0);
        if (bytes < 1024L * 1024 * 1024) return String.format("%.1f MB", bytes / (1024.0 * 1024));
        return String.format("%.2f GB", bytes / (1024.0 * 1024 * 1024));
    }

    private int dp(int v) {
        return (int) (v * getResources().getDisplayMetrics().density);
    }

    private void toast(String s) {
        Toast.makeText(this, s, Toast.LENGTH_SHORT).show();
    }
}
