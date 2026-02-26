package com.fastllm.chat;

import android.app.Activity;
import android.content.Intent;
import android.database.Cursor;
import android.graphics.Color;
import android.graphics.drawable.GradientDrawable;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.os.Handler;
import android.os.Looper;
import android.provider.OpenableColumns;
import android.provider.Settings;
import android.text.TextUtils;
import android.util.TypedValue;
import android.view.Gravity;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;
import android.widget.Toast;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;

import android.app.AlertDialog;
import android.util.Log;

public class MainActivity extends Activity {

    private static final String TAG = "FastllmChat";
    private LinearLayout chatContainer;
    private ScrollView scrollView;
    private EditText inputBox;
    private Button sendBtn;
    private Button selectBtn;
    private Button resetBtn;
    private TextView statusText;
    private final Handler handler = new Handler(Looper.getMainLooper());

    private volatile boolean modelLoaded = false;
    private volatile boolean generating = false;
    private TextView currentBotBubble = null;

    private static final int REQ_FILE = 1001;
    private static final int REQ_PERM = 1002;
    private static final int REQ_FOLDER = 1003;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        int dp8 = dp(8), dp12 = dp(12), dp16 = dp(16);

        // Root layout
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setBackgroundColor(0xFFF5F5F5);

        // Top bar
        LinearLayout topBar = new LinearLayout(this);
        topBar.setOrientation(LinearLayout.HORIZONTAL);
        topBar.setBackgroundColor(0xFF1976D2);
        topBar.setPadding(dp16, dp8, dp8, dp8);
        topBar.setGravity(Gravity.CENTER_VERTICAL);

        TextView title = new TextView(this);
        title.setText("Fastllm Chat");
        title.setTextColor(Color.WHITE);
        title.setTextSize(TypedValue.COMPLEX_UNIT_SP, 18);
        LinearLayout.LayoutParams titleLp = new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1);
        topBar.addView(title, titleLp);

        selectBtn = new Button(this);
        selectBtn.setText("选择模型");
        selectBtn.setTextColor(Color.WHITE);
        selectBtn.setBackgroundColor(0xFF1565C0);
        selectBtn.setPadding(dp12, 0, dp12, 0);
        topBar.addView(selectBtn, new LinearLayout.LayoutParams(ViewGroup.LayoutParams.WRAP_CONTENT, dp(40)));

        resetBtn = new Button(this);
        resetBtn.setText("重置");
        resetBtn.setTextColor(Color.WHITE);
        resetBtn.setBackgroundColor(0xFF1565C0);
        resetBtn.setPadding(dp12, 0, dp12, 0);
        resetBtn.setVisibility(View.GONE);
        LinearLayout.LayoutParams resetLp = new LinearLayout.LayoutParams(ViewGroup.LayoutParams.WRAP_CONTENT, dp(40));
        resetLp.leftMargin = dp(4);
        topBar.addView(resetBtn, resetLp);

        root.addView(topBar, new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT));

        // Status
        statusText = new TextView(this);
        statusText.setText("请点击「选择模型」加载本地大模型文件");
        statusText.setTextColor(0xFF666666);
        statusText.setTextSize(TypedValue.COMPLEX_UNIT_SP, 14);
        statusText.setGravity(Gravity.CENTER);
        statusText.setPadding(dp16, dp12, dp16, dp12);
        root.addView(statusText, new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT));

        // Chat area
        scrollView = new ScrollView(this);
        scrollView.setFillViewport(true);
        chatContainer = new LinearLayout(this);
        chatContainer.setOrientation(LinearLayout.VERTICAL);
        chatContainer.setPadding(dp8, dp8, dp8, dp8);
        scrollView.addView(chatContainer, new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT));
        root.addView(scrollView, new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 0, 1));

        // Input area
        LinearLayout inputArea = new LinearLayout(this);
        inputArea.setOrientation(LinearLayout.HORIZONTAL);
        inputArea.setBackgroundColor(Color.WHITE);
        inputArea.setPadding(dp8, dp8, dp8, dp8);
        inputArea.setGravity(Gravity.CENTER_VERTICAL);
        inputArea.setVisibility(View.GONE);
        inputArea.setTag("inputArea");

        inputBox = new EditText(this);
        inputBox.setHint("输入消息...");
        inputBox.setTextSize(TypedValue.COMPLEX_UNIT_SP, 15);
        inputBox.setMaxLines(4);
        inputBox.setBackgroundColor(0xFFF0F0F0);
        inputBox.setPadding(dp12, dp8, dp12, dp8);
        LinearLayout.LayoutParams inputLp = new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1);
        inputArea.addView(inputBox, inputLp);

        sendBtn = new Button(this);
        sendBtn.setText("发送");
        sendBtn.setTextColor(Color.WHITE);
        sendBtn.setBackgroundColor(0xFF1976D2);
        LinearLayout.LayoutParams sendLp = new LinearLayout.LayoutParams(ViewGroup.LayoutParams.WRAP_CONTENT, dp(44));
        sendLp.leftMargin = dp8;
        inputArea.addView(sendBtn, sendLp);

        root.addView(inputArea, new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT));

        setContentView(root);

        // Listeners
        selectBtn.setOnClickListener(v -> {
            if (generating) { toast("请等待生成完成"); return; }
            requestStorageAndPick();
        });
        sendBtn.setOnClickListener(v -> sendMessage());
        resetBtn.setOnClickListener(v -> {
            if (generating) return;
            FastllmJNI.resetChat();
            chatContainer.removeAllViews();
        });
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        FastllmJNI.releaseModel();
    }

    private void requestStorageAndPick() {
        if (Build.VERSION.SDK_INT >= 30 && !Environment.isExternalStorageManager()) {
            try {
                Intent i = new Intent(Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION, Uri.parse("package:" + getPackageName()));
                startActivityForResult(i, REQ_PERM);
            } catch (Exception e) {
                Intent i = new Intent(Settings.ACTION_MANAGE_ALL_FILES_ACCESS_PERMISSION);
                startActivityForResult(i, REQ_PERM);
            }
            return;
        }
        pickFile();
    }

    private void pickFile() {
        new AlertDialog.Builder(this)
            .setTitle("选择方式")
            .setItems(new String[]{"选择文件", "选择文件夹"}, (dialog, which) -> {
                if (which == 0) {
                    Intent i = new Intent(Intent.ACTION_GET_CONTENT);
                    i.setType("*/*");
                    i.addCategory(Intent.CATEGORY_OPENABLE);
                    startActivityForResult(Intent.createChooser(i, "选择模型文件"), REQ_FILE);
                } else {
                    Intent i = new Intent(Intent.ACTION_OPEN_DOCUMENT_TREE);
                    startActivityForResult(i, REQ_FOLDER);
                }
            })
            .show();
    }

    @Override
    protected void onActivityResult(int req, int res, Intent data) {
        super.onActivityResult(req, res, data);
        if (req == REQ_PERM) { pickFile(); return; }
        if (req == REQ_FILE && res == RESULT_OK && data != null && data.getData() != null) {
            loadModel(data.getData());
        }
        if (req == REQ_FOLDER && res == RESULT_OK && data != null && data.getData() != null) {
            handleFolderSelection(data.getData());
        }
    }

    private void handleFolderSelection(Uri treeUri) {
        Log.i(TAG, "handleFolderSelection: uri=" + treeUri);
        String folderPath = getFolderPathFromUri(treeUri);
        Log.i(TAG, "handleFolderSelection: resolved path=" + folderPath);
        if (folderPath == null) {
            toast("无法获取文件夹路径");
            return;
        }
        File folder = new File(folderPath);
        if (!folder.exists() || !folder.isDirectory()) {
            toast("文件夹不存在: " + folderPath);
            return;
        }
        Log.i(TAG, "handleFolderSelection: folder exists, loading model from " + folderPath);
        loadModelFromPath(folderPath);
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
            return Environment.getExternalStorageDirectory().getAbsolutePath() + "/" + docId.substring("primary:".length());
        }
        // Try raw path for other storage volumes
        String[] parts = docId.split(":");
        if (parts.length == 2) {
            File[] externals = getExternalFilesDirs(null);
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

    private void loadModelFromPath(String path) {
        Log.i(TAG, "loadModelFromPath: " + path);
        statusText.setText("正在加载模型...");
        statusText.setVisibility(View.VISIBLE);
        selectBtn.setEnabled(false);

        new Thread(() -> {
            try {
                Log.i(TAG, "Calling FastllmJNI.initModel...");
                String modelType = FastllmJNI.initModel(path, 4);
                Log.i(TAG, "initModel returned: " + modelType);
                handler.post(() -> {
                    selectBtn.setEnabled(true);
                    if (modelType != null && !modelType.isEmpty()) {
                        modelLoaded = true;
                        statusText.setText("模型已加载: " + modelType);
                        findInputArea().setVisibility(View.VISIBLE);
                        resetBtn.setVisibility(View.VISIBLE);
                        chatContainer.removeAllViews();
                    } else {
                        statusText.setText("模型加载失败");
                    }
                });
            } catch (Throwable t) {
                Log.e(TAG, "loadModelFromPath crashed", t);
                handler.post(() -> {
                    selectBtn.setEnabled(true);
                    statusText.setText("模型加载异常: " + t.getMessage());
                });
            }
        }).start();
    }

    private void loadModel(Uri uri) {
        statusText.setText("正在加载模型...");
        statusText.setVisibility(View.VISIBLE);
        selectBtn.setEnabled(false);

        new Thread(() -> {
            String path = resolvePath(uri);
            if (path == null) {
                handler.post(() -> { statusText.setText("无法读取文件"); selectBtn.setEnabled(true); });
                return;
            }
            String modelType = FastllmJNI.initModel(path, 4);
            handler.post(() -> {
                selectBtn.setEnabled(true);
                if (modelType != null && !modelType.isEmpty()) {
                    modelLoaded = true;
                    statusText.setText("模型已加载: " + modelType);
                    findInputArea().setVisibility(View.VISIBLE);
                    resetBtn.setVisibility(View.VISIBLE);
                    chatContainer.removeAllViews();
                } else {
                    statusText.setText("模型加载失败");
                }
            });
        }).start();
    }

    private View findInputArea() {
        return ((ViewGroup)getWindow().getDecorView()).findViewWithTag("inputArea");
    }

    private String resolvePath(Uri uri) {
        // content:// -> copy to app dir
        try {
            String name = "model.bin";
            Cursor c = getContentResolver().query(uri, null, null, null, null);
            if (c != null && c.moveToFirst()) {
                int idx = c.getColumnIndex(OpenableColumns.DISPLAY_NAME);
                if (idx >= 0) name = c.getString(idx);
                c.close();
            }
            File out = new File(getExternalFilesDir("models"), name);
            if (out.exists() && out.length() > 1024 * 1024) {
                return out.getAbsolutePath();
            }
            InputStream is = getContentResolver().openInputStream(uri);
            if (is == null) return null;
            FileOutputStream fos = new FileOutputStream(out);
            byte[] buf = new byte[65536];
            int len;
            while ((len = is.read(buf)) != -1) fos.write(buf, 0, len);
            fos.close();
            is.close();
            return out.getAbsolutePath();
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }

    private void sendMessage() {
        if (!modelLoaded || generating) return;
        String text = inputBox.getText().toString().trim();
        if (text.isEmpty()) return;
        inputBox.setText("");
        generating = true;
        sendBtn.setEnabled(false);

        addBubble(text, true);

        new Thread(() -> FastllmJNI.chat(text, (index, content) ->
            handler.post(() -> {
                if (index == 0) {
                    currentBotBubble = addBubble(content, false);
                } else if (index > 0 && currentBotBubble != null) {
                    currentBotBubble.setText(currentBotBubble.getText() + content);
                    scrollToBottom();
                } else if (index == -1) {
                    generating = false;
                    currentBotBubble = null;
                    sendBtn.setEnabled(true);
                }
            })
        )).start();
    }

    private TextView addBubble(String text, boolean isUser) {
        LinearLayout row = new LinearLayout(this);
        row.setOrientation(LinearLayout.HORIZONTAL);
        row.setGravity(isUser ? Gravity.END : Gravity.START);
        LinearLayout.LayoutParams rowLp = new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT);
        rowLp.bottomMargin = dp(6);

        TextView tv = new TextView(this);
        tv.setText(text);
        tv.setTextSize(TypedValue.COMPLEX_UNIT_SP, 15);
        tv.setTextColor(0xFF333333);
        tv.setPadding(dp(12), dp(8), dp(12), dp(8));
        tv.setTextIsSelectable(true);

        GradientDrawable bg = new GradientDrawable();
        bg.setCornerRadius(dp(12));
        bg.setColor(isUser ? 0xFFDCF8C6 : Color.WHITE);
        tv.setBackground(bg);

        LinearLayout.LayoutParams tvLp = new LinearLayout.LayoutParams(ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT);
        tvLp.leftMargin = isUser ? dp(48) : 0;
        tvLp.rightMargin = isUser ? 0 : dp(48);
        row.addView(tv, tvLp);

        chatContainer.addView(row, rowLp);
        scrollToBottom();
        return tv;
    }

    private void scrollToBottom() {
        scrollView.post(() -> scrollView.fullScroll(View.FOCUS_DOWN));
    }

    private int dp(int v) {
        return (int)(v * getResources().getDisplayMetrics().density);
    }

    private void toast(String s) {
        Toast.makeText(this, s, Toast.LENGTH_SHORT).show();
    }
}
