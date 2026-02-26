package com.fastllm.app;

import android.graphics.Color;
import android.graphics.drawable.GradientDrawable;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;
import android.util.TypedValue;
import android.view.Gravity;
import android.view.View;
import android.view.ViewGroup;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.Spinner;
import android.widget.TextView;

import java.util.ArrayList;
import java.util.List;

public class ChatPage {

    private static final String TAG = "ChatPage";

    private final MainActivity activity;
    private final ModelStorage storage;
    private final View rootView;

    private Spinner modelSpinner;
    private LinearLayout chatContainer;
    private ScrollView scrollView;
    private EditText inputBox;
    private Button sendBtn;
    private TextView statusText;
    private LinearLayout inputArea;

    private final Handler handler = new Handler(Looper.getMainLooper());
    private final List<ModelStorage.ModelInfo> modelList = new ArrayList<>();
    private ArrayAdapter<String> spinnerAdapter;
    private final List<String> spinnerItems = new ArrayList<>();

    private volatile boolean modelLoaded = false;
    private volatile boolean generating = false;
    private TextView currentBotBubble = null;
    private String currentModelPath = null;

    // Sampling params
    private float topP = 0.9f;
    private int topK = 50;
    private float temperature = 0.7f;
    private float repeatPenalty = 1.0f;

    public ChatPage(MainActivity activity) {
        this.activity = activity;
        this.storage = new ModelStorage(activity);
        this.rootView = buildUI();
        refreshModelList();
    }

    View getView() {
        return rootView;
    }

    private View buildUI() {
        LinearLayout root = new LinearLayout(activity);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setBackgroundColor(0xFFF5F5F5);

        // Top area: model selector + settings
        LinearLayout topArea = new LinearLayout(activity);
        topArea.setOrientation(LinearLayout.HORIZONTAL);
        topArea.setBackgroundColor(Color.WHITE);
        topArea.setPadding(dp(12), dp(8), dp(8), dp(8));
        topArea.setGravity(Gravity.CENTER_VERTICAL);

        // Model spinner
        modelSpinner = new Spinner(activity);
        spinnerAdapter = new ArrayAdapter<>(activity,
                android.R.layout.simple_spinner_item, spinnerItems);
        spinnerAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        modelSpinner.setAdapter(spinnerAdapter);

        modelSpinner.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> parent, View view, int pos, long id) {
                if (pos == 0) return;
                int modelIdx = pos - 1;
                if (modelIdx < modelList.size()) {
                    loadModel(modelList.get(modelIdx));
                }
            }
            @Override
            public void onNothingSelected(AdapterView<?> parent) {}
        });

        topArea.addView(modelSpinner, new LinearLayout.LayoutParams(
                0, ViewGroup.LayoutParams.WRAP_CONTENT, 1));

        // Settings button
        Button settingsBtn = new Button(activity);
        settingsBtn.setText("⚙");
        settingsBtn.setTextSize(TypedValue.COMPLEX_UNIT_SP, 18);
        settingsBtn.setBackgroundColor(Color.TRANSPARENT);
        settingsBtn.setPadding(dp(8), 0, dp(8), 0);
        settingsBtn.setOnClickListener(v -> showSettings());
        topArea.addView(settingsBtn, new LinearLayout.LayoutParams(
                dp(44), dp(44)));

        // Reset button
        Button resetBtn = new Button(activity);
        resetBtn.setText("🔄");
        resetBtn.setTextSize(TypedValue.COMPLEX_UNIT_SP, 18);
        resetBtn.setBackgroundColor(Color.TRANSPARENT);
        resetBtn.setPadding(dp(8), 0, dp(8), 0);
        resetBtn.setOnClickListener(v -> {
            if (generating) return;
            FastllmJNI.resetChat();
            chatContainer.removeAllViews();
        });
        topArea.addView(resetBtn, new LinearLayout.LayoutParams(
                dp(44), dp(44)));

        root.addView(topArea, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT));

        View divider = new View(activity);
        divider.setBackgroundColor(0xFFEEEEEE);
        root.addView(divider, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, 1));

        // Status
        statusText = new TextView(activity);
        statusText.setText("请选择一个模型开始对话");
        statusText.setTextColor(0xFF666666);
        statusText.setTextSize(TypedValue.COMPLEX_UNIT_SP, 14);
        statusText.setGravity(Gravity.CENTER);
        statusText.setPadding(dp(16), dp(12), dp(16), dp(12));
        root.addView(statusText, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT));

        // Chat area
        scrollView = new ScrollView(activity);
        scrollView.setFillViewport(true);
        chatContainer = new LinearLayout(activity);
        chatContainer.setOrientation(LinearLayout.VERTICAL);
        chatContainer.setPadding(dp(10), dp(8), dp(10), dp(8));
        scrollView.addView(chatContainer, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT));
        root.addView(scrollView, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, 0, 1));

        // Input area
        inputArea = new LinearLayout(activity);
        inputArea.setOrientation(LinearLayout.HORIZONTAL);
        inputArea.setBackgroundColor(Color.WHITE);
        inputArea.setPadding(dp(10), dp(8), dp(10), dp(8));
        inputArea.setGravity(Gravity.CENTER_VERTICAL);
        inputArea.setVisibility(View.GONE);

        View inputDivider = new View(activity);
        inputDivider.setBackgroundColor(0xFFEEEEEE);
        root.addView(inputDivider, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, 1));

        inputBox = new EditText(activity);
        inputBox.setHint("输入消息...");
        inputBox.setTextSize(TypedValue.COMPLEX_UNIT_SP, 15);
        inputBox.setMaxLines(4);

        GradientDrawable inputBg = new GradientDrawable();
        inputBg.setCornerRadius(dp(20));
        inputBg.setColor(0xFFF0F0F0);
        inputBox.setBackground(inputBg);
        inputBox.setPadding(dp(16), dp(10), dp(16), dp(10));

        inputArea.addView(inputBox, new LinearLayout.LayoutParams(
                0, ViewGroup.LayoutParams.WRAP_CONTENT, 1));

        sendBtn = new Button(activity);
        sendBtn.setText("发送");
        sendBtn.setTextColor(Color.WHITE);
        sendBtn.setTextSize(TypedValue.COMPLEX_UNIT_SP, 14);
        sendBtn.setAllCaps(false);

        GradientDrawable sendBg = new GradientDrawable();
        sendBg.setCornerRadius(dp(20));
        sendBg.setColor(0xFF1976D2);
        sendBtn.setBackground(sendBg);

        LinearLayout.LayoutParams sendLp = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.WRAP_CONTENT, dp(40));
        sendLp.leftMargin = dp(8);
        inputArea.addView(sendBtn, sendLp);

        sendBtn.setOnClickListener(v -> sendMessage());

        root.addView(inputArea, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT));

        return root;
    }

    void refreshModelList() {
        modelList.clear();
        spinnerItems.clear();
        spinnerItems.add("-- 请选择模型 --");

        List<ModelStorage.ModelInfo> models = storage.getModels();
        for (ModelStorage.ModelInfo m : models) {
            modelList.add(m);
            spinnerItems.add(m.name);
        }
        spinnerAdapter.notifyDataSetChanged();
    }

    private void loadModel(ModelStorage.ModelInfo model) {
        if (generating) {
            activity.toast("请等待生成完成");
            return;
        }
        if (model.path.equals(currentModelPath) && modelLoaded) {
            return;
        }

        statusText.setText("正在加载模型: " + model.name + "...");
        statusText.setVisibility(View.VISIBLE);
        inputArea.setVisibility(View.GONE);

        new Thread(() -> {
            try {
                Log.i(TAG, "Loading model: " + model.path);
                String modelType = FastllmJNI.initModel(model.path, 4);
                handler.post(() -> {
                    if (modelType != null && !modelType.isEmpty()) {
                        modelLoaded = true;
                        currentModelPath = model.path;
                        statusText.setText("模型已加载: " + modelType);
                        inputArea.setVisibility(View.VISIBLE);
                        chatContainer.removeAllViews();
                        FastllmJNI.resetChat();
                    } else {
                        statusText.setText("模型加载失败");
                    }
                });
            } catch (Throwable t) {
                Log.e(TAG, "Model load error", t);
                handler.post(() -> statusText.setText("模型加载异常: " + t.getMessage()));
            }
        }).start();
    }

    private void sendMessage() {
        if (!modelLoaded || generating) return;
        String text = inputBox.getText().toString().trim();
        if (text.isEmpty()) return;
        inputBox.setText("");
        generating = true;
        sendBtn.setEnabled(false);

        addBubble(text, true);

        new Thread(() -> FastllmJNI.chat(text, topP, topK, temperature, repeatPenalty,
                (index, content) -> handler.post(() -> {
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
        LinearLayout row = new LinearLayout(activity);
        row.setOrientation(LinearLayout.HORIZONTAL);
        row.setGravity(isUser ? Gravity.END : Gravity.START);

        LinearLayout.LayoutParams rowLp = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT);
        rowLp.bottomMargin = dp(6);

        // Avatar
        TextView avatar = new TextView(activity);
        avatar.setTextSize(TypedValue.COMPLEX_UNIT_SP, 12);
        avatar.setGravity(Gravity.CENTER);
        avatar.setTextColor(Color.WHITE);

        GradientDrawable avatarBg = new GradientDrawable();
        avatarBg.setCornerRadius(dp(18));

        LinearLayout.LayoutParams avatarLp = new LinearLayout.LayoutParams(dp(36), dp(36));

        if (isUser) {
            avatar.setText("我");
            avatarBg.setColor(0xFF1976D2);
            avatarLp.leftMargin = dp(8);
        } else {
            avatar.setText("AI");
            avatarBg.setColor(0xFF43A047);
            avatarLp.rightMargin = dp(8);
        }
        avatar.setBackground(avatarBg);

        // Bubble
        TextView tv = new TextView(activity);
        tv.setText(text);
        tv.setTextSize(TypedValue.COMPLEX_UNIT_SP, 15);
        tv.setTextColor(isUser ? 0xFF333333 : 0xFF333333);
        tv.setPadding(dp(12), dp(8), dp(12), dp(8));
        tv.setTextIsSelectable(true);

        GradientDrawable bubbleBg = new GradientDrawable();
        if (isUser) {
            bubbleBg.setCornerRadii(new float[]{
                    dp(12), dp(12), dp(4), dp(4),
                    dp(12), dp(12), dp(12), dp(12)});
            bubbleBg.setColor(0xFF95EC69);
        } else {
            bubbleBg.setCornerRadii(new float[]{
                    dp(4), dp(4), dp(12), dp(12),
                    dp(12), dp(12), dp(12), dp(12)});
            bubbleBg.setColor(Color.WHITE);
        }
        tv.setBackground(bubbleBg);

        LinearLayout.LayoutParams tvLp = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT);
        tvLp.gravity = Gravity.TOP;
        int maxWidth = (int) (activity.getResources().getDisplayMetrics().widthPixels * 0.65);
        tv.setMaxWidth(maxWidth);

        if (isUser) {
            row.addView(tv, tvLp);
            row.addView(avatar, avatarLp);
        } else {
            row.addView(avatar, avatarLp);
            row.addView(tv, tvLp);
        }

        chatContainer.addView(row, rowLp);
        scrollToBottom();
        return tv;
    }

    private void scrollToBottom() {
        scrollView.post(() -> scrollView.fullScroll(View.FOCUS_DOWN));
    }

    private void showSettings() {
        SettingsDialogHelper.show(activity, topP, topK, temperature, repeatPenalty,
                (newTopP, newTopK, newTemp, newRepeat) -> {
                    topP = newTopP;
                    topK = newTopK;
                    temperature = newTemp;
                    repeatPenalty = newRepeat;
                });
    }

    void onDestroy() {
        FastllmJNI.releaseModel();
    }

    private int dp(int v) {
        return activity.dp(v);
    }
}
