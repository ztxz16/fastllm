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
    private TextView sendBtn;
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
    private boolean scrollPending = false;

    private float topP = 0.9f;
    private int topK = 5;
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
        root.setBackgroundColor(MainActivity.COLOR_BG);

        LinearLayout topArea = new LinearLayout(activity);
        topArea.setOrientation(LinearLayout.HORIZONTAL);
        topArea.setBackgroundColor(MainActivity.COLOR_SURFACE);
        topArea.setPadding(dp(16), dp(10), dp(10), dp(10));
        topArea.setGravity(Gravity.CENTER_VERTICAL);
        topArea.setElevation(dp(2));

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

        TextView settingsBtn = createIconButton("⚙️");
        settingsBtn.setOnClickListener(v -> showSettings());
        topArea.addView(settingsBtn, new LinearLayout.LayoutParams(dp(42), dp(42)));

        View spacer = new View(activity);
        topArea.addView(spacer, new LinearLayout.LayoutParams(dp(4), 0));

        TextView resetBtn = createIconButton("🔄");
        resetBtn.setOnClickListener(v -> {
            if (generating) return;
            FastllmJNI.resetChat();
            chatContainer.removeAllViews();
        });
        topArea.addView(resetBtn, new LinearLayout.LayoutParams(dp(42), dp(42)));

        root.addView(topArea, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT));

        statusText = new TextView(activity);
        statusText.setText("请选择一个模型开始对话");
        statusText.setTextColor(MainActivity.COLOR_TEXT_SECONDARY);
        statusText.setTextSize(TypedValue.COMPLEX_UNIT_SP, 14);
        statusText.setGravity(Gravity.CENTER);
        statusText.setPadding(dp(16), dp(14), dp(16), dp(14));

        GradientDrawable statusBg = new GradientDrawable();
        statusBg.setColor(0xFFF0EFFF);
        statusBg.setCornerRadius(dp(12));
        statusText.setBackground(statusBg);

        LinearLayout.LayoutParams statusLp = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT);
        statusLp.setMargins(dp(16), dp(12), dp(16), dp(4));
        root.addView(statusText, statusLp);

        scrollView = new ScrollView(activity);
        scrollView.setFillViewport(true);
        chatContainer = new LinearLayout(activity);
        chatContainer.setOrientation(LinearLayout.VERTICAL);
        chatContainer.setPadding(dp(12), dp(8), dp(12), dp(8));
        scrollView.addView(chatContainer, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT));
        root.addView(scrollView, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, 0, 1));

        LinearLayout inputWrapper = new LinearLayout(activity);
        inputWrapper.setOrientation(LinearLayout.VERTICAL);
        inputWrapper.setBackgroundColor(MainActivity.COLOR_SURFACE);
        inputWrapper.setElevation(dp(6));

        View inputDivider = new View(activity);
        inputDivider.setBackgroundColor(MainActivity.COLOR_DIVIDER);
        inputWrapper.addView(inputDivider, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, 1));

        inputArea = new LinearLayout(activity);
        inputArea.setOrientation(LinearLayout.HORIZONTAL);
        inputArea.setPadding(dp(12), dp(10), dp(12), dp(10));
        inputArea.setGravity(Gravity.CENTER_VERTICAL);
        inputArea.setVisibility(View.GONE);

        inputBox = new EditText(activity);
        inputBox.setHint("输入消息...");
        inputBox.setTextSize(TypedValue.COMPLEX_UNIT_SP, 15);
        inputBox.setTextColor(MainActivity.COLOR_TEXT_PRIMARY);
        inputBox.setHintTextColor(0xFFB0B3C5);
        inputBox.setMaxLines(4);

        GradientDrawable inputBg = new GradientDrawable();
        inputBg.setCornerRadius(dp(24));
        inputBg.setColor(0xFFF3F4F8);
        inputBg.setStroke(1, 0xFFE0E3EE);
        inputBox.setBackground(inputBg);
        inputBox.setPadding(dp(18), dp(12), dp(18), dp(12));

        inputArea.addView(inputBox, new LinearLayout.LayoutParams(
                0, ViewGroup.LayoutParams.WRAP_CONTENT, 1));

        sendBtn = new TextView(activity);
        sendBtn.setText("➤");
        sendBtn.setTextSize(TypedValue.COMPLEX_UNIT_SP, 20);
        sendBtn.setTextColor(Color.WHITE);
        sendBtn.setGravity(Gravity.CENTER);

        GradientDrawable sendBg = new GradientDrawable(
                GradientDrawable.Orientation.TL_BR,
                new int[]{MainActivity.COLOR_GRADIENT_START, MainActivity.COLOR_GRADIENT_END});
        sendBg.setCornerRadius(dp(22));
        sendBtn.setBackground(sendBg);

        LinearLayout.LayoutParams sendLp = new LinearLayout.LayoutParams(dp(44), dp(44));
        sendLp.leftMargin = dp(10);
        inputArea.addView(sendBtn, sendLp);

        sendBtn.setOnClickListener(v -> sendMessage());

        inputWrapper.addView(inputArea, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT));

        root.addView(inputWrapper, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT));

        return root;
    }

    private TextView createIconButton(String icon) {
        TextView btn = new TextView(activity);
        btn.setText(icon);
        btn.setTextSize(TypedValue.COMPLEX_UNIT_SP, 18);
        btn.setGravity(Gravity.CENTER);

        GradientDrawable bg = new GradientDrawable();
        bg.setCornerRadius(dp(12));
        bg.setColor(0xFFF3F4F8);
        btn.setBackground(bg);

        return btn;
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

        statusText.setText("⏳ 正在加载模型: " + model.name + "...");
        statusText.setVisibility(View.VISIBLE);
        GradientDrawable loadingBg = new GradientDrawable();
        loadingBg.setColor(0xFFFFF8E1);
        loadingBg.setCornerRadius(dp(12));
        statusText.setBackground(loadingBg);
        inputArea.setVisibility(View.GONE);

        new Thread(() -> {
            try {
                Log.i(TAG, "Loading model: " + model.path);
                String modelType = FastllmJNI.initModel(model.path, 4);
                handler.post(() -> {
                    if (modelType != null && !modelType.isEmpty()) {
                        modelLoaded = true;
                        currentModelPath = model.path;
                        statusText.setText("✅ 模型已加载: " + modelType);
                        GradientDrawable successBg = new GradientDrawable();
                        successBg.setColor(0xFFE8F5E9);
                        successBg.setCornerRadius(dp(12));
                        statusText.setBackground(successBg);
                        inputArea.setVisibility(View.VISIBLE);
                        chatContainer.removeAllViews();
                        FastllmJNI.resetChat();
                    } else {
                        statusText.setText("❌ 模型加载失败");
                        GradientDrawable errBg = new GradientDrawable();
                        errBg.setColor(0xFFFFEBEE);
                        errBg.setCornerRadius(dp(12));
                        statusText.setBackground(errBg);
                    }
                });
            } catch (Throwable t) {
                Log.e(TAG, "Model load error", t);
                handler.post(() -> {
                    statusText.setText("❌ 模型加载异常: " + t.getMessage());
                    GradientDrawable errBg = new GradientDrawable();
                    errBg.setColor(0xFFFFEBEE);
                    errBg.setCornerRadius(dp(12));
                    statusText.setBackground(errBg);
                });
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
        sendBtn.setAlpha(0.5f);

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
                        sendBtn.setAlpha(1f);
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
        rowLp.bottomMargin = dp(10);

        TextView avatar = new TextView(activity);
        avatar.setTextSize(TypedValue.COMPLEX_UNIT_SP, 13);
        avatar.setGravity(Gravity.CENTER);
        avatar.setTextColor(Color.WHITE);
        avatar.setTypeface(null, android.graphics.Typeface.BOLD);

        GradientDrawable avatarBg = new GradientDrawable();
        avatarBg.setCornerRadius(dp(20));

        LinearLayout.LayoutParams avatarLp = new LinearLayout.LayoutParams(dp(36), dp(36));
        avatarLp.gravity = Gravity.TOP;

        if (isUser) {
            avatar.setText("我");
            avatarBg = new GradientDrawable(
                    GradientDrawable.Orientation.TL_BR,
                    new int[]{MainActivity.COLOR_GRADIENT_START, MainActivity.COLOR_GRADIENT_END});
            avatarBg.setCornerRadius(dp(20));
            avatarLp.leftMargin = dp(8);
        } else {
            avatar.setText("AI");
            avatarBg = new GradientDrawable(
                    GradientDrawable.Orientation.TL_BR,
                    new int[]{MainActivity.COLOR_ACCENT, 0xFF009688});
            avatarBg.setCornerRadius(dp(20));
            avatarLp.rightMargin = dp(8);
        }
        avatar.setBackground(avatarBg);

        LinearLayout bubbleWrapper = new LinearLayout(activity);
        bubbleWrapper.setOrientation(LinearLayout.VERTICAL);

        TextView tv = new TextView(activity);
        tv.setText(text);
        tv.setTextSize(TypedValue.COMPLEX_UNIT_SP, 15);
        tv.setLineSpacing(dp(2), 1f);
        tv.setTextColor(isUser ? Color.WHITE : MainActivity.COLOR_TEXT_PRIMARY);
        tv.setPadding(dp(14), dp(10), dp(14), dp(10));
        tv.setTextIsSelectable(true);

        GradientDrawable bubbleBg;
        if (isUser) {
            bubbleBg = new GradientDrawable(
                    GradientDrawable.Orientation.TL_BR,
                    new int[]{MainActivity.COLOR_GRADIENT_START, MainActivity.COLOR_GRADIENT_END});
            bubbleBg.setCornerRadii(new float[]{
                    dp(16), dp(16), dp(4), dp(4),
                    dp(16), dp(16), dp(16), dp(16)});
        } else {
            bubbleBg = new GradientDrawable();
            bubbleBg.setColor(MainActivity.COLOR_SURFACE);
            bubbleBg.setCornerRadii(new float[]{
                    dp(4), dp(4), dp(16), dp(16),
                    dp(16), dp(16), dp(16), dp(16)});
            bubbleBg.setStroke(1, 0xFFE8EAF0);
        }
        tv.setBackground(bubbleBg);
        tv.setElevation(dp(1));

        LinearLayout.LayoutParams tvLp = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT);
        int maxWidth = (int) (activity.getResources().getDisplayMetrics().widthPixels * 0.7);
        tv.setMaxWidth(maxWidth);
        bubbleWrapper.addView(tv, tvLp);

        LinearLayout.LayoutParams wrapperLp = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT);

        if (isUser) {
            row.addView(bubbleWrapper, wrapperLp);
            row.addView(avatar, avatarLp);
        } else {
            row.addView(avatar, avatarLp);
            row.addView(bubbleWrapper, wrapperLp);
        }

        chatContainer.addView(row, rowLp);
        scrollToBottom();
        return tv;
    }

    private void scrollToBottom() {
        if (scrollPending) return;
        scrollPending = true;
        scrollView.post(() -> {
            scrollPending = false;
            int contentHeight = chatContainer.getHeight();
            int scrollHeight = scrollView.getHeight();
            if (contentHeight > scrollHeight) {
                scrollView.scrollTo(0, contentHeight - scrollHeight);
            }
        });
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
