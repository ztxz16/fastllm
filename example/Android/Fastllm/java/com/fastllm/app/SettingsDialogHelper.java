package com.fastllm.app;

import android.app.AlertDialog;
import android.content.Context;
import android.graphics.Color;
import android.graphics.drawable.GradientDrawable;
import android.util.TypedValue;
import android.view.Gravity;
import android.view.ViewGroup;
import android.widget.LinearLayout;
import android.widget.SeekBar;
import android.widget.TextView;

public class SettingsDialogHelper {

    public interface OnSettingsConfirmed {
        void onConfirm(float topP, int topK, float temperature, float repeatPenalty);
    }

    public static void show(Context context, float curTopP, int curTopK,
                            float curTemp, float curRepeat, OnSettingsConfirmed callback) {
        int dp12 = dp(context, 12);
        int dp20 = dp(context, 20);

        LinearLayout layout = new LinearLayout(context);
        layout.setOrientation(LinearLayout.VERTICAL);
        layout.setBackgroundColor(0xFFFFFFFF);
        layout.setPadding(dp20, dp20, dp20, dp12);

        TextView headerIcon = new TextView(context);
        headerIcon.setText("⚙️ 采样参数设置");
        headerIcon.setTextSize(TypedValue.COMPLEX_UNIT_SP, 18);
        headerIcon.setTextColor(0xFF1A1A2E);
        headerIcon.setTypeface(null, android.graphics.Typeface.BOLD);
        layout.addView(headerIcon);

        TextView headerDesc = new TextView(context);
        headerDesc.setText("调整模型生成文本的策略参数");
        headerDesc.setTextSize(TypedValue.COMPLEX_UNIT_SP, 13);
        headerDesc.setTextColor(0xFF888888);
        LinearLayout.LayoutParams hdLp = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT);
        hdLp.topMargin = dp(context, 4);
        hdLp.bottomMargin = dp(context, 16);
        layout.addView(headerDesc, hdLp);

        SeekBarRow topPRow = addSeekBarRow(layout, context, "Top P", "控制采样多样性",
                0, 100, (int) (curTopP * 100), v -> String.format("%.2f", v / 100f));

        SeekBarRow topKRow = addSeekBarRow(layout, context, "Top K", "候选token数量",
                1, 100, curTopK, v -> String.valueOf(v));

        SeekBarRow tempRow = addSeekBarRow(layout, context, "Temperature", "温度越高输出越随机",
                0, 200, (int) (curTemp * 100), v -> String.format("%.2f", v / 100f));

        SeekBarRow repeatRow = addSeekBarRow(layout, context, "Repeat Penalty", "重复惩罚系数",
                100, 200, (int) (curRepeat * 100), v -> String.format("%.2f", v / 100f));

        AlertDialog dlg = new AlertDialog.Builder(context)
                .setView(layout)
                .setPositiveButton("确定", (d, w) -> {
                    float topP = (topPRow.seekBar.getProgress() + (int) topPRow.seekBar.getTag()) / 100f;
                    int topK = topKRow.seekBar.getProgress() + (int) topKRow.seekBar.getTag();
                    float temp = (tempRow.seekBar.getProgress() + (int) tempRow.seekBar.getTag()) / 100f;
                    float repeat = (repeatRow.seekBar.getProgress() + (int) repeatRow.seekBar.getTag()) / 100f;
                    callback.onConfirm(topP, topK, temp, repeat);
                })
                .setNegativeButton("取消", null)
                .create();
        dlg.show();
    }

    private static class SeekBarRow {
        SeekBar seekBar;
        TextView valueText;
    }

    interface ValueFormatter {
        String format(int progress);
    }

    private static SeekBarRow addSeekBarRow(LinearLayout parent, Context context,
                                            String label, String hint, int min, int max,
                                            int current, ValueFormatter formatter) {
        int dp4 = dp(context, 4);
        int dp8 = dp(context, 8);
        int dp12 = dp(context, 12);

        LinearLayout card = new LinearLayout(context);
        card.setOrientation(LinearLayout.VERTICAL);
        card.setPadding(dp12, dp12, dp12, dp12);

        GradientDrawable cardBg = new GradientDrawable();
        cardBg.setCornerRadius(dp(context, 12));
        cardBg.setColor(0xFFF5F5FA);
        cardBg.setStroke(1, 0xFFE8E8F0);
        card.setBackground(cardBg);

        LinearLayout.LayoutParams cardLp = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT);
        cardLp.bottomMargin = dp(context, 10);

        LinearLayout labelRow = new LinearLayout(context);
        labelRow.setOrientation(LinearLayout.HORIZONTAL);
        labelRow.setGravity(Gravity.CENTER_VERTICAL);

        LinearLayout labelGroup = new LinearLayout(context);
        labelGroup.setOrientation(LinearLayout.VERTICAL);

        TextView labelView = new TextView(context);
        labelView.setText(label);
        labelView.setTextSize(TypedValue.COMPLEX_UNIT_SP, 14);
        labelView.setTextColor(0xFF222222);
        labelView.setTypeface(null, android.graphics.Typeface.BOLD);
        labelGroup.addView(labelView);

        TextView hintView = new TextView(context);
        hintView.setText(hint);
        hintView.setTextSize(TypedValue.COMPLEX_UNIT_SP, 11);
        hintView.setTextColor(0xFF888888);
        labelGroup.addView(hintView);

        labelRow.addView(labelGroup, new LinearLayout.LayoutParams(
                0, ViewGroup.LayoutParams.WRAP_CONTENT, 1));

        TextView valueText = new TextView(context);
        valueText.setText(formatter.format(current));
        valueText.setTextSize(TypedValue.COMPLEX_UNIT_SP, 16);
        valueText.setTextColor(0xFF4834DF);
        valueText.setTypeface(null, android.graphics.Typeface.BOLD);

        GradientDrawable valueBg = new GradientDrawable();
        valueBg.setCornerRadius(dp(context, 8));
        valueBg.setColor(0xFFEDE9FF);
        valueText.setBackground(valueBg);
        valueText.setPadding(dp(context, 10), dp4, dp(context, 10), dp4);
        valueText.setGravity(Gravity.CENTER);

        labelRow.addView(valueText, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT));

        card.addView(labelRow, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT));

        SeekBar seekBar = new SeekBar(context);
        seekBar.setMax(max - min);
        seekBar.setProgress(current - min);
        LinearLayout.LayoutParams seekLp = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT);
        seekLp.topMargin = dp8;

        seekBar.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar sb, int progress, boolean fromUser) {
                valueText.setText(formatter.format(progress + min));
            }
            @Override
            public void onStartTrackingTouch(SeekBar sb) {}
            @Override
            public void onStopTrackingTouch(SeekBar sb) {}
        });

        card.addView(seekBar, seekLp);
        parent.addView(card, cardLp);

        SeekBarRow result = new SeekBarRow();
        result.seekBar = seekBar;
        result.valueText = valueText;
        seekBar.setTag(min);

        return result;
    }

    private static int dp(Context context, int v) {
        return (int) (v * context.getResources().getDisplayMetrics().density);
    }
}
