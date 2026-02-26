package com.fastllm.app;

import android.app.AlertDialog;
import android.content.Context;
import android.graphics.Color;
import android.util.TypedValue;
import android.view.Gravity;
import android.view.ViewGroup;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.SeekBar;
import android.widget.TextView;
import android.widget.Toast;

public class SettingsDialogHelper {

    public interface OnSettingsConfirmed {
        void onConfirm(float topP, int topK, float temperature, float repeatPenalty);
    }

    public static void show(Context context, float curTopP, int curTopK,
                            float curTemp, float curRepeat, OnSettingsConfirmed callback) {
        int dp8 = dp(context, 8);
        int dp16 = dp(context, 16);

        LinearLayout layout = new LinearLayout(context);
        layout.setOrientation(LinearLayout.VERTICAL);
        layout.setPadding(dp16, dp16, dp16, dp8);

        // Top P
        SeekBarRow topPRow = addSeekBarRow(layout, context, "Top P",
                0, 100, (int) (curTopP * 100), v -> String.format("%.2f", v / 100f));

        // Top K
        SeekBarRow topKRow = addSeekBarRow(layout, context, "Top K",
                1, 100, curTopK, v -> String.valueOf(v));

        // Temperature
        SeekBarRow tempRow = addSeekBarRow(layout, context, "Temperature",
                0, 200, (int) (curTemp * 100), v -> String.format("%.2f", v / 100f));

        // Repeat Penalty
        SeekBarRow repeatRow = addSeekBarRow(layout, context, "Repeat Penalty",
                100, 200, (int) (curRepeat * 100), v -> String.format("%.2f", v / 100f));

        new AlertDialog.Builder(context)
                .setTitle("采样参数设置")
                .setView(layout)
                .setPositiveButton("确定", (d, w) -> {
                    float topP = (topPRow.seekBar.getProgress() + (int) topPRow.seekBar.getTag()) / 100f;
                    int topK = topKRow.seekBar.getProgress() + (int) topKRow.seekBar.getTag();
                    float temp = (tempRow.seekBar.getProgress() + (int) tempRow.seekBar.getTag()) / 100f;
                    float repeat = (repeatRow.seekBar.getProgress() + (int) repeatRow.seekBar.getTag()) / 100f;
                    callback.onConfirm(topP, topK, temp, repeat);
                })
                .setNegativeButton("取消", null)
                .show();
    }

    private static class SeekBarRow {
        SeekBar seekBar;
        TextView valueText;
    }

    interface ValueFormatter {
        String format(int progress);
    }

    private static SeekBarRow addSeekBarRow(LinearLayout parent, Context context,
                                            String label, int min, int max, int current,
                                            ValueFormatter formatter) {
        int dp4 = dp(context, 4);
        int dp8 = dp(context, 8);

        LinearLayout row = new LinearLayout(context);
        row.setOrientation(LinearLayout.VERTICAL);
        LinearLayout.LayoutParams rowLp = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT);
        rowLp.bottomMargin = dp(context, 12);

        // Label row
        LinearLayout labelRow = new LinearLayout(context);
        labelRow.setOrientation(LinearLayout.HORIZONTAL);

        TextView labelView = new TextView(context);
        labelView.setText(label);
        labelView.setTextSize(TypedValue.COMPLEX_UNIT_SP, 14);
        labelView.setTextColor(0xFF333333);
        labelRow.addView(labelView, new LinearLayout.LayoutParams(
                0, ViewGroup.LayoutParams.WRAP_CONTENT, 1));

        TextView valueText = new TextView(context);
        valueText.setText(formatter.format(current));
        valueText.setTextSize(TypedValue.COMPLEX_UNIT_SP, 14);
        valueText.setTextColor(0xFF1976D2);
        labelRow.addView(valueText, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT));

        row.addView(labelRow, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT));

        // SeekBar
        SeekBar seekBar = new SeekBar(context);
        seekBar.setMax(max - min);
        seekBar.setProgress(current - min);
        LinearLayout.LayoutParams seekLp = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT);
        seekLp.topMargin = dp4;

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

        row.addView(seekBar, seekLp);

        parent.addView(row, rowLp);

        SeekBarRow result = new SeekBarRow();
        result.seekBar = seekBar;
        result.valueText = valueText;

        // SeekBar progress is offset by min; we fix the getter
        // by wrapping: actual_value = progress + min
        // But we store with offset, so we need to adjust in the callback
        seekBar.setTag(min);

        return result;
    }

    private static int dp(Context context, int v) {
        return (int) (v * context.getResources().getDisplayMetrics().density);
    }
}
