package com.doujiao.xiaozhihuiassistant.widget;

import android.annotation.TargetApi;
import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Path;
import android.graphics.Rect;
import android.graphics.RectF;
import android.os.Build;
import android.util.AttributeSet;
import android.view.MotionEvent;
import android.view.View;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by chenpengfei on 2016/10/27.
 */
public class PromptView extends View {

    private Paint mPaint;
    private int mSingleWidth = 120, mSingleHeight = 60;
    private int mLineWidth = 1;
    private int mArrowWidth =45, mArrowHeight = 20;
    private String[] mContentArray = null;
    private List<Rect> textRectList = new ArrayList<>();
    private List<RectF> rectFList = new ArrayList<>();
    private OnItemClickListener onItemClickListener;

    public PromptView(Context context) {
        super(context);
        init();
    }

    public PromptView(Context context, AttributeSet attrs, int defStyleAttr) {
        super(context, attrs, defStyleAttr);
    }

    public PromptView(Context context, AttributeSet attrs) {
        super(context, attrs);
        init();
    }

    @TargetApi(Build.VERSION_CODES.LOLLIPOP)
    public PromptView(Context context, AttributeSet attrs, int defStyleAttr, int defStyleRes) {
        super(context, attrs, defStyleAttr, defStyleRes);
    }

    private void init() {
        mPaint = new Paint();
        mPaint.setAntiAlias(true);
        mPaint.setStyle(Paint.Style.FILL);
        mPaint.setTextSize(30);
    }

    @Override
    protected void onMeasure(int widthMeasureSpec, int heightMeasureSpec) {
        int width = mSingleWidth * mContentArray.length + ((mContentArray.length - 1) * mLineWidth);
        setMeasuredDimension(width, mSingleHeight + mArrowHeight);
    }

    public void setContentArray(String[] contentArray) {
        mContentArray = contentArray;
        for(int m = 0; m < mContentArray.length; m++) {
            Rect rect = new Rect();
            mPaint.getTextBounds(mContentArray[m], 0, mContentArray[m].length(), rect);
            textRectList.add(rect);
        }
    }

    @Override
    protected void onDraw(Canvas canvas) {
        if(mContentArray == null) return;
        for(int i = 0; i < mContentArray.length; i++) {
            drawPromptRect(canvas, i);
        }
        //绘制下面的提示箭头
        drawArrow(canvas);
    }

    private void drawPromptRect(Canvas canvas, int i) {
        mPaint.setColor(Color.BLACK);
        RectF tipRectF = new RectF();
        tipRectF.left = mSingleWidth * i + i * mLineWidth;
        tipRectF.top = 0;
        tipRectF.right = tipRectF.left + mSingleWidth;
        tipRectF.bottom = mSingleHeight;
        canvas.drawRect(tipRectF, mPaint);

        rectFList.add(tipRectF);

        //绘制文本内容
        mPaint.setColor(Color.WHITE);
        canvas.drawText(mContentArray[i], (tipRectF.right - tipRectF.left - textRectList.get(i).width()) / 2 + tipRectF.left, getFontBaseLine(), mPaint);

        if(i == mContentArray.length - 1) return;
        //绘制白线
        RectF lineRectf = new RectF();
        lineRectf.left = tipRectF.right;
        lineRectf.top = 0;
        lineRectf.right = lineRectf.left + mLineWidth;
        lineRectf.bottom = mSingleHeight;
        canvas.drawRect(lineRectf, mPaint);
    }

    private void drawArrow(Canvas canvas) {
        mPaint.setColor(Color.BLACK);
        Path arrowPath = new Path();
        int start = (getWidth() - mArrowWidth) / 2;
        arrowPath.moveTo(start, mSingleHeight);
        arrowPath.lineTo(start + mArrowWidth / 2, mSingleHeight + mArrowHeight);
        arrowPath.lineTo(start + mArrowWidth, mSingleHeight);
        canvas.drawPath(arrowPath, mPaint);
    }

    private int getFontBaseLine() {
        Paint.FontMetricsInt fontMetrics = mPaint.getFontMetricsInt();
        return (getMeasuredHeight() - mArrowHeight) / 2 + (fontMetrics.descent- fontMetrics.ascent) / 2 - fontMetrics.descent;
    }

    @Override
    public boolean onTouchEvent(MotionEvent event) {
        switch (event.getAction()) {
            case MotionEvent.ACTION_UP:
                for(int n = 0; n < rectFList.size(); n++) {
                    RectF rect = rectFList.get(n);
                    if(event.getX() > rect.left && event.getX() < rect.right && event.getY() > rect.top && event.getY() < rect.bottom && onItemClickListener != null) {
                        onItemClickListener.onItemClick(n);
                        break;
                    }
                }
                break;
        }
        return true;
    }

    public void setOnItemClickListener(OnItemClickListener onItemClickListener) {
        this.onItemClickListener = onItemClickListener;
    }

    public interface OnItemClickListener {
        void onItemClick(int position);
    }
}
