package com.doujiao.xiaozhihuiassistant.widget;

import android.app.Activity;
import android.graphics.Color;
import android.graphics.drawable.ColorDrawable;
import android.view.Gravity;
import android.view.View;
import android.view.ViewGroup;
import android.view.ViewTreeObserver;
import android.widget.PopupWindow;

/**
 * Created by chenpengfei on 2016/11/2.
 */
public class PromptViewHelper {

    private PromptViewManager promptViewManager;
    private Activity activity;
    private PopupWindow popupWindow;
    private boolean isShow;
    private OnItemClickListener onItemClickListener;

    public PromptViewHelper(Activity activity) {
        this.activity = activity;
    }

    public void setPromptViewManager(PromptViewManager promptViewManager) {
        this.promptViewManager = promptViewManager;
        this.promptViewManager.setOnItemClickListener(new OnItemClickListener() {
            @Override
            public void onItemClick(int position) {
                if(onItemClickListener != null && popupWindow != null) {
                    onItemClickListener.onItemClick(position);
                    popupWindow.dismiss();
                }
            }
        });
    }

    public void addPrompt(View srcView) {
        srcView.setOnLongClickListener(new View.OnLongClickListener() {
            @Override
            public boolean onLongClick(View v) {
                createPrompt(v);
                return true;
            }
        });
    }

    private void createPrompt(final View srcView) {
        final View promptView = promptViewManager.getPromptView();
        if(popupWindow == null)
            popupWindow =  new PopupWindow(activity);
        popupWindow.setWindowLayoutMode(ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT);
        popupWindow.setTouchable(true);
        popupWindow.setOutsideTouchable(true);
        popupWindow.setBackgroundDrawable( new ColorDrawable(Color.TRANSPARENT));
        popupWindow.setContentView(promptView);
        final int[] location = new int[2];
        promptView.getViewTreeObserver().addOnGlobalLayoutListener(new ViewTreeObserver.OnGlobalLayoutListener() {
            @Override
            public void onGlobalLayout() {
                if(!isShow && popupWindow.isShowing()) {
                    popupWindow.dismiss();
                    show(srcView, promptView, location);
                    isShow = true;
                }
            }
        });
        srcView.getLocationOnScreen(location);
        show(srcView, promptView, location);
    }

    public void show(View srcView, View promptView, int[] srcViewLocation) {
        int[] xy = promptViewManager.getLocation().calculateLocation.calculate(srcViewLocation, srcView, promptView);
        popupWindow.showAtLocation(srcView, Gravity.NO_GRAVITY, xy[0], xy[1]);
    }


    public static abstract class PromptViewManager {

        private View promptView;
        protected Activity activity;
        private String[] dataArray;
        private Location location;
        public OnItemClickListener onItemClickListener;

        public PromptViewManager(Activity activity, String[] dataArray, Location location) {
            this.activity = activity;
            this.dataArray = dataArray;
            this.location = location;
            init();
        }

        public void setOnItemClickListener(OnItemClickListener onItemClickListener) {
            this.onItemClickListener = onItemClickListener;
        }

        public void init() {
            promptView = inflateView();
            bindData(promptView, dataArray);
        }

        public abstract View inflateView();

        public abstract void bindData(View view, String[] dataArray);

        public View getPromptView() {
            return promptView;
        }

        public Location getLocation() {
            return location;
        }
    }

    public void setOnItemClickListener(OnItemClickListener onItemClickListener) {
        this.onItemClickListener = onItemClickListener;
    }

    public interface OnItemClickListener {
        void onItemClick(int position);
    }
}
