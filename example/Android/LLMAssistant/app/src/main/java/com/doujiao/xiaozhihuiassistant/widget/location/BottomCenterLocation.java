package com.doujiao.xiaozhihuiassistant.widget.location;

import android.view.View;

/**
 * Created by chenpengfei on 2016/11/2.
 */
public class BottomCenterLocation implements ICalculateLocation {

    @Override
    public int[] calculate(int[] srcViewLocation, View srcView, View promptView) {
        int[] location = new int[2];
        int offset = (promptView.getWidth() - srcView.getWidth()) / 2;
        location[0] = srcViewLocation[0] - offset;
        location[1] = srcViewLocation[1] + promptView.getHeight();
        return location;
    }
}
