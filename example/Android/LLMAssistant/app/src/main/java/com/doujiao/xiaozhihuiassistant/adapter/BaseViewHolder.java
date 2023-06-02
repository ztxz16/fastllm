package com.doujiao.xiaozhihuiassistant.adapter;

import android.support.v7.widget.RecyclerView;
import android.view.View;

/**
 * Created by chenpengfei on 2016/10/27.
 */
public class BaseViewHolder extends RecyclerView.ViewHolder {

    private View iv;

    public BaseViewHolder(View itemView) {
        super(itemView);
        iv = itemView;
    }

    public View findViewById(int id) {
        return iv.findViewById(id);
    }
}
