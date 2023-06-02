package com.doujiao.xiaozhihuiassistant.adapter;

import android.app.Activity;
import android.support.v7.widget.RecyclerView;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;
import android.widget.Toast;


import com.doujiao.xiaozhihuiassistant.ChatMessage;
import com.doujiao.xiaozhihuiassistant.R;
import com.doujiao.xiaozhihuiassistant.utils.StatusBarUtils;
import com.doujiao.xiaozhihuiassistant.widget.ChatPromptViewManager;
import com.doujiao.xiaozhihuiassistant.widget.Location;
import com.doujiao.xiaozhihuiassistant.widget.PromptViewHelper;

import java.util.List;

public class MyAdapter extends RecyclerView.Adapter<BaseViewHolder> {

    private List<ChatMessage> mChatMessageList = null;
    private Activity mActivity;

    public MyAdapter(Activity activity) {
        mActivity = activity;
    }

    public void setMessages(List<ChatMessage> chatMessageList) {
        mChatMessageList = chatMessageList;
    }

    @Override
    public BaseViewHolder onCreateViewHolder(ViewGroup parent, int viewType) {
        if(viewType == 1) {
            return new LeftViewHolder(View.inflate(mActivity, R.layout.activity_item_left, null));
        } else {
            return new RightViewHolder(View.inflate(mActivity, R.layout.activity_item_right, null));
        }
    }

    @Override
    public int getItemCount() {
        return mChatMessageList.size();
    }

    @Override
    public int getItemViewType(int position) {
        return mChatMessageList.get(position).getType();
    }

    @Override
    public void onBindViewHolder(BaseViewHolder holder, int position) {
        PromptViewHelper pvHelper = new PromptViewHelper(mActivity);
        ChatMessage chatMessage = mChatMessageList.get(position);
        if(holder instanceof LeftViewHolder) {
            LeftViewHolder leftViewHolder = (LeftViewHolder) holder;
            leftViewHolder.tv.setText(chatMessage.getContent());
            pvHelper.setPromptViewManager(new ChatPromptViewManager(mActivity));
        }
        if(holder instanceof RightViewHolder) {
            RightViewHolder rightViewHolder = (RightViewHolder) holder;
            rightViewHolder.tv.setText(chatMessage.getContent());
            pvHelper.setPromptViewManager(new ChatPromptViewManager(mActivity, Location.TOP_RIGHT));
        }
        pvHelper.addPrompt(holder.itemView.findViewById(R.id.textview_content));
        pvHelper.setOnItemClickListener(new PromptViewHelper.OnItemClickListener() {
            @Override
            public void onItemClick(int position) {
                String str = "";
                switch (position) {
                    case 0:
                        str = "已复制到剪贴板!";
                        TextView tv = holder.itemView.findViewById(R.id.textview_content);
                        StatusBarUtils.copyStr2ClibBoard(mActivity.getApplicationContext(), tv.getText().toString());
                        break;
                }
                Toast.makeText(mActivity,  str, Toast.LENGTH_SHORT).show();
            }
        });
    }

    class LeftViewHolder extends BaseViewHolder {

        TextView tv;

        public LeftViewHolder(View view) {
            super(view);
            tv = (TextView) findViewById(R.id.textview_content);
        }
    }

    class RightViewHolder extends BaseViewHolder {

        TextView tv;

        public RightViewHolder(View view) {
            super(view);
            tv = (TextView) findViewById(R.id.textview_content);
        }
    }
}
