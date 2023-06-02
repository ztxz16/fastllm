package com.doujiao.xiaozhihuiassistant;

/**
 * Created by chenpengfei on 2016/10/27.
 */
public class ChatMessage {

    private String content;

    private int type;

    public ChatMessage(String content, int type) {
        this.content = content;
        this.type = type;
    }

    public ChatMessage(String content) {
        this(content, 1);
    }


    public String getContent() {
        return content;
    }

    public void setContent(String content) {
        this.content = content;
    }

    public int getType() {
        return type;
    }

    public void setType(int type) {
        this.type = type;
    }


}
