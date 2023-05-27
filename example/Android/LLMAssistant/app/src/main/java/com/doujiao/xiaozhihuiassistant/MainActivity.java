package com.doujiao.xiaozhihuiassistant;

import android.Manifest;
import android.content.pm.PackageManager;
import android.os.Build;
import android.os.Environment;
import android.os.Handler;
import android.os.Message;
import android.support.annotation.NonNull;
import android.support.v4.app.ActivityCompat;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;
import android.widget.Toast;

import com.doujiao.core.AssistantCore;
import com.doujiao.xiaozhihuiassistant.databinding.ActivityMainBinding;

public class MainActivity extends AppCompatActivity {
    private static final int MSG_TYPE_RECVRESULT = 0;
    private static final int MSG_TYPE_INITMODEL_END = 1;
    private ActivityMainBinding binding;
    private volatile boolean mIsRunning = false;
    private EditText mInputEt = null;
    private TextView mOutputTv = null;
    private Button sendBtn;
    private String mInputContent = "";
    private String mOutPutContent = "";
    private boolean mIsInit = false;

    //读写权限
    private static String[] PERMISSIONS_STORAGE = {Manifest.permission.WRITE_EXTERNAL_STORAGE};
    //请求状态码
    private final static int REQUEST_PERMISSION_CODE = 1;

    Handler mHandler = new Handler() {
        @Override
        public void handleMessage(Message msg) {
            super.handleMessage(msg);
            int msgType = msg.what;
            switch (msgType) {
                case MSG_TYPE_RECVRESULT: {
                    int index = msg.arg1;
                    String content = (String) msg.obj;
                    if (index == 0) {
                        mOutPutContent = "ChatGLM:";
                    }
                    if (index > -1) {
                        mOutPutContent += content;
                    }
                    mOutputTv.setText(mOutPutContent);
                }
                    break;
                case MSG_TYPE_INITMODEL_END:
                    sendBtn.setEnabled(true);
                    mOutputTv.setText("初始化完成");
                    mIsInit = true;
                    break;
            }
        }
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        binding = ActivityMainBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());
        initView();
        if (Build.VERSION.SDK_INT > Build.VERSION_CODES.LOLLIPOP) {
            if (ActivityCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE) == PackageManager.PERMISSION_GRANTED) {
                initModel();
            } else {
                sendBtn.setEnabled(false);
                ActivityCompat.requestPermissions(this, PERMISSIONS_STORAGE, REQUEST_PERMISSION_CODE);
            }
        }
    }

    private void initModel() {
        //init:
        new Thread(new Runnable() {
            @Override
            public void run() {
                String path = Environment.getExternalStorageDirectory().getAbsolutePath() + "/chatglm-6b-int4.bin";
                AssistantCore.getInstance().initLLM(path, new AssistantCore.runtimeResult() {
                    @Override
                    public void callbackResult(int index, String content) {
                        Message msg = Message.obtain();
                        msg.what = MSG_TYPE_RECVRESULT;
                        msg.arg1 = index;
                        msg.obj = content;
                        mHandler.sendMessage(msg);
                    }
                });
                mHandler.sendEmptyMessage(MSG_TYPE_INITMODEL_END);
            }
        }).start();
    }

    public void initView() {
        mInputEt = findViewById(R.id.edit_input);
        mOutputTv = findViewById(R.id.tv_output);
        sendBtn = findViewById(R.id.btn_send);
        if (!mIsInit)
            sendBtn.setEnabled(false);
        sendBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                sendMsg();
            }
        });
    }

    public void sendMsg() {
        mInputContent = mInputEt.getText().toString();
        if (mInputContent.isEmpty()) {
            Toast.makeText(getApplicationContext(),"输入不能为空",Toast.LENGTH_SHORT).show();
            return;
        }
        if (!mIsRunning) {
            mIsRunning = true;
            new Thread(new Runnable() {
                @Override
                public void run() {
                    int ret = AssistantCore.getInstance().chat(mInputContent);
                    Log.d("@@@","chat end:"+ret);
                    mIsRunning = false;
                }
            }).start();
        } else {
            Toast.makeText(getApplicationContext(),"Is Running...",Toast.LENGTH_SHORT).show();
        }
    }
    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == REQUEST_PERMISSION_CODE) {
            boolean isGrant = true;
            for (int i = 0; i < permissions.length; i++) {
                int grantResult = grantResults[i];
                if (grantResult != PackageManager.PERMISSION_GRANTED) {
                    isGrant = false;
                }
            }
            if (isGrant) {
                initModel();
            }else {
                Toast.makeText(this, "请允许访问文件", Toast.LENGTH_SHORT).show();
            }
        }
    }
}