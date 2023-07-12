package com.doujiao.xiaozhihuiassistant;

import android.Manifest;
import android.app.Activity;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.net.Uri;
import android.os.Build;
import android.os.Environment;
import android.os.Handler;
import android.os.Message;
import android.provider.Settings;
import android.support.annotation.NonNull;
import android.support.annotation.Nullable;
import android.support.v4.app.ActivityCompat;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.support.v7.widget.LinearLayoutManager;
import android.support.v7.widget.RecyclerView;
import android.text.TextUtils;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;
import android.widget.Toast;

import com.doujiao.core.AssistantCore;
import com.doujiao.xiaozhihuiassistant.adapter.MyAdapter;
import com.doujiao.xiaozhihuiassistant.databinding.ActivityMainBinding;
import com.doujiao.xiaozhihuiassistant.utils.PrefUtil;
import com.doujiao.xiaozhihuiassistant.utils.StatusBarUtils;
import com.doujiao.xiaozhihuiassistant.utils.UriUtils;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;

public class MainActivity extends AppCompatActivity {
    private static final int MSG_TYPE_RECVRESULT = 0;
    private static final int MSG_TYPE_INITMODEL_END = 1;
    private ActivityMainBinding binding;
    private volatile boolean mIsRunning = false;
    private EditText mInputEt = null;
    private TextView mTvTips = null;
    private RecyclerView mRcyViewMess = null;
    private Button sendBtn;
    private String mInputContent = "";
    private String mOutPutContent = "";
    private boolean mIsInit = false;

    ArrayList<ChatMessage> messageList = new ArrayList<>();
    private MyAdapter myAdapter = null;

    //读写权限
    private static String[] PERMISSIONS_STORAGE = {Manifest.permission.WRITE_EXTERNAL_STORAGE};
    //请求状态码
    private final static int REQUEST_PERMISSION_CODE = 1;
    protected static Uri modelUri = null;

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
                        addMsgList(content,2);
                    }
                    if (index > 0) {
                        updateMsgList(content);
                    }
                    if(index == -1) {
//                        sendBtn.setEnabled(true);
                    }
                }
                    break;
                case MSG_TYPE_INITMODEL_END:
                    String model = (String)msg.obj;
                    if (TextUtils.isEmpty(model)) {
                        Toast.makeText(getApplicationContext(), "模型不正确！请下载正确模型。", Toast.LENGTH_SHORT).show();
                    } else {
                        sendBtn.setEnabled(true);
                        mInputEt.setText("");
                        mTvTips.setText(PrefUtil.getModelPath());
                        Toast.makeText(getApplicationContext(), "欢迎使用:"+model, Toast.LENGTH_SHORT).show();
                        mIsInit = true;
                    }
                    break;
            }
        }
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        binding = ActivityMainBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());
        PrefUtil.initPref(getApplicationContext());
        initView();
        if (Build.VERSION.SDK_INT > Build.VERSION_CODES.LOLLIPOP) {
            if (ActivityCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE) == PackageManager.PERMISSION_GRANTED) {
                initModel();
            } else {
                sendBtn.setEnabled(false);
                ActivityCompat.requestPermissions(this, PERMISSIONS_STORAGE, REQUEST_PERMISSION_CODE);
            }
        }
        StatusBarUtils.hideStatusBar(getWindow(),true);
    }

    private void initModel() {
        //init:
        String modelPath = PrefUtil.getModelPath();
        if (!modelPath.isEmpty()) {
            mTvTips.setText("模型加载中..");
            new Thread(new Runnable() {
                @Override
                public void run() {
                    if(modelUri != null) {
                        try {
                            Log.d(MainActivity.class.getName(), " writing model file to " + modelPath);
                            InputStream inputStream = getContentResolver().openInputStream(modelUri);
                            FileOutputStream fileOutputStream = new FileOutputStream(modelPath);
                            byte[] buffer = new byte[1024];
                            int bytesRead;
                            while ((bytesRead = inputStream.read(buffer)) != -1) {
                                fileOutputStream.write(buffer, 0, bytesRead);
                            }
                            fileOutputStream.close();
                            inputStream.close();
                            Log.d("@@@", "model file write in " + modelPath);
                        } catch (IOException e) {
                            throw new RuntimeException(e);
                        }
                    }
                    String path = PrefUtil.getModelPath();//Environment.getExternalStorageDirectory().getAbsolutePath() + "/chatglm-6b-int4.bin";
                    File file = new File(path);
                    if(!file.exists() || file.length() < 1024 * 1024 * 1024) {
                        PrefUtil.setModelPath("");
                        String modelType = "";
                        Message msg = Message.obtain();
                        msg.obj = modelType;
                        msg.what = MSG_TYPE_INITMODEL_END;
                        mHandler.sendMessage(msg);
                        return;
                    }
                    String modelType = AssistantCore.getInstance().initLLM(path, new AssistantCore.runtimeResult() {
                        @Override
                        public void callbackResult(int index, String content) {
                            Message msg = Message.obtain();
                            msg.what = MSG_TYPE_RECVRESULT;
                            msg.arg1 = index;
                            msg.obj = content;
                            mHandler.sendMessage(msg);
                        }
                    });
                    Log.d("@@@","model:"+modelType);
                    Message msg = Message.obtain();
                    msg.obj = modelType;
                    msg.what = MSG_TYPE_INITMODEL_END;
                    mHandler.sendMessage(msg);
                }
            }).start();
        } else {
            mTvTips.setText(getString(R.string.app_model_tips));
        }
    }

    public void initView() {
        mTvTips = findViewById(R.id.tv_tips);
        mInputEt = findViewById(R.id.edit_input);
        mRcyViewMess = findViewById(R.id.rv_msgs);
        mRcyViewMess.setLayoutManager(new LinearLayoutManager(this));
        myAdapter = new MyAdapter(this);
        mRcyViewMess.setAdapter(myAdapter);
        myAdapter.setMessages(messageList);
        sendBtn = findViewById(R.id.btn_send);
        if (!mIsInit)
            sendBtn.setEnabled(false);
        sendBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                sendMsg();
            }
        });
        Button btnSel = findViewById(R.id.btn_sel);
        btnSel.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                if (!mIsRunning) {
                    Intent intent = new Intent(Intent.ACTION_GET_CONTENT);
                    intent.setType("*/*");
                    startActivityForResult(Intent.createChooser(intent, "需要选择文件"), 1);
                } else {
                    Toast.makeText(getApplicationContext(),"Is Running...",Toast.LENGTH_SHORT).show();
                }
            }
        });
    }

    public void updateMsgList(String content) {
        ChatMessage msg = messageList.get(messageList.size() - 1);
        messageList.remove(messageList.size() - 1);
        msg.setContent(msg.getContent() + content);
        messageList.add(msg);
        myAdapter.setMessages(messageList);
        myAdapter.notifyItemChanged(messageList.size() - 1);
    }

    public void addMsgList(String content,int role) {
        messageList.add(new ChatMessage(content,role));
        myAdapter.setMessages(messageList);
        myAdapter.notifyItemInserted(messageList.size());
    }

    public void sendMsg() {
        mInputContent = mInputEt.getText().toString();
        if (mInputContent.isEmpty()) {
            Toast.makeText(getApplicationContext(),"输入不能为空",Toast.LENGTH_SHORT).show();
            return;
        }
        if (!mIsRunning) {
            mIsRunning = true;
            addMsgList(mInputContent,1);
            mInputEt.setText("");
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

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (resultCode == Activity.RESULT_OK) {
            Uri uri = data.getData();
            String pathString = UriUtils.getPath(this, uri);
            modelUri = null;
            if(pathString == null || pathString.length()==0 || !pathString.startsWith("/")){
                pathString = getApplicationContext().getExternalFilesDir("models") + "/model2.flm";
                modelUri = uri;
            }
            Log.d(MainActivity.class.getName(), "uri: " + uri  + " pathString");
            PrefUtil.setModelPath(pathString);
            initModel();
        }
    }
}
