# LingBot World 2.0 原生部署

此入口面向 `lingbot-world-v2-1.3b-causal-fast`。模型接收起始图片、场景描述和相机动作，按片段生成连续画面，并保存世界会话的历史状态。

推理由 C++、CUDA 和仓库已有的 cuBLAS、CUTLASS 算子执行。HTTP 服务只使用 Python 标准库。PyTorch 与 safetensors 仅用于一次性的 `.pth` 权重转换及官方精度对照，不是推理服务依赖。

## 准备权重

从 [官方小模型仓库](https://huggingface.co/robbyant/lingbot-world-v2-1.3b-causal-fast) 获取 `transformers/` 中的全部分片和索引。从[官方共享资源仓库](https://huggingface.co/robbyant/lingbot-world-v2-14b-causal-fast) 获取 T5、Wan VAE 与 UMT5 分词器，无需下载 14B DiT。

也可以从对应的 ModelScope 仓库下载相同文件。目录结构如下：

```text
LingBot-World-2.0-1.3B/
├── transformers/
│   ├── model.safetensors.index.json
│   └── *.safetensors
├── models_t5_umt5-xxl-enc-bf16.pth
├── Wan2.1_VAE.pth
└── google/umt5-xxl/
    ├── tokenizer.json
    ├── tokenizer_config.json
    ├── special_tokens_map.json
    └── spiece.model
```

在权重准备环境中运行：

```bash
MODEL_DIR=/path/to/LingBot-World-2.0-1.3B
python3 tools/lingbot_world/convert_assets.py "$MODEL_DIR"
```

转换生成 `native/t5.safetensors`、`native/vae.safetensors` 和校验记录 `native/conversion.json`，保留原文件。将整个目录复制到部署环境即可。

## 构建

需要 C++17 编译器、CMake 和 CUDA Toolkit。`CUDA_ARCH` 按部署设备的计算能力填写。

```bash
cmake -S . -B build-lingbot \
  -DUSE_CUDA=ON -DUSE_NUMAS=OFF \
  -DBUILD_LINGBOT_WORLD=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-lingbot --target fastllm_lingbot lingbot_world_generate -j
```

生成原生库 `build-lingbot/libfastllm_lingbot.so`，以及 `build-lingbot/src/models/lingbot_world/lingbot_world_generate`。

## 启动 HTTP 与 Demo

```bash
CUDA_VISIBLE_DEVICES=0 python3 tools/lingbot_world/server.py \
  --model "$MODEL_DIR" \
  --library build-lingbot/libfastllm_lingbot.so \
  --host 0.0.0.0 --port 18009 \
  --output ./lingbot-output
```

浏览器访问 `http://localhost:18009/`，选择起始图片，填写描述，再创建场景。页面提供平移、前后移动、抬头低头、左右转向、继续观察及修改描述。W/A/S/D 控制移动，方向键控制视角，Q/E 控制升降。

可用 `--api-key` 设置 Bearer 密钥；页面的连接设置中填写同一密钥。`--max-worlds` 控制常驻世界数量，达到上限后先删除不用的会话。服务重启后会话需重新创建，输出 PNG 文件仍保留。

历史缓存会在推理阶段之间无损暂存在主机内存，计算时恢复到 GPU。部署时需同时预留系统内存；画面尺寸、历史长度和会话数量都会影响内存占用。

## HTTP 接口

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| GET | `/health` | 加载状态 |
| GET | `/v1/models` | 模型列表 |
| POST | `/v1/worlds` | 创建世界并生成第一段画面 |
| GET | `/v1/worlds/{id}` | 世界状态与最近一次结果 |
| POST | `/v1/worlds/{id}/actions` | 移动、转向或更新描述 |
| DELETE | `/v1/worlds/{id}` | 释放世界 |
| GET | `/v1/jobs/{id}` | 查询异步任务 |
| GET | `/v1/worlds/files/{name}.png` | 获取生成帧 |
| POST | `/v1/chat/completions` | 使用 messages 格式调用 |

创建请求中的 `image` 为 PNG data URL。Demo 会将浏览器可读取的图片转换为 PNG。`width`、`height` 为 64～832 内的 16 的倍数；`seed` 为非负整数。可选 `intrinsics` 为输出分辨率对应的 `[fx, fy, cx, cy]`。

```python
import base64
import json
import time
import urllib.request

base = 'http://localhost:18009'

def request(path, body=None):
    payload = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(base + path, data=payload,
        headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req) as response:
        return json.load(response)

with open('scene.png', 'rb') as source:
    image = 'data:image/png;base64,' + base64.b64encode(source.read()).decode()
job = request('/v1/worlds', {
    'model': 'lingbot-world-2.0-1.3b',
    'image': image,
    'prompt': 'A quiet lakeside landscape under a blue sky.',
    'width': 576, 'height': 320, 'seed': 42,
})
while job['status'] not in ('completed', 'failed'):
    time.sleep(0.5)
    job = request('/v1/jobs/' + job['id'])
if job['status'] == 'failed':
    raise RuntimeError(job['error'])
result = job['result']
print(result['world_id'], result['fps'], result['frames'])
next_job = request('/v1/worlds/' + result['world_id'] + '/actions',
                   {'action': 'forward'})
```

创建、动作和删除请求返回 HTTP 202 及任务对象。任务完成后的 `result` 包含世界 ID、片段序号、宽高、帧率、帧数量和 PNG URL 数组。按 URL 顺序与返回帧率播放；这些是视频帧，不是可在浏览器独立渲染的三维网格。

失败任务的 `error` 包含原因。若返回 `world_deleted: true`，该会话已释放，需要重新创建；例如生成完成后输出目录写入失败，服务会释放已前进但无法交付画面的会话。

动作可选 `stay`、`forward`、`backward`、`left`、`right`、`up`、`down`、`look_left`、`look_right`、`look_up`、`look_down`。`prompt` 可随动作更新；省略它则保留当前描述。同一世界有任务未完成时，再次提交动作会返回 HTTP 409。

### messages 格式

创建世界时，在最后一条 user 消息中提供描述和起始图片：

```json
{
  "model": "lingbot-world-2.0-1.3b",
  "messages": [{
    "role": "user",
    "content": [
      {"type": "text", "text": "A quiet lakeside landscape."},
      {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
    ]
  }],
  "width": 576,
  "height": 320,
  "seed": 42
}
```

此接口等待生成完成后返回 `chat.completion`，`choices[0].message.content` 是包含帧 URL 等字段的 JSON 字符串，顶层附带 `world_id`。后续操作传入该 `world_id` 和 `action`；不修改描述时可使用 `{"role":"user","content":[]}`。

模型只消费最后一条 user 消息作为当前图片/场景描述，历史画面由世界会话维护。它不执行聊天模型的文本推理或工具调用，不支持 `stream: true`；需要查询进度时使用异步世界接口。

## 命令行生成

```bash
CUDA_VISIBLE_DEVICES=0 build-lingbot/src/models/lingbot_world/lingbot_world_generate \
  --model "$MODEL_DIR" --image scene.png \
  --prompt 'A quiet lakeside landscape.' \
  --width 576 --height 320 --seed 42 \
  --actions stay,forward,look_left --output ./lingbot-frames
```

输出 `metadata.json` 和按顺序编号的 PNG 帧。

## 精度验证入口

`test/lingbot_world/` 包含官方分词器、T5、DiT、Wan VAE 与连续推演的对照脚本。测试环境需要官方仓库及其 Python 依赖，部署服务不需要这些依赖。连续推演测试读取 `lingbot_world_generate --trace DIR` 导出的噪声与中间结果，用相同输入运行官方代码。

```bash
python3 test/lingbot_world/compare_tokenizer.py --help
python3 test/lingbot_world/compare_ops.py --help
python3 test/lingbot_world/compare_text.py --help
python3 test/lingbot_world/compare_vae.py --help
python3 test/lingbot_world/prepare_inputs.py --help
python3 test/lingbot_world/compare_dit.py --help
python3 test/lingbot_world/compare_pipeline.py --help
```
