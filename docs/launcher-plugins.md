# Launcher 与工作室自定义界面

`ftllm launch` 的模型管理、下载、日志、硬件信息和工作室页面分别位于
`tools/fastllm_pytools/ui_plugins/` 下的文件夹中。原有页面布局、模型进程和会话数据保持兼容。
启动器主导航提供“自定义界面”入口，底层使用独立文件夹中的界面插件；工作室工具栏不再重复显示入口。

## 使用自然语言定制

1. 在 Launcher 中启动模型 API，打开“自定义界面”子页面。
2. 在左侧对话框输入修改要求并发送。“修改选项”中可选择参考界面、定制位置和保存名称，默认自动选择与命名。
3. 描述需求，例如“在主界面右上角增加硬件状态栏，每三秒刷新内存和显存用量”，
   “换成蓝色皮肤，把侧栏移到右边”，或“在工作室增加一个按钮，把当前对话的摘要插入输入框”。
4. 左侧按轮显示你的需求、助手的修改说明，以及成功、取消或失败的结果。修改过程中只显示简短状态与耗时，步骤摘要可以展开查看，不展示原始生成代码。
   右侧显示可以交互的整页预览，直接点击其中的导航切换页面、使用原有主题控件切换浅色/深色；上方只保留桌面/手机尺寸与刷新。
5. 生成完成后自动更新预览；展开“查看与编辑文件”也可以直接编辑，停顿后自动校验并刷新。
   确认效果后点击“应用修改”。继续描述要求时，模型会收到本次对话的历史需求、修改摘要与最新草稿（包括手动编辑）；生成失败或取消时保留上一份草稿。
6. “已有自定义项与恢复”中可以继续定制、停用、恢复上一版或恢复默认皮肤。自己创建的自定义项还可删除：确认后删除其文件与上一版本，界面立即卸载该项；内置项不可删除，编辑会话和草稿会保留。

“返回”及浏览器后退会回到原页面，工作室草稿和模型服务保持运行。生成中可以取消，离开子页面也会取消生成。
左上角“对话列表”可搜索、切换、重命名和删除编辑会话；页面只显示选中会话的消息。
“新建对话”创建空白会话，各自保留对话历史、未发送的输入、修改选项和待应用草稿，模型只接收当前会话的历史。
会话保存在当前浏览器的 IndexedDB 中，刷新或重新打开同一地址后可恢复；不同浏览器或不同服务地址不共享这些记录。
切换会重新校验草稿并恢复预览；手动编辑出错时仍保留原文和上一份有效画面。生成或应用过程中暂不能切换，可先取消生成。
删除会话前会确认，仅删除其记录和未应用草稿，已应用的界面不受影响。保存失败或其他页面更新了记录时会显示错误，避免静默覆盖。
历史不会被自动裁剪；超过对话长度限制或模型上下文不足时会明确报错。
在页面地址后加上 `#customize` 可以直接打开编辑子页面，独立 WebUI 也支持这种方式。独立 WebUI 预览整个工作室；Launcher 中预览整个应用。

整页预览复制当前界面及发布版模板，使用独立样式范围；不复制原生脚本和事件处理器。
预览中可点击导航、打开设置、展开侧栏和编辑表单，这些操作只影响预览副本。刷新或编辑文件后保留预览的页面与主题，表单恢复为当前应用的快照。
状态组件可通过 `hardware.read`、`runtime.read` 读取实时数据；
启动/停止模型、发送对话、插入草稿等操作在预览中不可用。需要更新当前应用状态的画面时点击“刷新预览”。
未打开过的工作室显示发布版的空白对话模板，不会为了预览创建会话。

生成预览不会写入文件。模型只返回文件内容，不获得 shell、Python 执行器或文件系统工具。
若模型 API 报告上下文不足、返回不完整 JSON，或生成了非法路径，界面显示错误，现有插件不变。
应用时检查预览对应的版本；插件若已被另一操作修改，则拒绝覆盖，需重新生成预览。
预览文件只存于进程内存，最多保留 8 份、每份有效期 10 分钟，不写入插件目录；“刷新预览”可续建预览。
无效编辑不能应用，并保留上一个有效画面。预览中的自定义脚本仍使用与正式组件相同的 iframe 隔离。

## 插件目录

默认用户目录是 `~/.fastllm/plugins`，启动时也可指定：

```bash
ftllm launch --plugins-dir /path/to/my-plugins
ftllm webui --api_base http://127.0.0.1:8000/v1 --plugins-dir /path/to/my-plugins
```

每个插件一个文件夹，文件夹名称必须与清单中的 `id` 相同：

```text
~/.fastllm/plugins/
  hardware-monitor/
    plugin.json
    index.html
    app.js
    styles.css
```

`plugin.json` 示例：

```json
{
  "apiVersion": 1,
  "id": "hardware-monitor",
  "name": "硬件监控",
  "entry": "index.html",
  "slot": "page",
  "capabilities": ["hardware.read"]
}
```

- `slot: "page"` 在 Launcher 中增加页面；`slot: "studio"` 在工作室中增加扩展区域。
- `slot: "topbar"`、`"sidebar"`、`"statusbar"` 分别在主界面右上角、侧栏、底栏增加常驻组件。
  切换页面不重载这些组件；未安装时不占用空间。独立 WebUI 只挂载工作室扩展和皮肤。
- `slot: "theme"` 修改主界面和工作室的配色、字体与圆角，以及启动器的侧栏和内容布局，详见下文。
- 页面插件可用 `replaces` 替换 `launch`、`download`、`logs`、`hardware` 或 `webui`。
  停用定制插件后恢复内置页面。同一页面有多个替换插件时，按插件 ID 排序使用第一个已启用的插件。
- ID 使用小写字母、数字、连字符，以字母开头，最多 48 字符。内置插件 ID 保留，不能覆盖。
- 一个插件最多 64 个 UTF-8 文本文件，总大小不超过 1 MiB。支持 HTML、JS、CSS、JSON、SVG、Markdown。
  禁止符号链接、绝对路径、隐藏文件和 `..` 路径。
- 原生内置插件随软件发布，使用 `native: true`；用户插件不能声明此字段。

用户插件可以直接在磁盘上编辑。界面每两秒检测一次目录变化，添加、修改、停用和移除只影响对应插件。
重新加载会销毁旧 iframe、关闭消息通道并取消旧前端请求，模型服务继续运行。会话仍由核心保存。
插件内未保存的临时界面状态会在该插件重载时消失，因此工作室扩展应通过核心接口操作对话。

清单暂时写坏时，在当前进程中保留最后一个有效版本并显示错误；第一次发现的无效插件不加载。
JS 运行错误显示在组件区域，自定义界面入口仍属于主界面。通过管理界面发布的版本保留一份历史，支持回退；
直接在磁盘上编辑不会自动生成历史版本。`.state.json`、`.history/` 和 `.lock` 由核心管理。

## 主界面组件和皮肤

右上角硬件状态栏使用普通 HTML/CSS/JS 插件，将清单改为：

```json
{
  "apiVersion": 1,
  "id": "hardware-monitor",
  "name": "硬件状态栏",
  "entry": "index.html",
  "slot": "topbar",
  "size": {"width": 280, "height": 44},
  "capabilities": ["hardware.read"]
}
```

三个主界面位置都支持 `size`：宽度 80–640、高度 24–160，单位为 CSS 像素。
默认宽 240；顶部默认高 44、侧栏高 96、底栏高 32。组件适应容器宽度，多个组件超出时在区域内滚动。
侧栏组件填满侧栏宽度，窄屏时改为横向排列。组件不能覆盖核心按钮，也不能直接修改主页面 DOM。

皮肤插件只需一个 `plugin.json`，不执行脚本，无需 HTML 入口或服务能力：

```json
{
  "apiVersion": 1,
  "id": "blue-skin",
  "name": "蓝色皮肤",
  "slot": "theme",
  "theme": {
    "light": {
      "background": "#f4f7ff", "surface": "#ffffff", "sidebar": "#eaf0ff",
      "primary": "#2563eb", "primaryHover": "#1d4ed8", "primarySoft": "#dbeafe"
    },
    "dark": {
      "background": "#101827", "surface": "#18233b", "sidebar": "#18233b",
      "primary": "#60a5fa", "primaryHover": "#3b82f6", "primarySoft": "#1e3a5f"
    },
    "layout": {"sidebarSide": "right", "sidebarWidth": 220, "radius": 12, "font": "system"}
  }
}
```

`light` 和 `dark` 分别对应原来的日间/黑夜模式。配色字段均可省略，未指定时沿用该模式默认值。
支持 `background`、`surface`、`surfaceMuted`、`text`、`mutedText`、`border`、`primary`、
`primaryHover`、`primaryText`、`primarySoft`、`sidebar`、`topbar`；颜色只接受 `#RGB` 或 `#RRGGBB`。

| 布局字段 | 可用值 | 作用范围 |
| --- | --- | --- |
| `sidebarSide` | `left` / `right` | 启动器桌面布局，窄屏保留顶部导航 |
| `sidebarWidth` | 180–320 | 启动器桌面侧栏宽度 |
| `contentWidth` | 720–1800 | 启动器页面最大宽度 |
| `spacing` | `compact` / `comfortable` | 启动器桌面内容间距 |
| `font` | `system` / `serif` / `mono` | 主界面与工作室字体 |
| `radius` | 0–24 | 主界面与工作室主要控件圆角 |

应用或启用皮肤时自动停用其他皮肤；编辑已停用的皮肤不会自动启用。手工放入多个已启用皮肤时，
按 ID 排序使用第一个。皮肤热更新不会重建工作室或清空草稿。自定义界面的“恢复默认皮肤”会停用所有皮肤，
保留页面、组件和模型服务；管理入口与编辑子页面有独立样式，不受自定义配色影响。

皮肤由核心将上述配置映射到指定样式属性，不接受任意 CSS、选择器或 JavaScript。
要同时增加功能和换肤，可以使用一个组件插件和一个皮肤插件，各自热更新、回退。

## 浏览器 SDK

核心在插件脚本之前注入 `window.ftllm`。HTML 可以引用插件内的相对 JS/CSS 路径，不需要引入 SDK。
接口采用异步调用，清单必须声明对应能力：

| 能力 | 参数 | 返回值 |
| --- | --- | --- |
| `hardware.read` | 无 | CPU、内存、GPU、NUMA、磁盘和构建信息 |
| `runtime.read` | 无 | 当前模型服务状态；独立 WebUI 返回模型名称 |
| `model.chat` | `messages`、可选 `max_tokens` | `{content, reasoning}`，调用当前模型 API |
| `studio.context` | 无 | `{conversation, draft}`，当前对话与输入框文本的副本 |
| `studio.insert` | `{text}` | 在工作室当前输入位置插入文本，返回 `{ok: true}` |

`model.chat` 只接受 system/user/assistant 文本消息，不执行工具，不接受任意 API 地址。
`studio.*` 需要已打开的工作室。插件无法直接访问主页面的会话对象或修改其内部状态。

硬件插件的 `app.js` 示例：

```javascript
async function refresh() {
  const output = document.querySelector("#status");
  try {
    const report = await ftllm.call("hardware.read");
    const gib = bytes => (bytes / 1024 ** 3).toFixed(1);
    output.textContent = `内存：${gib(report.memory.total - report.memory.available)} / ${gib(report.memory.total)} GiB`;
  } catch (error) {
    output.textContent = error.message;
  }
}
refresh();
setInterval(refresh, 3000);
```

对应 HTML 提供 `<p id="status">正在读取…</p>` 和 `<script src="app.js"></script>` 即可。
GPU 字段包括 `name`、`memoryTotalMiB`、`memoryFreeMiB`、`utilization` 和 `temperature`。

主题和语言通过事件传入，同时设置根元素的 `data-theme`：

```javascript
window.addEventListener("ftllm-context", event => {
  const {theme, locale, palette} = event.detail;
  // 使用插件自己的 CSS 和文案适配主题、语言。
  // palette 是当前皮肤对应模式的配色；恢复默认皮肤时为空对象。
});
```

## 核心边界

核心负责鉴权、模型进程生命周期、会话存储、插件校验/发布/回退、能力代理和恢复入口。
内置页面的业务 JS 和模板位于各自目录，公共导航、主题、对话 API 及受控硬件读取服务保持稳定接口。
内置模块属于经过发布验证的代码；停用内置页面隐藏入口内容，不会停止模型、下载任务或删除会话。

可编辑插件运行在不带 `allow-same-origin` 的 sandbox iframe 中；它们不能访问父页面 DOM、控制令牌、
cookie 或浏览器存储，不能直接发起网络 API 请求。消息通道绑定具体 iframe 的具体版本，核心再次校验能力。
插件静态资源可被浏览器读取，所以不要把 API 密钥或其他秘密写进插件源码。

目前支持前端页面与交互扩展，不运行用户提供的 Python 后端。需要新增系统权限或超出上述接口的能力时，
应在核心中增加一个经过校验的服务接口，再供插件调用，不能通过生成代码绕过边界。
这一边界保护应用内的修改流程，不替代操作系统文件权限，也不保证隔离浏览器死循环或资源耗尽。

## 验证

```bash
PYTHONPATH=tools:test/api:test python3 -m unittest test_ui_plugins test_launcher test_launcher_webui test_webui_helpers
PYTHONPATH=tools:test/api python3 -m unittest test_launcher_webui_browser
```

浏览器测试需要 Playwright 和 Chromium，覆盖既有工作室行为、热更新、父 DOM 隔离、工作室扩展与模型预览发布。
