// Adapt the official Claude Agent SDK to Launcher's native session protocol.
import fs from "node:fs";
import path from "node:path";
import {pathToFileURL} from "node:url";
import {createRequire} from "node:module";
import {execFileSync} from "node:child_process";
import {randomUUID} from "node:crypto";
import readline from "node:readline";

const sdkPath = process.argv[2];
const sdk = await import(pathToFileURL(sdkPath).href);
if (process.argv.includes("--version")) {
  const require = createRequire(sdkPath);
  const musl = process.platform === "linux" && !process.report.getReport().header.glibcVersionRuntime;
  const platform = `${process.platform}-${process.arch}${musl ? "-musl" : ""}`;
  const binary = require.resolve(`@anthropic-ai/claude-agent-sdk-${platform}/${process.platform === "win32" ? "claude.exe" : "claude"}`);
  process.stdout.write(execFileSync(binary, ["--version"]));
  process.exit(0);
}
const service = JSON.parse(process.env.FTLLM_CLAUDE_SERVICE);
delete process.env.FTLLM_CLAUDE_SERVICE;
const home = path.join(process.env.CLAUDE_CONFIG_DIR, "launcher-sessions");
fs.mkdirSync(home, {recursive:true});
const indexPath = path.join(home, "index.json");
const metadata = fs.existsSync(indexPath) ? JSON.parse(fs.readFileSync(indexPath, "utf8")) : {};
const sessions = new Map(), active = new Map(), approvals = new Map();
const emit = message => process.stdout.write(JSON.stringify(message) + "\n");
const event = (method, params) => emit({method, params});
const diagnostic = error => String(error?.message || error).replaceAll(process.env.ANTHROPIC_AUTH_TOKEN, "[redacted]");
function atomic(file, value) {
  fs.writeFileSync(file + ".tmp", JSON.stringify(value), {mode:0o600});
  fs.renameSync(file + ".tmp", file);
}
function save(thread) {
  const {turns, ...summary} = thread;
  metadata[thread.id] = summary;
  atomic(path.join(home, thread.id + ".json"), thread);
  atomic(indexPath, metadata);
}
function get(id) {
  if (typeof id !== "string" || !Object.hasOwn(metadata, id) || !/^[a-f0-9-]{36}$/.test(id))
    throw new Error("This Claude Code conversation is unavailable.");
  if (!sessions.has(id)) {
    const thread = JSON.parse(fs.readFileSync(path.join(home, id + ".json"), "utf8"));
    for (const turn of thread.turns) if (turn.status === "inProgress") turn.status = "interrupted";
    sessions.set(id, thread);
  }
  return sessions.get(id);
}
function permission(thread, turn, tool, input, options, controller) {
  return new Promise(resolve => {
    const id = randomUUID();
    function finish(result) {
      if (!approvals.delete(id)) return;
      options.signal.removeEventListener("abort", aborted);
      event("serverRequest/resolved", {threadId:thread.id, requestId:id});
      resolve(result);
    }
    const aborted = () => finish({behavior:"deny", message:"The turn was cancelled.", interrupt:true});
    approvals.set(id, {threadId:thread.id, turnId:turn.id, finish, controller, input, tool});
    options.signal.addEventListener("abort", aborted, {once:true});
    if (options.signal.aborted) { aborted(); return; }
    const params = {threadId:thread.id, turnId:turn.id, itemId:options.toolUseID,
      cwd:thread.cwd, reason:options.title || options.decisionReason || tool,
      command:tool === "Bash" ? input.command : `${tool}\n${JSON.stringify(input, null, 2)}`};
    if (tool === "AskUserQuestion" && Array.isArray(input.questions)) {
      params.questions = input.questions.map((question, i) => ({id:String(i), ...question}));
      emit({id, method:"item/tool/requestUserInput", params});
    } else emit({id, method:"item/commandExecution/requestApproval",
      params:{...params, availableDecisions:["accept", "decline", "cancel"]}});
  });
}
function answer(id, result) {
  const pending = approvals.get(id);
  if (!pending) return;
  if (pending.tool === "AskUserQuestion" && result.answers) {
    const answers = Object.fromEntries(pending.input.questions.map((q, i) => [q.question,
      (result.answers[String(i)]?.answers || []).join(", ")]));
    pending.finish({behavior:"allow", updatedInput:{...pending.input, answers}});
  } else if (result.decision === "accept") pending.finish({behavior:"allow", updatedInput:pending.input});
  else {
    pending.finish({behavior:"deny", message:"The user declined this action.", interrupt:result.decision === "cancel"});
    if (result.decision === "cancel") pending.controller.abort();
  }
}
function blockItem(block, id, cwd) {
  if (block.type === "text") return {id, type:"agentMessage", text:block.text || ""};
  if (block.type === "thinking") return {id, type:"reasoning", content:[block.thinking || ""], summary:[]};
  if (block.type === "tool_use") return block.name === "Bash"
    ? {id:block.id, type:"commandExecution", command:block.input?.command || "", cwd, aggregatedOutput:"", status:"inProgress"}
    : {id:block.id, type:"mcpToolCall", tool:block.name, arguments:block.input || {}, status:"inProgress"};
}
async function run(thread, turn, text, effort, controller) {
  let query, flush, messageId = "", blocks = new Map();
  const contentStates = new Map();
  const params = {threadId:thread.id, turnId:turn.id};
  function contentState(id) {
    if (!contentStates.has(id)) contentStates.set(id, {pending:[], nextIndex:0});
    return contentStates.get(id);
  }
  function update(item, complete = false) {
    const index = turn.items.findIndex(value => value.id === item.id);
    if (index < 0) turn.items.push(item); else turn.items[index] = item;
    event(complete ? "item/completed" : "item/started", {...params, item});
    if (!flush) flush = setTimeout(() => { flush = null; save(thread); }, 500);
  }
  try {
    const previous = await sdk.getSessionInfo(thread.id, {dir:thread.cwd});
    query = sdk.query({prompt:text, options:{
      cwd:thread.cwd, model:service.modelName, abortController:controller,
      ...(previous ? {resume:thread.id} : {sessionId:thread.id}),
      systemPrompt:{type:"preset", preset:"claude_code"}, settingSources:[],
      includePartialMessages:true, persistSession:true, permissionMode:"default",
      ...(effort === "none" ? {thinking:{type:"disabled"}}
        : effort && effort !== "minimal" ? {effort, thinking:{type:"adaptive"}} : {}),
      // The CLI otherwise injects its generic "high" default, including for
      // models that expose no effort control. Pin the actual provider fields
      // per query; null preserves FastLLM's service default when unspecified.
      env:{...process.env, CLAUDE_CODE_EXTRA_BODY:JSON.stringify({
        thinking:effort === "none" ? {type:"disabled"} : effort ? {type:"adaptive"} : null,
        output_config:{effort:effort && effort !== "none" ? effort : null}})},
      canUseTool:(tool, input, options) => permission(thread, turn, tool, input, options, controller),
      stderr:chunk => process.stderr.write(chunk),
    }});
    for await (const message of query) {
      if (message.parent_tool_use_id) continue;
      if (message.type === "stream_event") {
        const e = message.event;
        if (e.type === "message_start") { messageId = e.message.id; blocks = new Map(); }
        if (e.type === "content_block_start") {
          const state = contentState(messageId);
          state.pending.push({index:e.index, type:e.content_block.type, id:e.content_block.id});
          state.nextIndex = Math.max(state.nextIndex, e.index + 1);
          const item = blockItem(e.content_block, `${messageId}:${e.index}`, thread.cwd);
          if (item) { blocks.set(e.index, item); update(item); }
        }
        if (e.type === "content_block_delta") {
          const item = blocks.get(e.index);
          if (!item) continue;
          if (e.delta.type === "text_delta") {
            item.text += e.delta.text;
            event("item/agentMessage/delta", {...params, itemId:item.id, delta:e.delta.text});
          } else if (e.delta.type === "thinking_delta") {
            item.content[0] += e.delta.thinking;
            event("item/reasoning/textDelta", {...params, itemId:item.id, contentIndex:0, delta:e.delta.thinking});
          }
        }
      } else if (message.type === "assistant") {
        // The SDK delivers completed blocks separately under the same message
        // id. Reuse their stream indices; content's array index restarts at 0.
        const state = contentState(message.message.id);
        for (const block of message.message.content) {
          const pending = state.pending.findIndex(value => value.type === block.type && value.id === block.id);
          const index = pending < 0 ? state.nextIndex++ : state.pending.splice(pending, 1)[0].index;
          const item = blockItem(block, `${message.message.id}:${index}`, thread.cwd);
          if (item) update(item, true);
        }
      } else if (message.type === "user" && Array.isArray(message.message.content)) {
        for (const block of message.message.content) {
          if (block.type !== "tool_result") continue;
          const item = turn.items.find(item => item.id === block.tool_use_id);
          if (!item) continue;
          const output = typeof block.content === "string" ? block.content : JSON.stringify(block.content, null, 2);
          update({...item, status:block.is_error ? "failed" : "completed",
            ...(item.type === "commandExecution" ? {aggregatedOutput:output} : {result:output})}, true);
        }
      } else if (message.type === "result" && message.is_error) {
        throw new Error((message.errors || [message.result || message.subtype]).join("\n"));
      }
    }
    turn.status = controller.signal.aborted ? "interrupted" : "completed";
  } catch (error) {
    turn.status = controller.signal.aborted ? "interrupted" : "failed";
    if (!controller.signal.aborted) turn.error = {message:diagnostic(error)};
  } finally {
    query?.close(); clearTimeout(flush);
    for (const pending of approvals.values()) if (pending.turnId === turn.id)
      pending.finish({behavior:"deny", message:"The turn has ended.", interrupt:true});
    thread.updatedAt = Date.now() / 1000;
    save(thread); active.delete(thread.id);
    event("turn/completed", {...params, turn});
  }
}
function call(method, params) {
  if (method === "initialize") return {};
  if (method === "thread/list") {
    const offset = Number(params.cursor || 0);
    if (!Number.isSafeInteger(offset) || offset < 0) throw new Error("Invalid session cursor.");
    const search = (params.searchTerm || "").toLowerCase();
    const data = Object.values(metadata).filter(t => Boolean(t.archived) === Boolean(params.archived)
      && `${t.name} ${t.preview} ${t.cwd}`.toLowerCase().includes(search)).sort((a,b) => b.updatedAt - a.updatedAt);
    return {data:data.slice(offset, offset + 100), nextCursor:offset + 100 < data.length ? String(offset + 100) : null};
  }
  if (method === "thread/start") {
    if (!fs.statSync(params.cwd).isDirectory()) throw new Error("Choose an existing workspace directory on the server.");
    const thread = {id:randomUUID(), cwd:fs.realpathSync(params.cwd), name:"", preview:"", updatedAt:Date.now()/1000, turns:[]};
    sessions.set(thread.id, thread); save(thread); return {thread};
  }
  const thread = get(params.threadId);
  if (["thread/read", "thread/resume"].includes(method)) return {thread, reasoningEffort:thread.reasoningEffort};
  if (method === "thread/name/set") {
    if (!params.name?.trim()) throw new Error("Enter a conversation name.");
    thread.name = params.name.trim(); save(thread);
    event("thread/name/updated", {threadId:thread.id, threadName:thread.name}); return {};
  }
  if (method === "thread/archive") {
    if (active.has(thread.id)) throw new Error("Cancel the current turn before archiving.");
    thread.archived = true; save(thread); return {};
  }
  if (method === "turn/interrupt") {
    const pending = active.get(thread.id);
    if (pending && pending.turn.id !== params.turnId) throw new Error("The active turn has changed.");
    pending?.controller.abort(); return {};
  }
  if (method === "turn/start") {
    if (active.has(thread.id)) throw new Error("This conversation is already running.");
    if (thread.archived) throw new Error("This conversation is archived.");
    const turn = {id:randomUUID(), status:"inProgress", items:[]}, controller = new AbortController();
    thread.preview ||= params.text.slice(0,120); thread.reasoningEffort = params.effort || "";
    thread.updatedAt = Date.now()/1000; thread.turns.push(turn);
    const item = {id:`user:${turn.id}`, type:"userMessage", content:[{type:"text", text:params.text}]};
    turn.items.push(item); save(thread); active.set(thread.id, {turn, controller});
    event("turn/started", {threadId:thread.id, turn});
    event("item/completed", {threadId:thread.id, turnId:turn.id, item});
    setImmediate(() => run(thread, turn, params.text, params.effort, controller).catch(error => {
      process.stderr.write(diagnostic(error) + "\n"); process.exitCode = 1;
    }));
    return {turn};
  }
  throw new Error("Unsupported Claude Code operation.");
}
const input = readline.createInterface({input:process.stdin});
function shutdown() {
  for (const pending of active.values()) pending.controller.abort();
  for (const thread of sessions.values()) save(thread);
  input.close(); process.stdin.destroy();
}
process.on("SIGTERM", shutdown);
process.on("SIGINT", shutdown);
for await (const line of input) {
  let request;
  try {
    request = JSON.parse(line);
    if (!request.method) { answer(request.id, request.result); continue; }
    if (request.method === "initialized") continue;
    emit({id:request.id, result:call(request.method, request.params || {})});
  } catch (error) { emit({id:request?.id, error:{message:diagnostic(error)}}); }
}
shutdown();
