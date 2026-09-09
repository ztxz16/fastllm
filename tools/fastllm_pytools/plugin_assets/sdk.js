// Runs inside the opaque-origin plugin frame. Never receives host credentials.
(() => {
  const channel = new MessageChannel(), port = channel.port1;
  let sequence = 0;
  const pending = new Map();
  let connected;
  const ready = new Promise(resolve => { connected = resolve; });
  port.onmessage = ({data}) => {
    if (data.type === "connected") { connected(); port.postMessage({type:"ready"}); return; }
    if (data.type === "context") {
      document.documentElement.dataset.theme = data.value.theme;
      window.dispatchEvent(new CustomEvent("ftllm-context", {detail:data.value}));
      return;
    }
    const call = pending.get(data.id);
    if (!call) return;
    pending.delete(data.id); clearTimeout(call.timer);
    if (data.error) call.reject(new Error(data.error)); else call.resolve(data.value);
  };
  port.start();
  // The SDK executes before plugin code. A subsequent navigation cannot get
  // a new bridge: this port belongs to this document, not its WindowProxy.
  parent.postMessage({type:"ftllm:connect"}, "*", [channel.port2]);
  const call = async (capability, args = {}) => {
    await ready;
    if (pending.size >= 4) throw new Error("请等待当前操作完成");
    const id = ++sequence;
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => { pending.delete(id); reject(new Error("操作超时")); }, 180000);
      pending.set(id, {resolve, reject, timer});
      port.postMessage({id, capability, arguments:args});
    });
  };
  Object.defineProperty(window, "ftllm", {value:Object.freeze({call, ready}), writable:false});
  const failed = error => { if (port) port.postMessage({type:"error", message:String(error?.message || error).slice(0,1000)}); };
  window.addEventListener("error", event => failed(event.error || event.message));
  window.addEventListener("unhandledrejection", event => failed(event.reason));
})();
