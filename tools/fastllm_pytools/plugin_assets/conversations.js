// Editor conversations stay in the browser; publishing still uses the core's
// validated preview/apply APIs. IndexedDB accommodates complete file drafts.
export function conversationStore(namespace) {
  let database, pending = Promise.resolve(), revision = 0;
  function open() {
    if (!database) database = new Promise((resolve, reject) => {
      const request = indexedDB.open("ftllm-interface-editor", 1);
      request.onupgradeneeded = () => request.result.createObjectStore("conversations");
      request.onsuccess = () => {
        const db = request.result;
        db.onversionchange = () => { db.close(); database = null; };
        resolve(db);
      };
      request.onerror = () => { database = null; reject(request.error); };
      request.onblocked = () => reject(new Error("请关闭其他正在更新编辑记录的页面后重试"));
    });
    return database;
  }
  async function transaction(mode, value) {
    const db = await open();
    return new Promise((resolve, reject) => {
      const tx = db.transaction("conversations", mode), store = tx.objectStore("conversations");
      const request = store.get(namespace); let conflict;
      request.onsuccess = () => {
        if (mode === "readonly") revision = request.result?.revision || 0;
        else if ((request.result?.revision || 0) !== revision) {
          conflict = new Error("其他页面已更新编辑记录；请先保留本页内容，再重新打开编辑页"); tx.abort();
        } else store.put({...value, revision:revision + 1}, namespace);
      };
      tx.oncomplete = () => { if (mode === "readwrite") revision++; resolve(request.result); };
      tx.onerror = tx.onabort = () => reject(conflict || tx.error || new Error("无法保存编辑对话"));
    });
  }
  return {
    load:() => transaction("readonly"),
    save(value) {
      const snapshot = structuredClone(value);
      pending = pending.catch(() => {}).then(() => transaction("readwrite", snapshot));
      return pending;
    }
  };
}
