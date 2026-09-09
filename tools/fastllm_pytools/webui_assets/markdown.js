const parserURL = new URL("./marked.js", import.meta.url);
parserURL.search = new URL(import.meta.url).search;
const {marked} = await import(parserURL.href);

// Parse GFM with Marked, but build DOM nodes ourselves: message HTML is never
// inserted into the application as markup. HTML code runs only in the preview.
export function renderMarkdown(node, source, {t, onCopy, onPreview} = {}) {
  const doc = node.ownerDocument;
  const element = (tag, className) => {
    const result = doc.createElement(tag);
    if (className) result.className = className;
    return result;
  };
  const decode = text => {
    const decoder = element("textarea");
    decoder.innerHTML = String(text || "").replaceAll("<", "&lt;");
    return decoder.value;
  };
  const safeLink = value => {
    const href = decode(value).trim();
    try {
      const url = new URL(href, location.href);
      if (["http:", "https:", "mailto:"].includes(url.protocol)) return href;
    } catch (_) {}
    return null;
  };

  function codeBlock(token) {
    const block = element("div", "code-block");
    const head = element("div", "code-head");
    const label = element("span");
    const language = (token.lang || "").trim().split(/\s+/)[0];
    label.textContent = language || "code";
    const actions = element("div", "code-actions");
    if (["html", "htm", "text/html"].includes(language.toLowerCase()) && onPreview) {
      const preview = element("button", "preview-code");
      preview.type = "button";
      preview.textContent = t("html.preview");
      preview.onclick = () => onPreview(token.text);
      actions.append(preview);
    }
    if (onCopy) {
      const copy = element("button", "copy-code");
      copy.type = "button";
      copy.textContent = t("common.copy");
      copy.onclick = () => onCopy(token.text, copy);
      actions.append(copy);
    }
    head.append(label, actions);
    const pre = element("pre");
    const code = element("code");
    code.textContent = token.text;
    pre.append(code);
    block.append(head, pre);
    return block;
  }

  function append(parent, tokens) {
    for (const token of tokens || []) {
      let child;
      switch (token.type) {
        case "space": case "def": continue;
        case "code": parent.append(codeBlock(token)); continue;
        case "heading": child = element(`h${token.depth}`); break;
        case "paragraph": child = element("p"); break;
        case "blockquote": child = element("blockquote"); break;
        case "strong": case "em": case "del": child = element(token.type); break;
        case "hr": case "br": parent.append(element(token.type)); continue;
        case "codespan":
          child = element("code"); child.textContent = token.text; parent.append(child); continue;
        case "checkbox":
          child = element("input"); child.type = "checkbox";
          child.disabled = true; child.checked = token.checked;
          parent.append(child, doc.createTextNode(" ")); continue;
        case "list":
          child = element(token.ordered ? "ol" : "ul");
          if (token.ordered) child.start = token.start;
          for (const item of token.items) {
            const li = element("li", item.task ? "task-list-item" : "");
            append(li, item.tokens); child.append(li);
          }
          parent.append(child); continue;
        case "table": {
          const scroll = element("div", "markdown-table");
          scroll.tabIndex = 0;
          const table = element("table");
          const row = (cells, header) => {
            const tr = element("tr");
            cells.forEach((cell, index) => {
              const td = element(header ? "th" : "td");
              if (header) td.scope = "col";
              const align = token.align[index];
              if (["left", "center", "right"].includes(align)) td.className = `align-${align}`;
              append(td, cell.tokens); tr.append(td);
            });
            return tr;
          };
          const head = element("thead"); head.append(row(token.header, true));
          const body = element("tbody"); token.rows.forEach(cells => body.append(row(cells, false)));
          table.append(head, body); scroll.append(table); parent.append(scroll); continue;
        }
        case "link": {
          const href = safeLink(token.href);
          if (!href) { append(parent, token.tokens); continue; }
          child = element("a"); child.href = href;
          child.target = "_blank"; child.rel = "noopener noreferrer";
          if (token.title) child.title = decode(token.title);
          break;
        }
        case "image": {
          // Remote images remain explicit links, avoiding automatic requests
          // from model output and preserving Launcher's self-only image policy.
          const href = safeLink(token.href);
          child = element(href ? "a" : "span");
          if (href) { child.href = href; child.target = "_blank"; child.rel = "noopener noreferrer"; }
          child.textContent = decode(token.text) || href || "";
          parent.append(child); continue;
        }
        case "html":
          if (/^<br\s*\/?\s*>$/i.test(token.text.trim())) parent.append(element("br"));
          else parent.append(doc.createTextNode(token.text));
          continue;
        default:
          if (token.tokens) append(parent, token.tokens);
          else parent.append(doc.createTextNode(decode(token.text || token.raw)));
          continue;
      }
      append(child, token.tokens);
      parent.append(child);
    }
  }

  const content = doc.createDocumentFragment();
  append(content, marked.lexer(String(source || ""), {gfm: true, breaks: true}));
  node.replaceChildren(content);
}
