import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import { readFile } from "node:fs/promises";

interface ProjectFile {
	index: number;
	name: string;
	path: string;
	size: number;
	truncated?: boolean;
}

const MAX_READ_LINES = 400;
const MAX_READ_CHARS = 24000;
const MAX_SEARCH_MATCHES = 60;
const MAX_SEARCH_CHARS = 18000;
const webBridgeUrl = process.env.FTLLM_AGENT_WEB_BRIDGE_URL || "";
const webBridgeToken = process.env.FTLLM_AGENT_WEB_BRIDGE_TOKEN || "";
const bridgeVersion = process.env.FTLLM_AGENT_BRIDGE_VERSION || "unknown";
const piVersion = process.env.FTLLM_AGENT_PI_VERSION || "unknown";

async function callWebBridge(payload: Record<string, unknown>): Promise<any> {
	if (!webBridgeUrl || !webBridgeToken) throw new Error("FastLLM web bridge is unavailable");
	const response = await fetch(`${webBridgeUrl}/tool`, {
		method: "POST",
		headers: {
			"Authorization": `Bearer ${webBridgeToken}`,
			"Content-Type": "application/json",
		},
		body: JSON.stringify(payload),
	});
	let result: any;
	try {
		result = await response.json();
	} catch {
		throw new Error(`FastLLM web bridge returned HTTP ${response.status}`);
	}
	if (!response.ok) throw new Error(String(result?.error || `HTTP ${response.status}`));
	return result;
}

function loadManifest(): ProjectFile[] {
	const raw = process.env.FTLLM_AGENT_PROJECT_MANIFEST || "[]";
	const parsed = JSON.parse(raw);
	if (!Array.isArray(parsed)) throw new Error("project manifest must be an array");
	return parsed.map((item, offset) => {
		if (!item || typeof item !== "object") throw new Error("invalid project manifest entry");
		const index = Number(item.index);
		const path = String(item.path || "");
		if (!Number.isInteger(index) || index !== offset + 1 || !path) {
			throw new Error("invalid project manifest index or path");
		}
		return {
			index,
			name: String(item.name || `file-${index}`),
			path,
			size: Number(item.size || 0),
			truncated: Boolean(item.truncated),
		};
	});
}

function entry(files: ProjectFile[], index: number): ProjectFile {
	const selected = files[index - 1];
	if (!selected || selected.index !== index) throw new Error(`unknown project file index: ${index}`);
	return selected;
}

function lines(text: string): string[] {
	return text.replace(/\r\n/g, "\n").replace(/\r/g, "\n").split("\n");
}

function numbered(file: ProjectFile, body: string[], start: number): string {
	const rendered = body.map((line, offset) => `${String(start + offset).padStart(6)} | ${line}`).join("\n");
	return `[代码${file.index}:L${start}-L${start + Math.max(0, body.length - 1)}] ${file.name}\n${rendered}`;
}

export default function projectTools(pi: ExtensionAPI) {
	const files = loadManifest();

	pi.registerTool({
		name: "runtime_info",
		label: "FastLLM Agent Runtime",
		description: "Return the packaged Pi runtime version and the number of isolated project files.",
		parameters: Type.Object({}),
		async execute() {
			const details = {
				bridge: "ftllm-agent-runtime",
				bridgeVersion,
				piVersion,
				fileCount: files.length,
				web: Boolean(webBridgeUrl),
			};
			return {
				content: [{
					type: "text",
					text: JSON.stringify(details),
				}],
				details,
			};
		},
	});

	pi.registerTool({
		name: "list_project_files",
		label: "List project files",
		description: "List the read-only source files uploaded for this request. Call this before choosing a file.",
		parameters: Type.Object({}),
		async execute() {
			const text = files.map((file) =>
				`[代码${file.index}] ${file.name} (${file.size} bytes${file.truncated ? ", snapshot truncated" : ""})`,
			).join("\n");
			return { content: [{ type: "text", text }], details: { files } };
		},
	});

	pi.registerTool({
		name: "read_project_file",
		label: "Read project file",
		description: `Read a bounded line range from one isolated project file. At most ${MAX_READ_LINES} lines are returned.`,
		parameters: Type.Object({
			file: Type.Integer({ minimum: 1, description: "File index returned by list_project_files" }),
			start_line: Type.Optional(Type.Integer({ minimum: 1, description: "First line, inclusive; defaults to 1" })),
			end_line: Type.Optional(Type.Integer({ minimum: 1, description: "Last line, inclusive" })),
		}),
		async execute(_toolCallId, params) {
			const file = entry(files, params.file);
			const content = lines(await readFile(file.path, "utf8"));
			const start = Math.min(content.length || 1, Math.max(1, params.start_line || 1));
			const requestedEnd = params.end_line || start + MAX_READ_LINES - 1;
			const end = Math.min(content.length, Math.max(start, requestedEnd), start + MAX_READ_LINES - 1);
			let output = numbered(file, content.slice(start - 1, end), start);
			if (output.length > MAX_READ_CHARS) output = `${output.slice(0, MAX_READ_CHARS)}\n[output truncated]`;
			return { content: [{ type: "text", text: output }], details: { file: file.index, startLine: start, endLine: end } };
		},
	});

	pi.registerTool({
		name: "search_project_files",
		label: "Search project files",
		description: "Search for a literal, case-insensitive string across the isolated project files and return matching source lines.",
		parameters: Type.Object({
			query: Type.String({ minLength: 1, maxLength: 200, description: "Literal text to find" }),
			file: Type.Optional(Type.Integer({ minimum: 1, description: "Optional file index" })),
		}),
		async execute(_toolCallId, params) {
			const needle = params.query.toLocaleLowerCase();
			const selected = params.file ? [entry(files, params.file)] : files;
			const matches: string[] = [];
			for (const file of selected) {
				const content = lines(await readFile(file.path, "utf8"));
				for (let offset = 0; offset < content.length; offset += 1) {
					if (!content[offset].toLocaleLowerCase().includes(needle)) continue;
					matches.push(`[代码${file.index}:L${offset + 1}] ${file.name}: ${content[offset]}`);
					if (matches.length >= MAX_SEARCH_MATCHES) break;
				}
				if (matches.length >= MAX_SEARCH_MATCHES) break;
			}
			let output = matches.length ? matches.join("\n") : `No literal matches for: ${params.query}`;
			if (output.length > MAX_SEARCH_CHARS) output = `${output.slice(0, MAX_SEARCH_CHARS)}\n[output truncated]`;
			return { content: [{ type: "text", text: output }], details: { matches: matches.length } };
		},
	});

	if (webBridgeUrl) {
		pi.registerTool({
			name: "web_search",
			label: "Search the web",
			description: "Search the live public web. Use precise queries, include the current year for recent events, and run follow-up searches when results are ambiguous.",
			parameters: Type.Object({
				query: Type.String({ minLength: 1, maxLength: 500, description: "Precise search-engine query" }),
				limit: Type.Optional(Type.Integer({ minimum: 1, maximum: 10, description: "Number of results; defaults to 6" })),
			}),
			async execute(_toolCallId, params) {
				const result = await callWebBridge({ action: "search", query: params.query, limit: params.limit || 6 });
				const rows = Array.isArray(result.results) ? result.results : [];
				const text = rows.length ? rows.map((source: any) =>
					`[网页${source.index}] ${source.title}\nURL: ${source.url}\n摘要: ${source.snippet}`,
				).join("\n\n") : `没有找到与“${params.query}”匹配的网页。`;
				return { content: [{ type: "text", text }], details: { query: params.query, sources: rows } };
			},
		});

		pi.registerTool({
			name: "read_web_page",
			label: "Read a web page",
			description: "Read the bounded text of a search result by its [网页N] source index. Prefer authoritative and recent sources.",
			parameters: Type.Object({
				source: Type.Integer({ minimum: 1, maximum: 40, description: "Source index returned by web_search" }),
			}),
			async execute(_toolCallId, params) {
				const result = await callWebBridge({ action: "read", source: params.source, limit: 9000 });
				const source = result.source || {};
				const text = `[网页${source.index}] ${source.title}\nURL: ${source.url}\n正文摘录:\n${String(result.content || "")}`;
				return { content: [{ type: "text", text }], details: { source } };
			},
		});
	}
}
