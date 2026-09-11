/** Bounded recovery for FastLLM requests through public Harness plugin hooks. */
import { randomUUID } from 'node:crypto';

export const name = 'fastllm-harness-recovery';
export const inject = ['agents', 'sessions', 'llm'];
const EMPTY = 'FASTLLM_EMPTY_ANSWER';
const TOOL = 'FASTLLM_INVALID_TOOL_CALL';
const LIMIT = 'FASTLLM_RECOVERY_EXHAUSTED';

// The public Message wire representation uses UUID identity and plugin provenance.
function reminder(text) {
  return Object.freeze({
    id: randomUUID(), role: 'user',
    content: Object.freeze([Object.freeze({ type: 'text', text })]),
    source: Object.freeze({ kind: 'plugin', plugin: name, form: 'notice',
      summary: 'FastLLM is recovering an incomplete model response.' }),
  });
}

/** Install recovery, scoped to one provider and bounded across automatic turns. */
export function apply(ctx, config = {}) {
  const provider = config.provider ?? 'fastllm';
  const maxRecoveries = config.maxRecoveries ?? 2;
  if (typeof provider !== 'string' || !provider.trim()) throw new Error(`${name}: provider must be nonempty`);
  if (!Number.isInteger(maxRecoveries) || maxRecoveries < 0 || maxRecoveries > 10) {
    throw new Error(`${name}: maxRecoveries must be an integer from 0 to 10`);
  }
  for (const key of Object.keys(config)) {
    if (!['provider', 'maxRecoveries'].includes(key)) throw new Error(`${name}: unknown config key ${key}`);
  }
  const states = new WeakMap();
  function state(session) {
    let value = states.get(session);
    if (!value) {
      value = { used: 0, continueText: false };
      states.set(session, value);
    }
    return value;
  }
  function exhausted(reason) {
    return { kind: 'error', failure: { code: LIMIT,
      message: `FastLLM recovery stopped after ${maxRecoveries} attempts: ${reason}.` } };
  }
  function warn(message) { ctx.logger.warn(`${name}: ${message}`); }

  function handoffPendingInput(agent, signal) {
    if (signal.aborted || (!agent.inbox.nextStep.length && !agent.inbox.nextTurn.length)) return false;
    state(agent.session).continueText = false;
    // Error retries never enter pre-step. End this failed turn while retaining
    // the inbox so the driver can claim and prepare its messages normally.
    const cause = { kind: 'hook', reason: `${name}: pending input takes precedence over recovery` };
    agent.cancel(cause, { keepInbox: true });
    if (signal.reason !== cause) return true;
    // Public send-after-cancel latches a wake for the next driver. Remove only
    // our wake marker: it is not model input, and user queues keep their order.
    // The latch runs only while pending work remains, including after a cancel.
    const wake = reminder('Resume pending input.');
    agent.followup(wake);
    agent.inbox.remove(wake.id);
    warn('yielding recovery to pending input');
    return true;
  }

  ctx.on('session/event', (session, event) => {
    // Automatic reminders must not replenish their own budget.
    if (event.type === 'user/message' && event.data.source?.kind === 'user') {
      states.set(session, { used: 0, continueText: false });
    }
    if (event.type === 'turn/end' && event.data.reason.kind !== 'max-tokens') {
      const value = states.get(session);
      if (value) value.continueText = false;
    }
  });

  ctx.on('llm/stream', (options, next) => {
    if (options.provider !== provider || options.purpose !== undefined || !options.sessionId) return next();
    const session = ctx.sessions.get(options.sessionId);
    if (!session) return next();
    const value = state(session);
    value.continueText = false;
    return (async function* () {
      let text = false, tools = false;
      for await (const chunk of next()) {
        if (chunk.type === 'text-delta' && chunk.text.trim()) text = true;
        if (chunk.type === 'block-start' && chunk.blockType === 'tool-call') tools = true;
        if (chunk.type === 'block-end') {
          if (chunk.block.type === 'text' && chunk.block.text.trim()) text = true;
          if (chunk.block.type === 'tool-call') tools = true;
        }
        if (chunk.type !== 'finish' || options.signal?.aborted) { yield chunk; continue; }
        let reason = chunk.reason;
        let recovery;
        if (reason.kind === 'max-tokens') {
          if (tools) recovery = { code: TOOL, message: 'Tool call output was truncated at the token limit.' };
          else if (!text) recovery = { code: EMPTY, message: 'The output budget was spent without an answer or tool call.' };
          else if (value.used < maxRecoveries) value.continueText = true;
          else reason = exhausted('the response still reached the output token limit');
        } else if (reason.kind === 'stop' && !text && !tools) {
          recovery = { code: EMPTY, message: 'The model stopped without a usable answer or tool call.' };
        } else if (reason.kind === 'error' && /^Invalid tool call:/i.test(reason.failure.message)) {
          recovery = { code: TOOL, message: 'The model returned an invalid or incomplete tool call.' };
        }
        if (recovery) {
          reason = value.used < maxRecoveries
            ? { kind: 'error', failure: recovery }
            : exhausted(recovery.message);
        }
        yield { ...chunk, reason };
      }
    })();
  });

  ctx.on('agent/request-error', async ({ agent, provider: route, failure, signal }, next) => {
    if (route !== provider || ![EMPTY, TOOL, LIMIT].includes(failure.code) || signal.aborted) return next();
    if (handoffPendingInput(agent, signal)) return;
    const downstream = await next();
    if (signal.aborted) return downstream;
    // Input can arrive while another recovery hook is awaiting its decision.
    if (handoffPendingInput(agent, signal)) return;
    if (downstream?.kind === 'retry' || failure.code === LIMIT) return downstream;
    const value = state(agent.session);
    if (value.used >= maxRecoveries) return downstream;
    const instruction = failure.code === TOOL
      ? 'Your last response contained a rejected tool call. No tool from that rejected response executed; preserve results of earlier successful calls. Recreate a complete valid call using the declared tool schema. Keep arguments small; split large writes into smaller operations. Do not assume that the rejected call changed any files.'
      : 'Your last response contained no usable answer or tool call. Keep reasoning brief. Continue the original task by making a valid tool call or giving the actual answer.';
    // Retry rebuilds its request from the durable surface, not the pending inbox.
    agent.session.append('user/message', reminder(instruction), { surfaceOp: 'append' });
    value.used++;
    warn(`retry ${value.used}/${maxRecoveries}: ${failure.code}`);
    return { kind: 'retry' };
  });

  ctx.on('agent/turn-stopping', ({ agent, signal }) => {
    const value = states.get(agent.session);
    if (!value?.continueText) return;
    value.continueText = false;
    if (signal.aborted || value.used >= maxRecoveries
        || agent.inbox.nextTurn.length || agent.inbox.nextStep.length) return;
    value.used++;
    warn(`continuation ${value.used}/${maxRecoveries}: max-tokens`);
    // A new turn retains the original max-tokens outcome and permits the eventual
    // completed turn to be reported correctly by Harness's Headless/Web clients.
    agent.followup(reminder('Your previous response reached the output token limit. Continue the original task from where it stopped. Avoid repeating completed work or restarting the answer. Keep reasoning brief and finish the remaining answer or tool work.'));
  });
}
