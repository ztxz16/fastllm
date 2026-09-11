import test from 'node:test';
import assert from 'node:assert/strict';
import { apply } from '../../tools/fastllm_pytools/harness_recovery.mjs';

function fixture(config = {}) {
  const handlers = new Map(), appended = [], followups = [], cancellations = [];
  const session = { append(...args) { appended.push(args); } };
  const agent = { session, inbox: { nextTurn: [], nextStep: [], remove(id) {
    for (const queue of [this.nextStep, this.nextTurn]) {
      const index = queue.findIndex(message => message.id === id);
      if (index >= 0) { queue.splice(index, 1); return true; }
    }
    return false;
  } },
    cancel(cause, options) { cancellations.push({ cause, options }); signal.abort(cause); },
    followup(message) { followups.push(message); this.inbox.nextTurn.push(message); },
  };
  const ctx = { on(event, listener) { handlers.set(event, listener); },
    sessions: { get(id) { return id === 's' ? session : undefined; } }, logger: { warn() {} } };
  apply(ctx, config);
  const signal = new AbortController();
  async function stream(chunks, options = {}) {
    const result = [];
    for await (const chunk of handlers.get('llm/stream')(
      { provider: 'fastllm', sessionId: 's', signal: signal.signal, ...options },
      async function* () { yield* chunks; })) result.push(chunk);
    return result;
  }
  async function repair(code = 'FASTLLM_EMPTY_ANSWER', next = async () => undefined, options = {}) {
    return handlers.get('agent/request-error')(
      { agent, provider: 'fastllm', failure: { code }, signal: signal.signal, ...options }, next);
  }
  return { ctx, handlers, session, agent, appended, followups, cancellations, signal, stream, repair };
}
const thought = [
  { type: 'block-start', index: 0, blockType: 'reasoning' },
  { type: 'reasoning-delta', index: 0, text: 'thinking' },
  { type: 'block-end', index: 0, block: { type: 'reasoning', text: 'thinking' } },
  { type: 'finish', reason: { kind: 'stop' } },
];

test('cancellation before a finish prevents classification and recovery', async () => {
  const f = fixture(); f.signal.abort();
  assert.deepEqual(await f.stream(thought), thought);
  assert.equal(await f.repair(), undefined);
  assert.equal(f.appended.length, 0);
});

test('cancellation during downstream recovery does not queue another request', async () => {
  const f = fixture(); await f.stream(thought);
  let release;
  const pending = f.repair(undefined, () => new Promise(resolve => { release = resolve; }));
  f.signal.abort(); release(undefined);
  assert.equal(await pending, undefined);
  assert.equal(f.appended.length, 0);
});

test('a canceled text continuation cannot wake the agent', async () => {
  const f = fixture();
  await f.stream([{ type: 'text-delta', text: 'partial answer' },
    { type: 'finish', reason: { kind: 'max-tokens' } }]);
  f.signal.abort();
  f.handlers.get('agent/turn-stopping')({ agent: f.agent, signal: f.signal.signal });
  assert.equal(f.followups.length, 0);
});

test('human queued work has precedence over an automatic continuation', async () => {
  const f = fixture();
  await f.stream([{ type: 'text-delta', text: 'partial answer' },
    { type: 'finish', reason: { kind: 'max-tokens' } }]);
  f.agent.inbox.nextTurn.push({ source: { kind: 'user' } });
  f.handlers.get('agent/turn-stopping')({ agent: f.agent, signal: f.signal.signal });
  assert.equal(f.followups.length, 0);
});

test('pending input hands off at a native turn boundary without consuming or reordering it', async () => {
  const f = fixture();
  const steering = { id: 'steering', source: { kind: 'user' } };
  const first = { id: 'first', source: { kind: 'user' } };
  const last = { id: 'last', source: { kind: 'user' } };
  f.agent.inbox.nextStep.push(steering);
  f.agent.inbox.nextTurn.push(first, last);
  let delegated = false;
  assert.equal(await f.repair(undefined, async () => { delegated = true; }), undefined);
  assert.equal(delegated, false);
  assert.deepEqual(f.agent.inbox.nextStep, [steering]);
  assert.deepEqual(f.agent.inbox.nextTurn, [first, last]);
  assert.equal(f.appended.length, 0);
  assert.equal(f.cancellations.length, 1);
  assert.deepEqual(f.cancellations[0].options, { keepInbox: true });
  assert.equal(f.signal.signal.reason.kind, 'hook');
  assert.equal(f.followups.length, 1);
  assert.equal(f.followups[0].source.kind, 'plugin');
});

test('queued input also takes precedence over an exhausted recovery budget', async () => {
  const f = fixture({ maxRecoveries: 0 });
  f.agent.inbox.nextStep.push({ id: 'user', source: { kind: 'user' } });
  assert.equal(await f.repair('FASTLLM_RECOVERY_EXHAUSTED'), undefined);
  assert.equal(f.cancellations.length, 1);
  assert.equal(f.followups.length, 1);
  assert.equal(f.appended.length, 0);
});

test('input arriving during downstream recovery supersedes its retry', async () => {
  const f = fixture();
  let release;
  const pending = f.repair(undefined, () => new Promise(resolve => { release = resolve; }));
  f.agent.inbox.nextStep.push({ id: 'user', source: { kind: 'user' } });
  release({ kind: 'retry' });
  assert.equal(await pending, undefined);
  assert.equal(f.cancellations.length, 1);
  assert.equal(f.appended.length, 0);
});

test('user cancellation during downstream recovery cannot wake queued work', async () => {
  const f = fixture();
  let release;
  const pending = f.repair(undefined, () => new Promise(resolve => { release = resolve; }));
  f.agent.inbox.nextStep.push({ id: 'user', source: { kind: 'user' } });
  f.signal.abort({ kind: 'user' });
  release(undefined);
  assert.equal(await pending, undefined);
  assert.equal(f.cancellations.length, 0);
  assert.equal(f.followups.length, 0);
});

test('a cancellation that wins the handoff cannot be overwritten or woken', async () => {
  const f = fixture();
  f.agent.inbox.nextStep.push({ id: 'user', source: { kind: 'user' } });
  f.agent.cancel = () => f.signal.abort({ kind: 'user' });
  assert.equal(await f.repair(), undefined);
  assert.equal(f.signal.signal.reason.kind, 'user');
  assert.equal(f.followups.length, 0);
});

test('foreign providers and ordinary transport failures do not hand off via this plugin', async () => {
  const f = fixture();
  f.agent.inbox.nextStep.push({ id: 'user', source: { kind: 'user' } });
  const retry = async () => ({ kind: 'retry' });
  assert.deepEqual(await f.repair(undefined, retry, { provider: 'another-provider' }), { kind: 'retry' });
  assert.deepEqual(await f.repair('TRANSPORT', retry), { kind: 'retry' });
  assert.equal(f.cancellations.length, 0);
  assert.equal(f.followups.length, 0);
});

test('foreign providers and compaction keep their own completion semantics', async () => {
  const f = fixture();
  assert.deepEqual(await f.stream(thought, { provider: 'another-provider' }), thought);
  assert.deepEqual(await f.stream(thought, { purpose: 'compaction' }), thought);
  assert.deepEqual(await f.stream(thought, { purpose: 'session-title' }), thought);
});

test('a final text block without deltas counts as an answer', async () => {
  const f = fixture();
  const chunks = [...thought.slice(0, -1),
    { type: 'block-end', index: 1, block: { type: 'text', text: '42' } }, thought.at(-1)];
  assert.deepEqual(await f.stream(chunks), chunks);
});

test('recovery is durable plugin input and its own reminder cannot reset the cap', async () => {
  const f = fixture({ maxRecoveries: 1 });
  assert.equal((await f.stream(thought)).at(-1).reason.failure.code, 'FASTLLM_EMPTY_ANSWER');
  assert.deepEqual(await f.repair(), { kind: 'retry' });
  const [event, message, intent] = f.appended[0];
  assert.equal(event, 'user/message');
  assert.equal(message.source.kind, 'plugin');
  assert.equal(message.source.plugin, 'fastllm-harness-recovery');
  assert.deepEqual(intent, { surfaceOp: 'append' });
  assert.ok(Object.isFrozen(message));
  f.handlers.get('session/event')(f.session, { type: event, data: message });
  assert.equal((await f.stream(thought)).at(-1).reason.failure.code, 'FASTLLM_RECOVERY_EXHAUSTED');
  f.handlers.get('session/event')(f.session, { type: event, data: { source: { kind: 'user' } } });
  assert.equal((await f.stream(thought)).at(-1).reason.failure.code, 'FASTLLM_EMPTY_ANSWER');
});

test('session recovery counts are independent', async () => {
  const f = fixture({ maxRecoveries: 1 });
  await f.stream(thought); await f.repair();
  const second = { append() {} };
  f.ctx.sessions.get = id => id === 's2' ? second : f.session;
  assert.equal((await f.stream(thought, { sessionId: 's2' })).at(-1).reason.failure.code, 'FASTLLM_EMPTY_ANSWER');
  assert.equal((await f.stream(thought)).at(-1).reason.failure.code, 'FASTLLM_RECOVERY_EXHAUSTED');
});

test('invalid config fails before hooks install and zero disables automatic recovery', async () => {
  for (const config of [{ maxRecoveries: -1 }, { maxRecoveries: 1.5 }, { maxRecoveries: 11 },
    { provider: '' }, { typo: true }]) assert.throws(() => fixture(config));
  const f = fixture({ maxRecoveries: 0 });
  assert.equal((await f.stream(thought)).at(-1).reason.failure.code, 'FASTLLM_RECOVERY_EXHAUSTED');
  assert.equal(await f.repair(), undefined);
  assert.equal(f.appended.length, 0);
});


test('whitespace-only text cannot be reported as a completed answer', async () => {
  const f = fixture();
  const chunks = [{ type: 'text-delta', text: '  \n' }, { type: 'finish', reason: { kind: 'stop' } }];
  assert.equal((await f.stream(chunks)).at(-1).reason.failure.code, 'FASTLLM_EMPTY_ANSWER');
});
