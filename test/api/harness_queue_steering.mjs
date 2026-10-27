/** Test-only user input during an actual Harness request; no GPU required. */
import { randomUUID } from 'node:crypto';
import { writeFileSync } from 'node:fs';

export const name = 'test-queue-steering';
export const inject = ['agents', 'sessions'];

export function apply(ctx, { queueAt = 1, target = 'next-step', auditPath }) {
  let attempt = 0, queued = false;
  const audit = { accepted: [], claimed: [], preStep: [], discarded: [], pending: [] };
  const marker = message => message.content?.find(block => block.type === 'text')?.text.match(/USER_STEER_(FIRST|LAST)/)?.[0];
  const record = (key, message) => { const id = marker(message); if (id) audit[key].push(id); };
  const save = () => writeFileSync(auditPath, JSON.stringify(audit, null, 2));
  ctx.on('agent/assistant-stream', ({ agent, frame }) => {
    if (frame.type === 'start') attempt++;
    if (queued || attempt !== queueAt || frame.type !== 'chunk'
        || !['reasoning-delta', 'text-delta', 'tool-call-delta'].includes(frame.chunk.type)) return;
    queued = true;
    for (const suffix of ['FIRST', 'LAST']) {
      agent.send(Object.freeze({ id: randomUUID(), role: 'user',
        content: Object.freeze([Object.freeze({ type: 'text',
          text: `USER_STEER_${suffix}: Change the task. Reply STEERING_${suffix}_RECEIVED.` })]),
        source: Object.freeze({ kind: 'user' }),
      }), target, true);
    }
  });
  ctx.on('session/event', (_session, event) => {
    if (event.type === 'user/message') record('accepted', event.data);
    save();
  });
  ctx.on('agent/inbox/claimed', ({ message }) => { record('claimed', message); save(); });
  ctx.on('agent/inbox/discarded', ({ message }) => { record('discarded', message); save(); });
  ctx.on('agent/pre-step', async ({ messages }, next) => {
    for (const message of messages) record('preStep', message);
    save();
    return next();
  });
  ctx.on('agent/status', ({ agent, status }) => {
    if (status !== 'idle') return;
    audit.pending = [...agent.inbox.nextStep, ...agent.inbox.nextTurn].map(marker).filter(Boolean);
    save();
  });
}
