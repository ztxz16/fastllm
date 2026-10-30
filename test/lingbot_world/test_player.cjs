// Run the shipped page's control/playback code without a browser or packages.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const source = fs.readFileSync(path.join(__dirname, '../../tools/lingbot_world/demo.html'), 'utf8')
  .match(/<script>([\s\S]*?)<\/script>/)[1];
const elements = new Map(), timers = new Map(), listeners = {}, calls = [], drawn = [], closed = [];
const buttons = ['forward','backward','left','right','up','down','look_left','look_right','look_up','look_down','stay']
  .map(action => ({dataset:{action}}));
let now = 1000.5, nextTimer = 0, nextImage = 0, reply;
function element(id) {
  if (!elements.has(id)) elements.set(id, {
    value: id === 'size' ? '576,320' : id === 'seed' ? '42' : '', style:{},
    classList:{toggle(){}}, getContext(){return {drawImage(image){assert.ok(image);drawn.push(image.id);}};}
  });
  return elements.get(id);
}
const document = {getElementById:element,querySelectorAll:()=>buttons,activeElement:{tagName:'BODY'},
  addEventListener(name, callback){listeners[name]=callback;}};
const context = vm.createContext({document,URL,location:{href:'http://localhost/'},
  history:{replaceState(){}},performance:{now:()=>now},setTimeout,
  requestAnimationFrame(callback){timers.set(++nextTimer,callback);return nextTimer;},
  cancelAnimationFrame(id){timers.delete(id);},
  createImageBitmap:async()=>({id:++nextImage,width:576,height:320,close(){closed.push(this.id);}}),
  fetch:async(url,options)=>{
    calls.push([url,options]);
    if (url.startsWith('/frame')) return {ok:true,blob:async()=>({})};
    return {ok:true,json:async()=>await reply(url,options)};
  }});
vm.runInContext(source,context);
function tick(time) {
  now=time; const [id,callback]=timers.entries().next().value;
  timers.delete(id);callback(time);
}
function completed(frames=3) {
  return {status:'completed',result:{world_id:'world_test',fps:16,
    frames:Array.from({length:frames},(_,i)=>'/frame'+i)}};
}
(async()=>{
  assert.ok(buttons.every(b=>b.disabled));
  await element('create').onclick();
  assert.match(element('statusText').textContent,/选择起始图片/);
  assert.equal(calls.length,0);
  vm.runInContext("png='data:image/png;base64,test'",context);
  reply=async()=>completed();
  await element('create').onclick();
  assert.equal(calls[0][0],'/v1/worlds');
  assert.equal(JSON.parse(calls[0][1].body).width,576);
  assert.ok(buttons.every(b=>!b.disabled));
  tick(1000); // rAF's timestamp can precede performance.now() at scheduling.
  assert.equal(drawn.at(-1),1);
  tick(1300);
  assert.equal(drawn.at(-1),3);
  assert.equal(timers.size,0); // Hold the final viewpoint.
  element('replay').onclick();tick(1300);assert.equal(drawn.at(-1),1);
  let release;
  reply=()=>new Promise(resolve=>{release=resolve;});
  const move=buttons[0].onclick();
  assert.ok(buttons.every(b=>b.disabled));
  assert.equal(JSON.parse(calls.at(-1)[1].body).action,'forward');
  await new Promise(setImmediate);
  release(completed(4));await move;
  assert.deepEqual(closed,[1,2,3]);
  assert.equal(element('create').disabled,false);
  reply=async()=>completed();
  element('editPrompt').value='A sunny garden';
  await element('edit').onclick();
  const request=calls.findLast(([url])=>url.endsWith('/actions'));
  assert.deepEqual(JSON.parse(request[1].body),{action:'stay',prompt:'A sunny garden'});
  const before=calls.length;
  document.activeElement.tagName='TEXTAREA';
  listeners.keydown({key:'w',preventDefault(){throw Error('typing must not move the camera');}});
  document.activeElement.tagName='BODY';
  listeners.keydown({key:'w',repeat:true,preventDefault(){throw Error('ignore key repeat');}});
  assert.equal(calls.length,before);
  reply=async()=>{throw Error('connection lost');};
  await buttons[1].onclick();
  assert.match(element('statusText').textContent,/connection lost/);
  assert.ok(buttons.every(b=>!b.disabled));
  reply=async()=>({status:'failed',world_deleted:true,world_id:'world_test',error:'disk full'});
  await buttons[1].onclick();
  assert.match(element('statusText').textContent,/重新创建场景/);
  assert.ok(buttons.every(b=>b.disabled));
  assert.equal(element('create').disabled,false);
  console.log('LingBot playback, controls, editing, focus, locking and failure recovery passed');
})().catch(error=>{console.error(error);process.exitCode=1;});
