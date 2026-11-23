// Dependency-free regression test for the shipped page's playback controls.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const source = fs.readFileSync(path.join(__dirname, '../../tools/fastllm_pytools/nomas_demo.html'), 'utf8').match(/<script>([\s\S]*?)<\/script>/)[1];
let now = 1000.5, nextTimer = 0;
const timers = new Map(), elements = new Map(), drawn = [], listeners = {};
function element(id) {
    if (!elements.has(id)) elements.set(id, {
        value: id === 'speed' ? '1' : '0', textContent: '', hidden: true,
        setAttribute(name, value) { this[name] = value; },
        setPointerCapture() {}, focus() {},
        classList: {toggle() {}},
        getContext() { return {drawImage(image) { assert.ok(image, 'frame must exist'); drawn.push(image.src); }}; }
    });
    return elements.get(id);
}
const context = vm.createContext({
    document: {getElementById: element, addEventListener(name,fn) {listeners[name]=fn;}},
    window: {addEventListener() {}},
    performance: {now: () => now},
    setTimeout(callback) { timers.set(++nextTimer, callback); return nextTimer; },
    clearTimeout(id) { timers.delete(id); },
    Image: class {set src(value) { this.url = value; this.onload(); } get src() { return this.url; }},
    fetch: async () => ({ok:true,json: async () => ({precision:'fp32',prompt_enhancer:{enabled:false}})}),
    location: {search:''}, history: {replaceState() {}}, URLSearchParams,
});
vm.runInContext(source, context);
assert.equal(element('generate').disabled, true); // Wait for the initial capability check.
function tickAt(time) {
    now = time;
    const [id, callback] = timers.entries().next().value;
    timers.delete(id);
    callback();
}
(async () => {
    const job = {width:128,height:128,fps:24,output:{frames:Array.from({length:16},(_,i)=>`frame-${i}`)}};
    context.job = job;
    await vm.runInContext('showResult(job)', context);
    assert.equal(drawn.at(-1),'frame-0');
    assert.equal(timers.size,1);
    tickAt(1000); // A clock edge must never produce the old frame -1 failure.
    assert.equal(drawn.at(-1),'frame-0');
    tickAt(1250.5);
    assert.equal(drawn.at(-1),'frame-6');
    tickAt(2000.5);
    assert.equal(drawn.at(-1),'frame-8'); // loop beyond the 16-frame clip
    element('play').onclick();
    assert.equal(timers.size,0);
    assert.equal(element('play').textContent,'播放');
    now=5000;
    element('play').onclick();
    tickAt(5125);
    assert.equal(drawn.at(-1),'frame-11'); // resume from paused position
    element('seek').value='15';
    element('seek').oninput();
    assert.equal(drawn.at(-1),'frame-15');
    assert.equal(timers.size,0);
    element('speed').value='0.5';
    element('speed').onchange();
    assert.equal(timers.size,0); // speed changes do not unpause
    element('play').onclick();
    tickAt(5375);
    assert.equal(drawn.at(-1),'frame-2'); // half speed, wrap
    assert.match(element('frame-info').textContent,/第 3 \/ 16 帧/);
    assert.equal(element('player').hidden,false);
    // A world action plays once and holds its final viewpoint; it never loops
    // the camera back to an old position while waiting for the next operation.
    context.job={...job,world_id:'world_test',segment_index:1,action:'stay'};
    await vm.runInContext('showResult(job)',context);
    tickAt(8000);
    assert.equal(drawn.at(-1),'frame-15');
    assert.equal(timers.size,0);
    element('play').onclick();
    assert.equal(drawn.at(-1),'frame-0');
    tickAt(10000);
    assert.equal(timers.size,0);
    const calls=[];
    context.fetch=async(url,options)=>{
        calls.push([url,options]);
        return {ok:true,json:async()=>url.endsWith('/actions')?
            {...job,id:'video_next',world_id:'world_test',segment_index:2,action:'forward',status:'completed'}:
            {id:'world_test',status:'ready',segments:2,generated_frames:40}};
    };
    element('mode').value='world';
    vm.runInContext("world={id:'world_test',status:'ready',segments:1,generated_frames:20};controls();",context);
    assert.equal(element('forward').disabled,false);
    const moving=element('forward').onclick();
    assert.equal(element('forward').disabled,true);
    assert.equal(element('generate').disabled,true);
    await moving;
    assert.equal(calls[0][0],'./v1/worlds/world_test/actions');
    assert.equal(JSON.parse(calls[0][1].body).action,'forward');
    assert.equal(element('forward').disabled,false);
    assert.equal(element('session-info').textContent,'当前场景');
    context.enhanced = {enhanced_prompt: JSON.stringify({subject_list:[{display_name:'baby skier',subject_attribute:'One baby on two skis.'}],segments:[{visual_description_long:'A snowy ski resort.',contained_actions:[]}]})};
    vm.runInContext('showEnhancement(enhanced)', context);
    assert.equal(element('expanded').hidden, false);
    assert.match(element('expanded-text').textContent, /One baby on two skis/);
    vm.runInContext('showEnhancement({})', context);
    assert.equal(element('expanded').hidden, true);
    let prevented=false;
    listeners.keydown({repeat:false,key:'w',target:{tagName:'TEXTAREA'},preventDefault(){prevented=true;}});
    listeners.keydown({repeat:true,key:'w',target:{tagName:'CANVAS'},preventDefault(){prevented=true;}});
    assert.equal(prevented,false);
    assert.equal(calls.length,2); // action + refreshed persistent world
    // Continuous input stays responsive during inference without submitting
    // another job. Only the latest direction is used by the next segment.
    vm.runInContext("continuous=true;busy=true;controls();",context);
    assert.equal(element('forward').disabled,false);
    listeners.keydown({repeat:false,key:'w',target:{tagName:'CANVAS'},preventDefault(){}});
    assert.equal(vm.runInContext('desiredAction',context),'forward');
    listeners.keydown({repeat:false,key:'ArrowLeft',target:{tagName:'CANVAS'},preventDefault(){}});
    assert.equal(vm.runInContext('desiredAction',context),'look_left');
    listeners.keyup({key:'ArrowLeft'});
    assert.equal(vm.runInContext('desiredAction',context),'forward'); // W is still held.
    assert.equal(vm.runInContext('takeAction()',context),'look_left'); // A short turn is latched once.
    assert.equal(vm.runInContext('takeAction()',context),'forward');
    listeners.keyup({key:'w'});
    assert.equal(vm.runInContext('takeAction()',context),'stay');
    // A complete key tap between segment boundaries must not disappear.
    listeners.keydown({repeat:false,key:'d',target:{tagName:'CANVAS'},preventDefault(){}});
    listeners.keyup({key:'d'});
    assert.equal(vm.runInContext('takeAction()',context),'right');
    assert.equal(vm.runInContext('takeAction()',context),'stay');
    // Mouse/touch holds repeat; releasing retains at most one pending action.
    element('left').onpointerdown({pointerId:7,preventDefault(){}});
    assert.equal(vm.runInContext('takeAction()',context),'left');
    assert.equal(vm.runInContext('takeAction()',context),'left');
    element('left').onpointerup({pointerId:7});
    element('left').onclick({detail:1}); // The synthetic click after release must not enqueue again.
    assert.equal(vm.runInContext('takeAction()',context),'stay');
    assert.equal(calls.length,2);
    await element('continuous').onclick();
    assert.equal(vm.runInContext('continuous',context),false);
    vm.runInContext('busy=false;controls();',context);
    // Editing an established scene uses PATCH + a continuation, never DELETE
    // or POST /worlds. Refresh must preserve the user's current draft.
    calls.length=0;
    vm.runInContext("world={id:'world_test',status:'ready',segments:2,generated_frames:40,text_frames:120,prompt:'Old room',prompt_revision:1,applied_prompt_revision:1};",context);
    element('prompt').value='Old room';
    element('edit-frames').value='40';
    element('instruction').value='The same room with blue walls';
    context.fetch=async(url,options={})=>{
        calls.push([url,options]);
        return {ok:true,json:async()=>options.method==='PATCH'?
            {id:'world_test',status:'ready',prompt:'The same room with blue walls',prompt_revision:2}:
            url.endsWith('/actions')?
            {...job,id:'video_edit',world_id:'world_test',segment_index:3,action:'stay',status:'completed'}:
            {id:'world_test',status:'ready',segments:3,generated_frames:60,prompt:'The same room with blue walls',prompt_revision:2,applied_prompt_revision:2}};
    };
    await element('apply-scene').onclick();
    assert.deepEqual(calls.map(([url,opts])=>[url,opts.method||'GET']),[
        ['./v1/worlds/world_test','PATCH'],['./v1/worlds/world_test/actions','POST'],['./v1/worlds/world_test','GET']]);
    assert.equal(JSON.parse(calls[0][1].body).prompt,'The same room with blue walls');
    assert.equal(JSON.parse(calls[0][1].body).text_frames,40);
    assert.equal(JSON.parse(calls[1][1].body).action,'follow_prompt');
    assert.equal(vm.runInContext('world.segments',context),3);
    assert.equal(element('instruction').value,'');
    assert.equal(element('prompt').value,'Old room');
    // During generation text remains editable; accepting a replacement must
    // not start a second job or clobber the current world status.
    calls.length=0;
    vm.runInContext('busy=true;continuous=true;controls();',context);
    element('instruction').value='It stops raining';
    element('instruction').oninput();
    assert.equal(element('prompt').disabled,true);
    assert.equal(element('instruction').disabled,false);
    assert.equal(element('apply-scene').disabled,false);
    await element('apply-scene').onclick();
    assert.equal(calls.length,1);
    assert.equal(calls[0][1].method,'PATCH');
    assert.equal(vm.runInContext('takeAction()',context),'follow_prompt');
    vm.runInContext('stopContinuous();busy=false;controls();',context);
    // A hidden or unchecked PE option must not submit a stale perspective.
    vm.runInContext('world=null;busy=false;configureEnhancer(false);',context);
    element('mode').value='video';element('size').value='128 × 128';
    element('frames').value='16';element('seed').value='42';
    element('prompt').value='A quiet room';element('perspective').value='first-person';
    element('enhance-prompt').checked=true;
    element('enhance-prompt').onchange();
    assert.equal(element('enhancer-hint').hidden,false);
    assert.equal(element('enhance-prompt').disabled,true);
    assert.equal(element('perspective-option').hidden,true);
    assert.equal(element('perspective').disabled,true);
    context.fetch=async(url,options)=>{
        calls.push([url,options]);
        return {ok:true,json:async()=>({...job,id:'video_options',status:'completed'})};
    };
    async function submittedOptions(){
        calls.length=0;
        await element('form').onsubmit({preventDefault(){}});
        assert.equal(calls.length,1);
        assert.equal(calls[0][0],'./v1/videos');
        return JSON.parse(calls[0][1].body);
    }
    let options=await submittedOptions();
    assert.equal(options.enhance_prompt,false);
    assert.equal(options.perspective,'auto');
    vm.runInContext('configureEnhancer(true)',context);
    assert.equal(element('perspective-option').hidden,false);
    assert.equal(element('perspective').disabled,false);
    element('enhance-prompt').checked=false;element('enhance-prompt').onchange();
    assert.equal(element('perspective').disabled,true);
    options=await submittedOptions();
    assert.equal(options.enhance_prompt,false);
    assert.equal(options.perspective,'auto');
    element('enhance-prompt').checked=true;element('enhance-prompt').onchange();
    assert.equal(element('perspective').disabled,false);
    options=await submittedOptions();
    assert.equal(options.enhance_prompt,true);
    assert.equal(options.perspective,'first-person');
    element('size').value='custom';
    element('custom-width').value='736';element('custom-height').value='416';
    element('size').onchange();
    assert.equal(element('custom-size').hidden,false);
    options=await submittedOptions();
    assert.deepEqual([options.width,options.height],[736,416]);
    for(const invalid of [[720,416],[1280,1280]]){
        element('custom-width').value=String(invalid[0]);element('custom-height').value=String(invalid[1]);
        calls.length=0;
        await element('form').onsubmit({preventDefault(){}});
        assert.equal(calls.length,0);
        assert.match(element('status').textContent,/宽高/);
    }
    element('size').value='1280 × 704';element('size').onchange();
    assert.equal(element('custom-size').hidden,true);
    options=await submittedOptions();
    assert.deepEqual([options.width,options.height],[1280,704]);
    element('mode').value='world';
    vm.runInContext("world={id:'world_restore',status:'ready',segments:1,width:128,height:128,seed:42,enhance_prompt:false,perspective:'third-person observer'};restoreWorldInputs();controls();",context);
    assert.equal(element('enhance-prompt').checked,false);
    assert.equal(element('perspective').disabled,true);
    vm.runInContext("world.segments=2;busy=true;active='video_next';controls();",context);
    const drag=(type,x,y)=>element('video')['onpointer'+type]({
        pointerId:2,button:0,isPrimary:true,clientX:x,clientY:y
    });
    drag('down',100,100);drag('up',106,105);
    assert.equal(vm.runInContext('queuedStep',context),null);
    drag('down',100,100);drag('move',200,105);
    assert.equal(element('gesture-hint').textContent,'松开右转');
    drag('up',200,105);
    assert.equal(vm.runInContext('queuedStep',context),'look_right');
    element('forward').onclick();
    assert.equal(vm.runInContext('queuedStep',context),'forward');
    vm.runInContext("queuedStep='follow_prompt'",context);
    element('right').onclick();
    assert.equal(vm.runInContext('queuedStep',context),'follow_prompt');
    let submitted=0;
    element('apply-scene').disabled=false;element('apply-scene').click=()=>submitted++;
    for(const extra of [{isComposing:true},{keyCode:229},{shiftKey:true},{repeat:true},{}]){
        element('instruction').onkeydown({key:'Enter',preventDefault(){},...extra});
    }
    assert.equal(submitted,1);
    element('leave-scene').onclick();
    assert.equal(vm.runInContext('queuedStep',context),null);
    assert.equal(vm.runInContext('exploring',context),false);
    context.Option=class { constructor(text,value){this.text=text;this.value=value;} };
    element('size').replaceChildren=(first)=>{element('size').value=first.value;};
    vm.runInContext('configureLingBot();controls()',context);
    assert.equal(element('enhance-option').hidden,true);
    assert.equal(element('vertical-controls').hidden,false);
    assert.deepEqual(Array.from(vm.runInContext('selectedResolution()',context)),[576,320]);
    const movie={id:'job_test',world_id:'world_test',status:'completed',result:{
        width:576,height:320,frame_count:12,fps:16,segment:1,format:'mp4',url:'/v1/worlds/files/job_test.mp4'}};
    context.movie=movie;
    const segment=vm.runInContext('lingJob(movie)',context);
    assert.equal(segment.num_frames,12);
    assert.equal(segment.segment_index,2);
    assert.equal(segment.output.url,movie.result.url);
    console.log('PASS: playback, world controls, scene edits, PE capability/toggle/restore and submitted options');
})().catch(error => {console.error(error);process.exitCode=1;});
