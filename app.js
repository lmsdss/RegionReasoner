'use strict';
const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
const turns = [
  {query:'Find the player wearing jersey number 18.', target:[205,22,511,412], name:'batter', focus:'No prior reference in this turn. Locate the player from the scene.', think:'The foreground <mark>batter</mark> has <mark>18</mark> on his jersey. Ground the answer in that visible player.'},
  {query:'Find the standing player behind <span class="reference-text">R1</span>, on the left.', target:[101,7,221,253], name:'player behind R1', focus:'<span class="bbox-citation">R1 = [205, 22, 511, 412]</span> is the <mark>batter</mark> wearing jersey <mark>18</mark>.', think:'Using reference <span class="bbox-citation">[205, 22, 511, 412]</span>, look <mark>behind</mark> the <mark>batter</mark> and to his <mark>left</mark>. Localize the standing <mark>player</mark> in the dugout.'},
  {query:'Find another player next to <span class="reference-text">R2</span>.', target:[22,8,112,252], name:'neighbor of R2', focus:'<span class="bbox-citation">R2 = [101, 7, 221, 253]</span> is a standing <mark>player</mark> behind the <mark>batter</mark>.', think:'Using reference <span class="bbox-citation">[101, 7, 221, 253]</span>, look <mark>next to</mark> this <mark>player</mark>. The adjacent player to the <mark>left</mark> is the new target.'}
];
const statuses = ['Reading the whole scene','Inspecting the reference region','Citing evidence & spatial relations','Saving the localized region'];
const state = {round:0,phase:0,playing:!reducedMotion.matches,elapsed:0};
const phaseDuration = 2600;
let lastTick = performance.now();
let demoVisible = true;
function positionBox(el, box){
  el.style.left = `${box[0]/640*100}%`;
  el.style.top = `${box[1]/424*100}%`;
  el.style.width = `${(box[2]-box[0])/640*100}%`;
  el.style.height = `${(box[3]-box[1])/424*100}%`;
}
function renderDemo(){
  const t=turns[state.round];
  $('#round-caption').textContent=`ROUND 0${state.round+1} / 03`;
  $('#query').innerHTML=t.query;
  $('#focus-text').innerHTML=t.focus;
  $('#think-text').innerHTML=t.think;
  $('#answer-text').textContent=`{"bbox_2d": [${t.target.join(', ')}]}`;
  $('#trace-step').textContent=`0${state.phase+1} / 04`;
  $('#image-stage').dataset.phase=state.phase;
  $('#image-status').textContent=state.round===0&&state.phase===1?'No previous reference required':statuses[state.phase];
  $$('.trace-block').forEach((el,i)=>{el.classList.toggle('active',i===state.phase);el.classList.toggle('complete',i<state.phase);});
  $$('[data-round]').forEach(el=>el.setAttribute('aria-pressed',Number(el.dataset.round)===state.round));
  $$('.phase-tabs [data-phase]').forEach(el=>el.setAttribute('aria-pressed',Number(el.dataset.phase)===state.phase));
  const target=$('#target-box'),ref=$('#reference-box');
  positionBox(target,t.target);
  $('#target-label').textContent=`R${state.round+1} · localized`;
  target.classList.toggle('visible',state.phase===3);
  ref.classList.toggle('visible',state.round>0&&state.phase>=1);
  if(state.round>0){
    const b=turns[state.round-1].target;
    positionBox(ref,b);
    $('#reference-label').textContent=`R${state.round} · reference`;
    const x1=(b[0]+b[2])/2,y1=(b[1]+b[3])/2;
    const x2=(t.target[0]+t.target[2])/2,y2=(t.target[1]+t.target[3])/2;
    $('#relation-path').setAttribute('d',`M ${x1} ${y1} Q ${(x1+x2)/2} ${Math.max(10,Math.min(y1,y2)-55)} ${x2} ${y2}`);
  }
  $('#relation-arrow').classList.toggle('visible',state.round>0&&state.phase===2);
  const count=state.round+(state.phase===3?1:0);
  $('#memory-items').innerHTML=count?turns.slice(0,count).map((turn,i)=>`<span class="memory-chip ${i===count-1?'latest':''}"><b>R${i+1}</b>${turn.name}</span>`).join(''):'<span class="memory-empty">The next turn builds on this one.</span>';
  const cited=state.round>0&&state.phase>=2;
  $('#consistency-title').textContent=cited?'Reference cited. Semantics connected.':'Keep the scene in the loop';
  $('#consistency-text').textContent=cited?'The same entities link scene, focus, and reasoning.':'Shared objects connect visual evidence and reasoning.';
  updatePlayButton();
  updateProgress();
}
function updatePlayButton(){
  $('#idea').dataset.playing=String(state.playing);
  $('#play-button').setAttribute('aria-label',state.playing?'Pause animation':'Play animation');
  $('#play-button use').setAttribute('href',state.playing?'#i-pause':'#i-play');
}
function updateProgress(){
  const fraction=(state.round*4+state.phase+state.elapsed/phaseDuration)/12;
  $('#demo-progress-fill').style.width=`${fraction*100}%`;
}
function advance(){
  state.elapsed=0;
  if(state.phase<3)state.phase++;else{state.phase=0;state.round=(state.round+1)%turns.length;}
  renderDemo();
}
$('#play-button').addEventListener('click',()=>{state.playing=!state.playing;lastTick=performance.now();updatePlayButton();});
$('#replay-button').addEventListener('click',()=>{Object.assign(state,{round:0,phase:0,elapsed:0,playing:!reducedMotion.matches});lastTick=performance.now();renderDemo();});
$('#next-button').addEventListener('click',()=>{state.playing=false;advance();});
$$('[data-round]').forEach(el=>el.addEventListener('click',()=>{state.round=Number(el.dataset.round);state.phase=0;state.elapsed=0;state.playing=false;renderDemo();}));
$$('.phase-tabs [data-phase]').forEach(el=>el.addEventListener('click',()=>{state.phase=Number(el.dataset.phase);state.elapsed=0;state.playing=false;renderDemo();}));
new IntersectionObserver(entries=>{demoVisible=entries[0].isIntersecting;lastTick=performance.now();},{threshold:.1}).observe($('#idea'));
reducedMotion.addEventListener('change',()=>{if(reducedMotion.matches){state.playing=false;updatePlayButton();}});
setInterval(()=>{
  const now=performance.now(),delta=Math.min(now-lastTick,150);lastTick=now;
  if(!state.playing||!demoVisible||document.hidden)return;
  state.elapsed+=delta;
  if(state.elapsed>=phaseDuration)advance();else updateProgress();
},100);
renderDemo();

const results={
  detection:{
    plus:[{name:'RegionReasoner',v:[89.3,83.2,81.6,69.6,61.9,69.1,64.7],avg:80.7},{name:'VisionReasoner',v:[88.3,74.7,75.8,64.2,56.3,57.3,47],avg:74.8},{name:'Seg-Zero',v:[90.5,71.2,73.6,59.6,48.8,58.2,48.2],avg:73.1},{name:'Qwen2.5-VL',v:[65.5,49,48.1,36.5,30,38.2,25.9],avg:49.9},{name:'Qwen2-VL',v:[6.2,8.5,6.5,5.4,7.5,3.6,3.5],avg:6.7}],
    g:[{name:'RegionReasoner',v:[87.1,73.7,71.8,68.6,75,78.4,75],avg:78.2},{name:'VisionReasoner',v:[85,65.8,66.8,69.3,68.3,75.2,72.5],avg:73.6},{name:'Seg-Zero',v:[85.3,61.8,61.6,64.8,70,69.6,68.8],avg:71.1},{name:'Qwen2.5-VL',v:[63.9,43.7,39,37.9,42.2,43.2,33.8],avg:49.7},{name:'Qwen2-VL',v:[7.8,6.2,3.5,3.5,5.6,4,5],avg:6.1}]
  },
  segmentation:{
    plus:[{name:'RegionReasoner',v:[76.4,73.1,72,58.8,51.3,59.4,54.6],avg:69.6},{name:'VisionReasoner',v:[75.6,65,65.9,54.9,46.6,48.9,40.8],avg:64.3},{name:'Seg-Zero',v:[78.6,62.8,64,51.6,42.4,50.8,46.7],avg:64},{name:'Qwen2.5-VL',v:[56.5,43.3,41.4,34.5,23.4,33.6,24.9],avg:43.6},{name:'SegLLM',v:[71.1,71.7,70.4,58.7,41.9,39.2,30.3],avg:60.7},{name:'Qwen2-VL',v:[12.2,9,6.5,5.3,6.3,6.3,9.5],avg:9.4}],
    g:[{name:'RegionReasoner',v:[73.9,62.9,60.7,58.9,64.4,66.8,63.3],avg:66.5},{name:'VisionReasoner',v:[69.5,52.7,55.4,56,57.8,64.1,57.6],avg:59.9},{name:'Seg-Zero',v:[72.3,52.3,53.5,55.4,59.4,59.5,58.3],avg:60.5},{name:'Qwen2.5-VL',v:[53.8,36.3,35.5,31.6,37.3,36.8,28.6],avg:42.1},{name:'SegLLM',v:[68.9,55.3,50.5,47.7,47.3,37.8,25.4],avg:56.7},{name:'Qwen2-VL',v:[8.5,11.6,8.8,10,7,6.4,4.4],avg:9.3}]
  }
};
const colors={'RegionReasoner':'#426b30','VisionReasoner':'#9274b9','Seg-Zero':'#7a9892','Qwen2.5-VL':'#c2a47a','SegLLM':'#8c9bb1','Qwen2-VL':'#b1b8a8'};
let task='detection',dataset='plus';
const chartCompact = window.matchMedia('(max-width: 760px)');
function renderResults(){
  const rows=results[task][dataset],metric=task==='detection'?'AP':'gIoU',datasetName=dataset==='plus'?'RefCOCO+':'RefCOCOg';
  const title=`${task==='detection'?'Detection':'Segmentation'} · ${datasetName} Multi-turn`;
  $('#chart-title').textContent=title;$('#chart-metric').textContent=`${metric} (%) ↑`;
  $('#aggregate-score').textContent=rows[0].avg.toFixed(1);$('#baseline-score').textContent=rows[1].avg.toFixed(1);$('#aggregate-gain').textContent='+'+(rows[0].avg-rows[1].avg).toFixed(1);
  $('#result-gain').textContent='+'+(rows[0].v[6]-rows[1].v[6]).toFixed(1);
  $('#result-gain-label').innerHTML=`${metric} at Round 7 vs. VisionReasoner<br>${datasetName} Multi-turn`;
  $$('[data-task]').forEach(el=>el.setAttribute('aria-pressed',el.dataset.task===task));
  $$('[data-dataset]').forEach(el=>el.setAttribute('aria-pressed',el.dataset.dataset===dataset));
  const left=chartCompact.matches?40:62, right=chartCompact.matches?505:938;
  $('#results-chart').setAttribute('viewBox',`0 0 ${chartCompact.matches?540:1000} 355`);
  const x=i=>left+i*(right-left)/6, y=v=>305-v*2.75;
  let content=`<title id="chart-accessible-title">${title}: performance over seven dialogue rounds</title><desc id="chart-accessible-desc">RegionReasoner scores ${rows[0].v.join(', ')} ${metric} across rounds 1 through 7. Full model values are available in the table below.</desc>`;
  for(let v=0;v<=100;v+=20)content+=`<line x1="${left}" y1="${y(v)}" x2="${right}" y2="${y(v)}" class="chart-grid"/><text x="${left-15}" y="${y(v)+4}" text-anchor="end" class="chart-axis">${v}</text>`;
  for(let i=0;i<7;i++)content+=`<text x="${x(i)}" y="332" text-anchor="middle" class="chart-axis">${chartCompact.matches?'R':'Round '}${i+1}</text>`;
  const chartRows=rows.filter(row=>row.name!=='Qwen2-VL');
  [...chartRows].reverse().forEach(row=>{
    content+=`<polyline points="${row.v.map((v,i)=>`${x(i)},${y(v)}`).join(' ')}" stroke="${colors[row.name]}" class="chart-line ${row.name==='RegionReasoner'?'ours':''}"/>`;
    row.v.forEach((v,i)=>{content+=`<circle cx="${x(i)}" cy="${y(v)}" r="${row.name==='RegionReasoner'?5:3.5}" fill="${colors[row.name]}" class="chart-point" tabindex="0" role="img" aria-label="${row.name}, round ${i+1}: ${v.toFixed(1)} ${metric}" data-model="${row.name}" data-r="${i}" data-value="${v}"/>`;});
  });
  $('#results-chart').innerHTML=content;
  $('#chart-legend').innerHTML=chartRows.map(row=>`<span><i style="background:${colors[row.name]}"></i>${row.name}</span>`).join('');
  $('#results-table tbody').innerHTML=rows.map((row,i)=>`<tr class="${i===0?'ours':''}"><th scope="row">${row.name}-7B</th>${[...row.v,row.avg].map(v=>`<td>${v.toFixed(1)}</td>`).join('')}</tr>`).join('');
  $('#results-table caption').textContent=`${title} · ${metric} (%) · Table ${task==='detection'?'1':'2'}`;
  $$('.chart-point').forEach(el=>{
    const show=()=>{const tip=$('#chart-tooltip');tip.textContent=`${el.dataset.model} · Round ${Number(el.dataset.r)+1} · ${Number(el.dataset.value).toFixed(1)} ${metric}`;tip.hidden=false;};
    const hide=()=>{$('#chart-tooltip').hidden=true;};
    el.addEventListener('mouseenter',show);el.addEventListener('mouseleave',hide);el.addEventListener('focus',show);el.addEventListener('blur',hide);el.addEventListener('click',show);
  });
  $('#chart-tooltip').hidden=true;
}
$$('[data-task]').forEach(el=>el.addEventListener('click',()=>{task=el.dataset.task;renderResults();}));
$$('[data-dataset]').forEach(el=>el.addEventListener('click',()=>{dataset=el.dataset.dataset;renderResults();}));
renderResults();
chartCompact.addEventListener('change',renderResults);

$('#copy-citation').addEventListener('click',async()=>{
  const text=$('#bibtex').textContent;
  try{
    if(navigator.clipboard&&window.isSecureContext)await navigator.clipboard.writeText(text);
    else{
      const field=document.createElement('textarea');field.value=text;field.style.position='fixed';field.style.opacity='0';document.body.append(field);field.select();const copied=document.execCommand('copy');field.remove();if(!copied)throw new Error('Copy unavailable');
    }
    $('#copy-citation span').textContent='Copied';$('#copy-status').textContent='BibTeX copied to clipboard.';
    setTimeout(()=>{$('#copy-citation span').textContent='Copy BibTeX';},2500);
  }catch{$('#copy-status').textContent='Select the citation above to copy it.';}
});
const figureDialog=$('#figure-dialog');
function openFigure(){figureDialog.showModal();document.body.style.overflow='hidden';}
$('#figure-open').addEventListener('click',openFigure);$('#figure-preview').addEventListener('click',openFigure);
$('#figure-close').addEventListener('click',()=>figureDialog.close());
figureDialog.addEventListener('close',()=>{document.body.style.overflow='';});
figureDialog.addEventListener('click',e=>{if(e.target===figureDialog){const rect=figureDialog.getBoundingClientRect();if(e.clientX<rect.left||e.clientX>rect.right||e.clientY<rect.top||e.clientY>rect.bottom)figureDialog.close();}});
