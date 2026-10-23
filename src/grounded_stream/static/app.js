'use strict';
const $ = id => document.getElementById(id);
const terminal = new Set(['completed','partial','failed','cancelled','timed_out']);
let active = null, lastJob = null, generation = 0, polling = null, available = [], connected = false, submitting = false;
const cards = new Map();
function headers(){return $('token').value ? {Authorization:'Bearer '+$('token').value} : {};}
async function api(path, options = {}){
  const response = await fetch(path, {...options, headers:{...headers(),...options.headers}, signal:AbortSignal.timeout(10000)});
  if(!response.ok){let detail;try{detail=(await response.json()).detail;}catch{detail=response.status===401?'Enter the configured access token.':response.statusText;}
    const error=new Error(detail || 'Request failed');error.status=response.status;throw error;}
  return response.json();
}
function message(text){$('message').textContent=text;}
function modeControls(){
  const timed=$('timed').checked;
  $('duration-options').hidden=!timed;
  $('duration').disabled=!timed || !!active || submitting;
  $('mode-summary').textContent=timed?'Timed':'Continuous';
  $('mode-hint').textContent=timed?'Stops automatically after the selected duration.':'Runs until stopped, even if this tab closes. Refresh to reconnect.';
}
function busyControls(){
  const busy=!!active || submitting;
  $('start').disabled=!connected || busy;
  $('cancel').hidden=!active;
  $('reconnect').disabled=submitting;
  $('timed').disabled=busy;
  $('targets').disabled=busy;
  document.querySelectorAll('input[name=source]').forEach(input=>input.disabled=busy);
  $('health').classList.toggle('busy',busy);
  modeControls();
}
async function connect(){
  try{
    const health=await api('/v1/health');available=health.sources;connected=health.status==='ready';
    $('health').textContent=connected?'Local · Connected':health.status;
    // Retain this tab's selections without storing credentials or camera images.
    const previous=[...document.querySelectorAll('input[name=source]:checked')].map(input=>input.value);
    let saved=[];
    try{saved=JSON.parse(sessionStorage.getItem('grounded.sources') || '[]');}catch{}
    const selected=previous.length?previous:Array.isArray(saved)&&saved.length?saved:[available[0]?.id];
    $('sources').replaceChildren();
    for(const source of available){
      const label=document.createElement('label'), check=document.createElement('input'), text=document.createElement('span'), small=document.createElement('small');
      check.type='checkbox';check.name='source';check.value=source.id;check.checked=selected.includes(source.id);
      text.textContent=source.name;small.textContent=source.kind==='file'?'Recorded video':source.kind==='webcam'?'USB camera':'Network camera';
      text.append(small);label.append(check,text);$('sources').append(label);
    }
    if(health.active_search){
      const job=await api('/v1/searches/'+health.active_search+'?latest=true');
      if(!terminal.has(job.status)){
        clearTimeout(polling);const turn=++generation;active=job.id;
        $('targets').value=job.request.targets.join('; ');
        $('timed').checked=job.request.duration_seconds!==null;
        if($('timed').checked){
          const value=String(job.request.duration_seconds);
          if(![...$('duration').options].some(option=>option.value===value)){
            const option=document.createElement('option');option.value=value;option.textContent=value+' seconds';$('duration').append(option);
          }
          $('duration').value=value;
        }
        document.querySelectorAll('input[name=source]').forEach(input=>input.checked=job.request.source_ids.includes(input.value));
        cards.clear();$('cards').replaceChildren();busyControls();
        await render(job,turn);poll(job.id,turn);return;
      }
    }
    busyControls();message(active?'Reconnected.':'Ready when you are.');
  }catch(error){connected=false;$('health').textContent='Disconnected';busyControls();message(error.message);}
}

function cardFor(id){
  if(cards.has(id))return cards.get(id);
  const section=document.createElement('article'), head=document.createElement('div'), title=document.createElement('h3'), state=document.createElement('span'), canvas=document.createElement('canvas'), list=document.createElement('p');
  section.className='card';head.className='card-head';list.className='detection-list';canvas.hidden=true;
  title.textContent=available.find(source=>source.id===id)?.name || id;canvas.setAttribute('role','img');
  head.append(title,state);section.append(head,canvas,list);$('cards').append(section);
  const card={section,state,canvas,list,frame:null};cards.set(id,card);return card;
}
async function preview(card, observation, turn){
  if(card.frame===observation.frame_id)return;
  let objectUrl;
  try{
    const response=await fetch(observation.evidence_url,{headers:headers(),signal:AbortSignal.timeout(10000)});
    if(!response.ok)return;
    objectUrl=URL.createObjectURL(await response.blob());
    const img=new Image();img.src=objectUrl;await img.decode();
    if(turn!==generation)return;
    const canvas=card.canvas;canvas.width=img.width;canvas.height=img.height;
    const ctx=canvas.getContext('2d');ctx.drawImage(img,0,0);
    const scale=Math.max(1,img.width/700);
    ctx.lineWidth=2*scale;ctx.font=`${13*scale}px system-ui`;
    for(const item of observation.detections){
      const [x1,y1,x2,y2]=item.box, x=x1*img.width,y=y1*img.height;
      ctx.strokeStyle='#79f5ff';ctx.strokeRect(x,y,(x2-x1)*img.width,(y2-y1)*img.height);
      const label=`${item.label} · ${Math.round(item.score*100)}%`;
      const textWidth=ctx.measureText(label).width, top=Math.max(0,y-22*scale);
      ctx.fillStyle='#07101eed';ctx.fillRect(x,top,Math.min(img.width-x,textWidth+12*scale),22*scale);
      ctx.fillStyle='#a7f7ff';ctx.fillText(label,x+6*scale,top+16*scale);
    }
    canvas.hidden=false;card.frame=observation.frame_id;
    const labels=observation.detections.map(item=>`${item.label} ${Math.round(item.score*100)}%`);
    canvas.setAttribute('aria-label',`Frame from ${observation.source_id}. ${labels.join(', ') || 'No matches'}.`);
    card.list.textContent=labels.join(' · ') || 'No matches in this frame.';
  }catch(error){if(turn===generation)card.list.textContent='Preview unavailable. Detection details remain in the JSON result.';}
  finally{if(objectUrl)URL.revokeObjectURL(objectUrl);}
}
async function render(job,turn){
  if(turn!==generation)return;
  lastJob=job;$('empty').hidden=true;$('result-title').textContent=job.request.targets.join(' · ');
  $('export').hidden=false;$('metrics').hidden=false;
  $('metrics').textContent=`${job.processed_frames} frames processed${job.metrics.inference_p50_ms!=null?' · Median inference '+Math.round(job.metrics.inference_p50_ms)+' ms':''}`;
  message(job.error || (job.status==='loading'?'Loading detector. The first run may download model weights.':(job.status==='running' && job.request.duration_seconds===null?'Searching continuously':job.status.replaceAll('_',' '))));
  const latest=new Map(job.observations.map(item=>[item.source_id,item]));
  await Promise.all(Object.entries(job.sources).map(async([id,state])=>{
    const card=cardFor(id);card.state.textContent=state.state+' · '+state.processed_frames+' frames';
    if(latest.has(id))await preview(card,latest.get(id),turn);
    else card.list.textContent=state.detail || 'Waiting for a frame.';
  }));
  if(turn!==generation)return;
  if(terminal.has(job.status)){
    active=null;busyControls();
  }
}
async function poll(id,turn){
  let delay=document.hidden?1500:150;
  try{const job=await api('/v1/searches/'+id+'?latest=true');if(turn===generation)$('health').textContent='Local · Connected';await render(job,turn);}
  catch(error){
    if(turn!==generation)return;
    if(error.status===404){
      active=null;generation++;busyControls();
      message('This search expired. Start a new search.');return;
    }
    delay=1500;$('health').textContent='Reconnecting';message(error.message+' Retrying…');
  }
  if(turn===generation && active)polling=setTimeout(()=>poll(id,turn),delay);
}
$('search-form').addEventListener('submit',async event=>{
  event.preventDefault();if(active || submitting)return;
  const ids=[...document.querySelectorAll('input[name=source]:checked')].map(input=>input.value);
  const targets=$('targets').value.split(';').map(value=>value.trim()).filter(Boolean);
  if(!ids.length || !targets.length){message('Choose at least one source and enter an object.');return;}
  const turn=++generation;submitting=true;busyControls();
  try{sessionStorage.setItem('grounded.sources',JSON.stringify(ids));sessionStorage.setItem('grounded.targets',$('targets').value);}catch{}
  try{
    const job=await api('/v1/searches',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({targets,source_ids:ids,duration_seconds:$('timed').checked?Number($('duration').value):null,timeout_seconds:180})});
    active=job.id;cards.clear();$('cards').replaceChildren();busyControls();
    await render(job,turn);poll(job.id,turn);
  }catch(error){message(error.message);}
  finally{submitting=false;busyControls();}
});
$('cancel').addEventListener('click',async()=>{
  if(!active)return;
  try{const job=await api('/v1/searches/'+active,{method:'DELETE'});clearTimeout(polling);generation++;await render(job,generation);}
  catch(error){message('Could not stop search: '+error.message);}
});
$('reconnect').addEventListener('click',connect);
$('timed').addEventListener('change',modeControls);
$('export').addEventListener('click',async()=>{
  if(!lastJob)return;
  const id=lastJob.id;
  $('export').disabled=true;
  try{
  const exported=await api('/v1/searches/'+id);
  const url=URL.createObjectURL(new Blob([JSON.stringify(exported,null,2)],{type:'application/json'}));
  const link=document.createElement('a');link.href=url;link.download='grounded-search-'+id+'.json';link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
  }catch(error){message('Could not export: '+error.message);}
  finally{$('export').disabled=false;}
});
try{const saved=sessionStorage.getItem('grounded.targets');if(saved)$('targets').value=saved;}catch{}
modeControls();
connect();
