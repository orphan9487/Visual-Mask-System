(async()=>{
const me=await requireSession();if(!me)return;
const select=document.getElementById('mask-select'),status=document.getElementById('mask-status');document.getElementById('who').textContent=me.username;
async function api(url,options={}){const r=await fetch(url,{...options,headers:{...authHeaders(),...options.headers}});const d=await r.json();if(!r.ok)throw Error(d.detail||'操作失敗');return d}
function go(id){location.href='/chat?room_id='+encodeURIComponent(id)}
function entry(name,hint,onclick){const b=document.createElement('button'),small=document.createElement('small');b.className='item';b.append(document.createTextNode(name));small.textContent=hint;b.append(small);b.onclick=onclick;return b}
try{const d=await api('/api/masks');select.replaceChildren();for(const m of d.masks.filter(m=>m.status==='ready')){const option=document.createElement('option');option.value=m.key;option.textContent=m.display_name;option.selected=m.is_active;select.append(option)}select.disabled=!select.options.length;status.textContent=select.options.length?'切換後會套用到下一則訊息':'請到面具管理新增並確認一張面具';
select.onchange=async()=>{const old=user()?.default_mask_id;select.disabled=true;try{await api('/api/masks/active',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({mask_id:select.value})});status.textContent='已切換，下一則訊息生效'}catch(e){status.textContent=e.message;if(old)select.value=old}finally{select.disabled=false}};
const [a,b]=await Promise.all([api('/api/chats'),api('/api/chat-candidates')]);const rooms=document.getElementById('rooms'),people=document.getElementById('people');if(!a.rooms.length)rooms.textContent='尚無聊天室';for(const x of a.rooms)rooms.append(entry(x.peer_name,'開啟既有聊天',()=>go(x.id)));for(const x of b.users)people.append(entry(x.username,'建立私人聊天室',async()=>{try{const d=await api('/api/chats',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({peer_id:x.id})});go(d.room.id)}catch(e){status.textContent=e.message}}));
}catch(e){status.textContent=e.message}
})();
