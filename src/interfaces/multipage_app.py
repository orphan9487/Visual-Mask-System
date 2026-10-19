"""Authenticated multi-page Visual Mask application."""
from __future__ import annotations
import asyncio, json
from collections import defaultdict, deque
from pathlib import Path
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from db.user_repo import get_user, register_user, verify_login
from db.appearance_repo import DEFAULTS, get_appearance, save_appearance
from db.chat_repo import get_or_create_direct_room, is_room_member, list_chat_candidates, list_messages, list_rooms, room_members, save_message, update_message_generation
from src.services import portrait_masks
from src.security.tokens import issue_token, verify_token
from src.services.pipeline_service import pipeline
from src.config import inference as inference_config

ROOT=Path(__file__).resolve().parents[2]; WEB=ROOT/'web'; OUT=ROOT/'output'/'current'
OUT.mkdir(parents=True,exist_ok=True)
class Credentials(BaseModel): account:str; password:str
class Registration(Credentials): username:str
class Peer(BaseModel): peer_id:int
class EmotionCorrection(BaseModel):
    correct_label: str
class MaskConfirmation(BaseModel):
    accepted: bool=False
class Appearance(BaseModel):
    gender_presentation:str = DEFAULTS['gender_presentation']
    age_group:str = DEFAULTS['age_group']
    hair_length:str = DEFAULTS['hair_length']
    hair_color:str = DEFAULTS['hair_color']
    hair_style:str = DEFAULTS['hair_style']
    glasses:bool = DEFAULTS['glasses']
    facial_hair:str = DEFAULTS['facial_hair']
    framing:str = DEFAULTS['framing']
def public(u): return {'id':u['id'],'username':u['username'],'face_profile_status':'ready' if portrait_masks.active_key(u['id']) else 'missing','default_mask_id':portrait_masks.active_key(u['id'])}
def login_result(u): return {'access_token':issue_token(user_id=u['id'],username=u['username']),'user':public(u)}
def current(request:Request):
    h=request.headers.get('Authorization',''); p=verify_token(h[7:]) if h.startswith('Bearer ') else None; u=get_user(p['user_id']) if p else None
    if not u: raise HTTPException(401,'登入已失效，請重新登入')
    return u
class Manager:
    def __init__(self): self.c=defaultdict(set)
    async def connect(self,w,u): await w.accept();self.c[u].add(w)
    def drop(self,w,u): self.c[u].discard(w)
    async def room(self,m,p):
        for u in m:
            for w in list(self.c[u]):
                try: await w.send_json(p)
                except Exception:self.drop(w,u)
def create_app():
    app=FastAPI(title='Visual Mask Multi-page Chat'); app.mount('/static',StaticFiles(directory=str(WEB)),name='static');app.mount('/output',StaticFiles(directory=str(OUT)),name='output')
    manager=Manager(); histories={}; explain_contexts={}; explanations={}
    for route,page in [('/','login.html'),('/login','login.html'),('/register','register.html'),('/appearance','appearance.html'),('/enroll','masks.html'),('/masks','masks.html'),('/chats','chats.html'),('/chat','chat.html')]:
        app.add_api_route(route,lambda page=page:FileResponse(WEB/page,headers={'Cache-Control':'no-store, max-age=0'}),methods=['GET'])
    @app.get('/health')
    def health(): return {'ok':True,'transport':'room-websocket','image_backend':'liveportrait','mask_management':'photo-masks'}
    @app.post('/api/register')
    def register(r:Registration):
        try:return login_result(register_user(r.account.strip(),r.username.strip(),r.password))
        except ValueError as e:raise HTTPException(400,str(e))
    @app.post('/api/login')
    def login(r:Credentials):
        u=verify_login(r.account.strip(),r.password)
        if not u:raise HTTPException(401,'帳號或密碼錯誤')
        return login_result(u)
    @app.get('/api/me')
    def me(u=Depends(current)):return public(u)
    @app.get('/api/appearance')
    def appearance(u=Depends(current)):return get_appearance(u['id'])
    @app.put('/api/appearance')
    def update_appearance(a:Appearance,u=Depends(current)):
        try:return save_appearance(u['id'],a.model_dump())
        except ValueError as e:raise HTTPException(400,str(e))
    @app.get('/lora-training')
    def old_training_page():
        from fastapi.responses import RedirectResponse
        return RedirectResponse('/masks',status_code=307)
    @app.post('/api/masks')
    async def add_photo_mask(name:str=Form(...),crop_json:str=Form(...),mode:str=Form('auto'),image:UploadFile=File(...),u=Depends(current)):
        try:
            content=await image.read(8*1024*1024+1)
            async with pipeline._generation_lock:
                return await asyncio.to_thread(portrait_masks.create,u['id'],u['username'],name,content,image.content_type,crop_json,mode)
        except ValueError as exc:raise HTTPException(400,str(exc))
        except RuntimeError as exc:raise HTTPException(503,str(exc))
    @app.get('/api/masks/{key}/preview')
    def photo_mask_preview(key:str,kind:str='base',u=Depends(current)):
        try:return FileResponse(portrait_masks.preview(key,u['id'],kind),headers={'Cache-Control':'private, no-store'})
        except PermissionError as exc:raise HTTPException(403,str(exc))
        except LookupError as exc:raise HTTPException(404,str(exc))
        except ValueError as exc:raise HTTPException(400,str(exc))
    @app.post('/api/masks/{key}/confirm')
    def confirm_photo_mask(key:str,payload:MaskConfirmation,u=Depends(current)):
        try:portrait_masks.confirm(key,u['id'],payload.accepted);return {'ready':True}
        except PermissionError as exc:raise HTTPException(403,str(exc))
        except LookupError as exc:raise HTTPException(404,str(exc))
        except ValueError as exc:raise HTTPException(400,str(exc))
    @app.delete('/api/masks/{key}')
    def delete_photo_mask(key:str,u=Depends(current)):
        try:return portrait_masks.delete(key,u['id'])
        except PermissionError as exc:raise HTTPException(403,str(exc))
        except LookupError as exc:raise HTTPException(404,str(exc))
        except ValueError as exc:raise HTTPException(400,str(exc))
    @app.get('/api/chat-candidates')
    def candidates(u=Depends(current)):return {'users':list_chat_candidates(u['id'])}
    @app.post('/api/chats/{room_id}/messages/{message_id}/explain')
    async def explain_message(room_id:str,message_id:str,u=Depends(current)):
        if not await asyncio.to_thread(is_room_member,room_id,u['id']):
            raise HTTPException(403,'你沒有此聊天室的存取權')
        ctx=explain_contexts.get(message_id)
        if not ctx or ctx['room_id'] != room_id:
            raise HTTPException(404,'找不到分析時的語境；目前僅支援本次啟動後分析的訊息')
        if message_id in explanations:
            return {'explanation':explanations[message_id],'cached':True}
        from src.services.emotion_service import explain_emotion
        result=await explain_emotion(ctx['text'],ctx['history'],ctx['emotion'],speaker=ctx['speaker'])
        if result.get('source') == 'llm':
            explanations[message_id]=result['explanation']
            if len(explanations)>1000: explanations.pop(next(iter(explanations)))
        return {**result,'cached':False}
    @app.get('/api/chats')
    def chats(u=Depends(current)):return {'rooms':list_rooms(u['id'])}
    @app.get('/api/masks')
    def masks(u=Depends(current)):
        portrait_masks.import_legacy(u['id'],u['username'])
        return {'masks':portrait_masks.list_masks(u['id'])}
    @app.post('/api/masks/active')
    def active_mask(payload:dict,u=Depends(current)):
        key=str(payload.get('mask_id','')).strip()
        try:portrait_masks.set_active(key,u['id'])
        except PermissionError as exc:raise HTTPException(403,str(exc))
        except LookupError as exc:raise HTTPException(404,str(exc))
        except ValueError as exc:raise HTTPException(400,str(exc))
        return {'active_mask':key,'masks':portrait_masks.list_masks(u['id'])}
    @app.get('/api/chats/{room_id}/messages')
    def chat_messages(room_id:str,u=Depends(current)):
        if not is_room_member(room_id,u['id']):raise HTTPException(403,'你沒有此聊天室的存取權')
        from db.message_feedback import corrections_for
        corrections=corrections_for(room_id)
        messages=list_messages(room_id)
        for message in messages:
            message['corrected_label']=corrections.get(message['msg_id'])
        return {'messages':messages}
    @app.post('/api/chats/{room_id}/messages/{message_id}/feedback')
    async def correct_emotion(room_id:str,message_id:str,payload:EmotionCorrection,u=Depends(current)):
        if not await asyncio.to_thread(is_room_member,room_id,u['id']):
            raise HTTPException(403,'你沒有此聊天室的存取權')
        from db.message_feedback import correct_message
        try:
            label=await asyncio.to_thread(correct_message,room_id,message_id,u['id'],u['username'],payload.correct_label)
        except PermissionError as exc: raise HTTPException(403,str(exc))
        except ValueError as exc: raise HTTPException(400,str(exc))
        return {'msg_id':message_id,'corrected_label':label}
    @app.post('/api/chats')
    def new_chat(p:Peer,u=Depends(current)):
        try:return {'room':get_or_create_direct_room(u['id'],p.peer_id)}
        except ValueError as e:raise HTTPException(400,str(e))
    @app.websocket('/ws/{room_id}')
    async def ws(w:WebSocket,room_id:str,token:str=''):
        p=verify_token(token);u=get_user(p['user_id']) if p else None
        if not u or not await asyncio.to_thread(is_room_member,room_id,u['id']):await w.close(code=4401);return
        await manager.connect(w,u['id']);members=await asyncio.to_thread(room_members,room_id);history=histories.setdefault(room_id,deque(maxlen=max(inference_config.ERC_HISTORY_TURNS,1)))
        try:
            while True:
                text=str(json.loads(await w.receive_text()).get('text','')).strip()
                if not text:continue
                mid=__import__('uuid').uuid4().hex;mask=await asyncio.to_thread(portrait_masks.active_key,u['id'])
                try:
                    await asyncio.to_thread(save_message,room_id,u['id'],text,None,None,mid)
                    context = list(history) if inference_config.ERC_HISTORY_TURNS else []
                    history.append({'role':u['username'],'content':text})
                    if not mask:raise ValueError('請先到面具管理新增並確認一張面具')
                    a=await pipeline.analyze(text,context,user_id=u['username'],mask_id=mask)
                    explain_contexts[mid]={'room_id':room_id,'text':text,'history':a.history,'emotion':a.emotion,'speaker':u['username']}
                    if len(explain_contexts)>1000: explain_contexts.pop(next(iter(explain_contexts)))
                    await manager.room(members,{'msg_id':mid,'sender':u['username'],'text':text,'ai_label':a.emotion,'status':'generating'})
                    result=await pipeline.generate(a);await asyncio.to_thread(update_message_generation,mid,result.emotion,result.public_path)
                    await manager.room(members,{'msg_id':mid,'sender':u['username'],'text':text,'ai_label':a.emotion,'mask_file':result.public_path,'status':'done'})
                except Exception as e:await manager.room(members,{'msg_id':mid,'sender':u['username'],'text':text,'status':'error','error':str(e)})
        except WebSocketDisconnect:manager.drop(w,u['id'])
    return app
app=create_app()
