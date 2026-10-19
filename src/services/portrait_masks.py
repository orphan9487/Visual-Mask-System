"""Private photo masks, independent of LoRA jobs and weights."""
import io
import json
import re
import threading
import uuid
from pathlib import Path
from PIL import Image,ImageOps

ROOT=Path(__file__).resolve().parents[2]
STORE=ROOT/'data/portrait_masks'
LOCK=threading.RLock()


def _write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8');temporary.replace(path)


def read(key,user_id=None):
    if not isinstance(key,str):raise ValueError('面具 ID 不合法')
    match=re.fullmatch(r'portrait_([1-9][0-9]*)_([a-f0-9]{32})',key)
    if not match:raise ValueError('面具 ID 不合法')
    owner=int(match[1])
    if user_id is not None and owner!=user_id:raise PermissionError('無權操作此面具')
    path=STORE/str(owner)/match[2]/'mask.json'
    if not path.is_file():raise LookupError('面具不存在')
    data=json.loads(path.read_text(encoding='utf-8'))
    if data['status']=='deleted':raise LookupError('面具已刪除')
    return data


def _path(record):return STORE/str(record['user_id'])/record['id']/'mask.json'


def list_masks(user_id):
    folder=STORE/str(user_id)
    if not folder.exists():return []
    active=active_key(user_id)
    records=[json.loads(p.read_text(encoding='utf-8')) for p in folder.glob('*/mask.json')]
    return [dict(key=r['key'],display_name=r['name'],status=r['status'],method=r['method'],is_active=r['key']==active) for r in records if r['status']!='deleted']


def import_legacy(user_id,username):
    """Copy existing uploaded bases once, without reading/loading LoRA weights."""
    with LOCK:
        marker=STORE/str(user_id)/'legacy_import.json'
        if marker.exists():return
        imported=[]
        for path in (ROOT/'data/lora_jobs').glob('*.json'):
            old=json.loads(path.read_text(encoding='utf-8'))
            if old.get('user_id')!=user_id or old.get('status')!='ready' or not old.get('selected_base'):continue
            source=Path(old['selected_base']).resolve()
            if not source.is_relative_to(ROOT.resolve()) or not source.is_file():continue
            identifier=uuid.uuid4().hex;folder=STORE/str(user_id)/identifier;folder.mkdir(parents=True)
            with Image.open(source) as im:
                ImageOps.pad(ImageOps.exif_transpose(im).convert('RGB'),(1024,1024),color=(127,127,127)).save(folder/'base.png')
            record=dict(id=identifier,key=f'portrait_{user_id}_{identifier}',user_id=user_id,username=username,
                        name=old.get('mask_name','既有面具'),status='ready' if old.get('base_source')=='user_confirmed_neutral' else 'draft',
                        source=str(folder/'base.png'),base=str(folder/'base.png'),method='legacy_base_import',legacy_job=old['id'])
            _write(folder/'mask.json',record);imported.append(record)
        if not active_key(user_id):
            first=next((r for r in imported if r['status']=='ready'),None)
            if first:set_active(first['key'],user_id)
        _write(marker,{'imported':[r['key'] for r in imported]})


def active_key(user_id):
    path=STORE/str(user_id)/'active.json'
    if path.is_file():
        key=json.loads(path.read_text(encoding='utf-8')).get('key')
        try:
            if read(key,user_id)['status']=='ready':return key
        except (ValueError,LookupError):pass
    return None


def set_active(key,user_id):
    with LOCK:
        record=read(key,user_id)
        if record['status']!='ready':raise ValueError('請先預覽並確認面具')
        _write(STORE/str(user_id)/'active.json',{'key':key})


def square_crop(image,crop_json):
    try:
        c=json.loads(crop_json);x,y,w,h=(c[k] for k in ('x','y','width','height'))
        limit=2*max(image.size)
        left=(image.width-limit)//2;top=(image.height-limit)//2
        if any(type(v) is not int for v in (x,y,w,h)) or w!=h or w<256 or w>limit or x<left or y<top or x+w>left+limit+1 or y+h>top+limit+1:raise ValueError()
        if x>=image.width or y>=image.height or x+w<=0 or y+h<=0:raise ValueError()
    except (ValueError,TypeError,KeyError):raise ValueError('請在白色畫布內選取至少 256×256 的正方形，且包含照片中的人臉')
    # Render directly to bounded output: never allocate a huge padded canvas.
    return image.transform((1024,1024),Image.Transform.EXTENT,(x,y,x+w,y+h),resample=Image.Resampling.BICUBIC,fillcolor=(255,255,255))


def create(user_id,username,name,image_bytes,content_type,crop_json,mode='auto'):
    if mode not in ('auto','natural','new_identity'):raise ValueError('不支援的模式')
    name=name.strip()
    if not 1<=len(name)<=100:raise ValueError('面具名稱需為 1～100 字')
    if content_type not in ('image/jpeg','image/png','image/webp') or not image_bytes or len(image_bytes)>8*1024*1024:raise ValueError('請上傳 8 MB 以內的 JPG、PNG 或 WebP')
    try:
        with Image.open(io.BytesIO(image_bytes)) as raw:image=ImageOps.exif_transpose(raw).convert('RGB')
    except Exception as exc:raise ValueError('無法讀取照片') from exc
    image=square_crop(image,crop_json)
    with LOCK:
        identifier=uuid.uuid4().hex;folder=STORE/str(user_id)/identifier;folder.mkdir(parents=True)
        source=folder/'source.png';image.save(source)
        from src.services.face_profile_service import _validate_single_face
        _validate_single_face(source)
        base=folder/'base.png';method='natural';score=None
        # Each new mask is independent, even when the account already has a
        # confirmed mask of the same person. Never substitute an older photo.
        if mode=='natural':image.save(base)
        else:
            from src.integrations.liveportrait_expression import generate_from_expression_profile
            output=generate_from_expression_profile({'emotion':'neutral','identity':{'source_override':str(source),'neutralize_source':True}})
            with Image.open(output) as rendered:ImageOps.pad(rendered.convert('RGB'),(1024,1024),color=(127,127,127)).save(base)
            _validate_single_face(base)
            score=identity_similarity(source,base)
            if score<.30:raise ValueError('中性化後相似度不足，請補照片，結果不會啟用')
            method='neutral_attempt'
        record=dict(id=identifier,key=f'portrait_{user_id}_{identifier}',user_id=user_id,username=username,name=name,status='draft',source=str(source),base=str(base),method=method,identity_cosine=score)
        _write(folder/'mask.json',record)
        return {'key':record['key'],'method':method,'status':'draft','message':{'natural':'原照片，未修改表情','reuse_reference':'沿用已確認的本人基底，不是重畫新照片','neutral_attempt':'自動中性化嘗試，請檢查大小眼、嘴形與本人相似度'}[method]}


def identity_similarity(first,second):
    import cv2
    import numpy as np
    from insightface.app import FaceAnalysis
    analyzer=FaceAnalysis(name='buffalo_l',root=str(ROOT/'models/insightface'),providers=['CPUExecutionProvider']);analyzer.prepare(ctx_id=-1,det_size=(640,640))
    embeddings=[]
    for path in (first,second):
        image=cv2.imdecode(np.fromfile(str(path),dtype=np.uint8),1);faces=analyzer.get(image)
        if not faces:
            pad=int(max(image.shape[:2])*.25);faces=analyzer.get(cv2.copyMakeBorder(image,pad,pad,pad,pad,cv2.BORDER_CONSTANT,value=(127,127,127)))
        if len(faces)!=1:raise ValueError('無法可靠比對身分，請補清晰正臉照')
        embeddings.append(faces[0].normed_embedding)
    return float(np.dot(*embeddings))


def preview(key,user_id,kind='base'):
    record=read(key,user_id)
    if kind not in ('source','base'):raise ValueError('不支援的預覽')
    path=Path(record[kind]).resolve()
    if not path.is_relative_to(_path(record).parent.resolve()) or not path.is_file():raise LookupError('照片不存在')
    return path


def confirm(key,user_id,accepted):
    with LOCK:
        if accepted is not True:raise ValueError('請確認預覽像本人且表情自然')
        record=read(key,user_id);preview(key,user_id)
        record['status']='ready';_write(_path(record),record);set_active(key,user_id)


def delete(key,user_id):
    with LOCK:
        record=read(key,user_id);was_active=active_key(user_id)==key
        record['status']='deleted';_write(_path(record),record)
        if was_active:
            remaining=next((r for r in list_masks(user_id) if r['status']=='ready'),None)
            _write(STORE/str(user_id)/'active.json',{'key':remaining['key'] if remaining else None})
        return {'deleted':key,'active_mask':active_key(user_id)}
