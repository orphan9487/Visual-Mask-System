"""Store an explicitly uploaded base for an owned trained mask."""
import io
import uuid
from pathlib import Path
from PIL import Image, ImageOps
from src.services.lora_training import ROOT, PHOTO_JOB_LOCK, save
from src.services.mask_bases import owned


def identity_similarity(first,second):
    """Conservative identity gate, not an expression or symmetry validator."""
    import cv2
    import numpy as np
    from insightface.app import FaceAnalysis
    analyzer=FaceAnalysis(name='buffalo_l',root=str(ROOT/'models/insightface'),providers=['CPUExecutionProvider'])
    analyzer.prepare(ctx_id=-1,det_size=(640,640))
    embeddings=[]
    for path in (first,second):
        image=cv2.imdecode(np.fromfile(str(path),dtype=np.uint8),1)
        faces=analyzer.get(image)
        if not faces:
            pad=int(max(image.shape[:2])*.25)
            faces=analyzer.get(cv2.copyMakeBorder(image,pad,pad,pad,pad,cv2.BORDER_CONSTANT,value=(127,127,127)))
        if len(faces)!=1:raise ValueError('無法可靠比對身分，請補一張清晰正臉照')
        embeddings.append(faces[0].normed_embedding)
    return float(np.dot(*embeddings))


def upload_base(job_id,user_id,image_bytes,content_type,mode='auto'):
    if mode not in ('auto','natural','new_identity'):raise ValueError('不支援的基底建立模式')
    if content_type not in {'image/jpeg','image/png','image/webp'}:
        raise ValueError('請上傳 JPG、PNG 或 WebP')
    if not image_bytes or len(image_bytes)>8*1024*1024:
        raise ValueError('圖片不得超過 8 MB')
    with PHOTO_JOB_LOCK:
        job=owned(job_id,user_id)
        if job.get('base_status')=='generating':raise ValueError('面具仍在處理中，請稍後再試')
        try:
            with Image.open(io.BytesIO(image_bytes)) as raw:
                image=ImageOps.exif_transpose(raw).convert('RGB')
        except Exception as exc:raise ValueError('無法讀取圖片') from exc
        if min(image.size)<256:raise ValueError('圖片短邊至少需要 256 像素')
        image.thumbnail((2048,2048),Image.Resampling.LANCZOS)
        folder=ROOT/'models/user_loras'/job_id/'uploaded_bases'
        folder.mkdir(parents=True,exist_ok=True)
        path=folder/f'{uuid.uuid4().hex}.png'
        image.save(path)
        try:
            from src.services.face_profile_service import _validate_single_face
            _validate_single_face(path)
            trusted=job.get('neutral_identity_reference') or (job.get('selected_base') if job.get('base_source')=='user_confirmed_neutral' else None)
            method='natural_neutral_photo';neutral_path=path;score=None
            if mode!='natural':
                if trusted and Path(trusted).is_file() and mode!='new_identity':
                    score=identity_similarity(path,trusted)
                    if score<.30:raise ValueError('新照片與既有身分的相似度不足，無法安全沿用基底。請補清晰正臉照，或明確選擇建立新身分。')
                    # Reuse, not expression restoration of the newly uploaded image.
                    neutral_path=Path(trusted);method='reuse_confirmed_reference'
                else:
                    from src.integrations.liveportrait_expression import generate_from_expression_profile
                    rendered=generate_from_expression_profile({'emotion':'neutral','identity':{'source_override':str(path),'neutralize_source':True}})
                    neutral_path=folder/f'{path.stem}-attempt.png'
                    with Image.open(rendered) as neutral:neutral.convert('RGB').save(neutral_path)
                    _validate_single_face(neutral_path)
                    score=identity_similarity(path,neutral_path)
                    if score<.30:raise ValueError('中性化後的身分相似度不足，本次結果不啟用。請補充清楚正臉或自然中性照片。')
                    method='automatic_neutral_attempt'
            job.update(pending_base={'token':path.stem,'source':str(path),'neutral':str(neutral_path),
                                     'method':method,'identity_cosine':score})
            save(job)
        except Exception:
            path.unlink(missing_ok=True)
            raise
        message={'natural_neutral_photo':'原照片預覽，未修改表情。',
                 'reuse_confirmed_reference':'身分初步比對通過，以下沿用既有中性基底；不是將新照片重繪成中性，背景與姿態也沿用舊基底。',
                 'automatic_neutral_attempt':'已嘗試自動中性化，尚未驗收；大小眼、歪嘴或不像本人時，請勿確認，請補照片。'}[method]
        return {'ready':False,'token':path.stem,'method':method,'identity_cosine':score,'message':message+' 確認前不會替換目前基底。'}


def preview_base(job_id,user_id,kind='selected',token=None):
    job=owned(job_id,user_id)
    if kind in ('source','neutral'):
        pending=job.get('pending_base',{})
        if not token or token!=pending.get('token'):raise LookupError('此預覽已失效，請重新上傳')
        value=pending.get(kind,'')
    else:value=job.get('selected_base','')
    path=Path(value).resolve()
    root=(ROOT/'models/user_loras'/job_id).resolve()
    if not path.is_relative_to(root) or not path.is_file():raise LookupError('尚未上傳基底圖片')
    return path


def confirm_base(job_id,user_id,token,neutral_confirmed=False):
    with PHOTO_JOB_LOCK:
        job=owned(job_id,user_id)
        path=preview_base(job_id,user_id,'neutral',token)
        pending=job['pending_base']
        if pending.get('method') not in ('natural_neutral_photo','reuse_confirmed_reference','automatic_neutral_attempt'):raise ValueError('此預覽已過期，請重新上傳')
        if neutral_confirmed is not True:raise ValueError('請先確認照片符合自然中性表情要求')
        job.update(selected_base=str(path),base_source='user_confirmed_neutral',
                   original_base=pending['source'],neutral_identity_reference=str(path),
                   neutral_reference_method=pending['method'])
        job.pop('pending_base',None)
        save(job)
        return {'ready':True,'message':'中性基底已確認，下一則訊息生效'}
