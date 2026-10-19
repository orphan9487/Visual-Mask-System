"""Per-mask inference settings, independent of the training parameters."""
import json
from pathlib import Path
from src.services.lora_training import JOBS, ROOT, PHOTO_JOB_LOCK, save

DEFAULTS = dict(source_index=0, checkpoint=300, lora_weight=0.7, strength=0.22, seed=42)

def read_job(job_id):
    if len(job_id)!=32 or any(c not in '0123456789abcdef' for c in job_id): raise ValueError('面具 ID 不合法')
    path=JOBS/(job_id+'.json')
    if not path.is_file(): raise LookupError('找不到面具')
    return json.loads(path.read_text(encoding='utf-8'))

def settings(job):
    return {**DEFAULTS, **job.get('generation',{})}

def checkpoint_path(job, checkpoint):
    path=ROOT/'models'/'user_loras'/job['id']/str(checkpoint)/'pytorch_lora_weights.safetensors'
    if not path.is_file(): raise ValueError('權重版本不存在')
    return str(path)

def update(job_id, user_id, values):
    with PHOTO_JOB_LOCK:
        job=read_job(job_id)
        if job['user_id']!=user_id: raise PermissionError('無權修改此面具')
        if job['status']!='ready': raise ValueError('面具尚未訓練完成')
        config={**settings(job),**values}
        if not 0<=config['source_index']<len(job['paths']): raise ValueError('基底照片不存在')
        if config['checkpoint'] not in (100,200,300): raise ValueError('權重版本不合法')
        if not 0<=config['lora_weight']<=1: raise ValueError('LoRA 強度必須介於 0～1')
        if not .1<=config['strength']<=.5: raise ValueError('重繪強度必須介於 0.1～0.5')
        if not 0<=config['seed']<=2147483647: raise ValueError('seed 超出範圍')
        checkpoint_path(job,config['checkpoint'])
        if not Path(job['paths'][config['source_index']]).is_file(): raise ValueError('來源照片遺失')
        job['generation']=config;save(job)
        return config
