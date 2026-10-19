"""Persistent, serialized GPU training jobs. Run worker in its own process."""
import json
import subprocess
import sys
import uuid
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
JOBS = ROOT / 'data' / 'lora_jobs'
PHOTO_JOB_LOCK = threading.RLock()
PARAMS = dict(steps=300, rank=8, alpha=8, learning_rate=5e-5,
              resolution=512, batch_size=1, gradient_accumulation=1, seed=42)

def save(job):
    if job.get('smoke_test'):
        return
    path = JOBS / (job['id'] + '.json')
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)

def jobs_for(user_id):
    JOBS.mkdir(parents=True, exist_ok=True)
    return [j for p in JOBS.glob('*.json') if (j := json.loads(p.read_text(encoding='utf-8')))['user_id'] == user_id]

def submit(user_id, username, mask_name, photo_ids):
    with PHOTO_JOB_LOCK:
        return _submit(user_id, username, mask_name, photo_ids)

def _submit(user_id, username, mask_name, photo_ids):
    from db.training_repo import list_training_photos
    if not mask_name.strip() or len(mask_name.strip()) > 100:
        raise ValueError('請輸入 1～100 字的面具名稱')
    photos = {p['id']: p for p in list_training_photos(user_id)}
    ids = list(dict.fromkeys(photo_ids))
    if len(ids) != 3 or any(i not in photos or photos[i]['status'] != 'ready' for i in ids):
        raise ValueError('請選擇三張已通過處理的照片')
    JOBS.mkdir(parents=True, exist_ok=True)
    if any(j['status'] in ('queued','training') or j.get('base_status')=='generating' for p in JOBS.glob('*.json') for j in [json.loads(p.read_text(encoding='utf-8'))]):
        raise ValueError('GPU 已有訓練任務，請等候完成')
    job = dict(id=uuid.uuid4().hex, user_id=user_id, username=username,
               mask_name=mask_name.strip(), paths=[photos[i]['path'] for i in ids],
               photo_ids=ids,
               status='queued', step=0, parameters=PARAMS, error=None)
    save(job)
    log = (JOBS / (job['id'] + '.log')).open('w', encoding='utf-8')
    try:
        subprocess.Popen([sys.executable, '-m', 'scripts.training.train_user_lora', '--job', job['id']], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    except Exception as exc:
        job.update(status='failed', error=str(exc)); save(job)
        raise
    finally:
        log.close()
    return job
