"""Ownership checked deletion of raw/processed uploads."""
import shutil
from pathlib import Path
from db.connection import get_connection
from src.services.lora_training import jobs_for, save, ROOT, PHOTO_JOB_LOCK

def delete_photo(user_id: int, photo_id: int):
    with PHOTO_JOB_LOCK:
        return _delete_photo(user_id, photo_id)

def _delete_photo(user_id: int, photo_id: int):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute('SELECT path FROM user_training_photos WHERE id=%s AND user_id=%s',(photo_id,user_id))
            row=cur.fetchone()
            if not row: raise LookupError('找不到照片或無權刪除')
            path=Path(row[0]).resolve()
            base=(ROOT/'data'/'user_lora_training'/str(user_id)).resolve()
            if not path.is_relative_to(base) or path.parent.name not in ('raw','processed'):
                raise ValueError('照片路徑不合法')
            targets=[(base/d/path.name).resolve() for d in ('raw','processed')]
            if any(not p.is_relative_to(base) for p in targets): raise ValueError('照片路徑不合法')
            jobs=jobs_for(user_id)
            matches=lambda j:any(Path(p).resolve() in targets for p in j.get('paths',[]))
            if any(j['status'] in ('queued','training') and matches(j) for j in jobs):
                raise ValueError('照片正在排隊或訓練中，暫時無法刪除')
            # Completed masks still need their LivePortrait source. Preserve
            # that model asset independently from the uploaded-photo library.
            for job in jobs:
                if job['status']=='ready' and matches(job):
                    asset=ROOT/'models'/'user_loras'/job['id']/'sources'/path.name
                    asset.parent.mkdir(parents=True,exist_ok=True)
                    if path.is_file(): shutil.copy2(path,asset)
                    elif not asset.is_file(): raise ValueError('面具來源照片遺失，無法安全刪除')
                    job['paths']=[str(asset) if Path(p).resolve() in targets else p for p in job['paths']]
                    save(job)
            for target in targets:
                target.unlink(missing_ok=True)
            cur.execute('DELETE FROM user_training_photos WHERE id=%s AND user_id=%s',(photo_id,user_id))
        conn.commit()
