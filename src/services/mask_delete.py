"""Recoverable removal of a trained personal mask from the user's library."""
from datetime import datetime, timezone
from db.connection import get_connection
from src.services.lora_training import PHOTO_JOB_LOCK, save
from src.services.mask_generation_settings import read_job

def delete_mask(job_id: str, user_id: int, username: str):
    with PHOTO_JOB_LOCK:
        job=read_job(job_id)
        if job['user_id']!=user_id: raise PermissionError('無權刪除此面具')
        if job['status'] in ('queued','training') or job.get('base_status')=='generating': raise ValueError('面具正在訓練或產生基底，暫時無法刪除')
        if job['status']=='deleted': return {'deleted':job_id}
        key='lora_'+job_id
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute('SELECT is_active FROM user_loras WHERE username=%s AND lora_key=%s FOR UPDATE',(username,key))
                row=cur.fetchone()
                fallback=None
                if row and row[0]:
                    cur.execute("SELECT mask_id FROM face_profiles WHERE user_id=%s AND status='ready'",(user_id,))
                    profile=cur.fetchone()
                    if not profile: raise ValueError('請先建立 FaceID 面具，再刪除目前使用中的面具')
                    fallback=profile[0]
                    cur.execute('UPDATE user_loras SET is_active=0 WHERE username=%s',(username,))
                    cur.execute('INSERT INTO user_loras (username,lora_key,display_name,is_active) VALUES (%s,%s,%s,1) ON DUPLICATE KEY UPDATE is_active=1',(username,fallback,username+' 的 FaceID 面具'))
                cur.execute('DELETE FROM user_loras WHERE username=%s AND lora_key=%s',(username,key))
                cur.execute('UPDATE users SET default_mask_id=%s WHERE id=%s AND default_mask_id=%s',(fallback or 'faceid_user_'+str(user_id),user_id,key))
            conn.commit()
        job['previous_status']=job['status'];job['status']='deleted'
        job['deleted_at']=datetime.now(timezone.utc).isoformat();save(job)
        return {'deleted':job_id,'active_mask':fallback}
