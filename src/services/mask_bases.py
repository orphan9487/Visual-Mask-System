"""Generate neutral portraits with the trained LoRA, never a training photo."""
import json
import subprocess
import sys
from pathlib import Path
from src.services.lora_training import ROOT, JOBS, PHOTO_JOB_LOCK, save
from src.services.mask_generation_settings import read_job, checkpoint_path

def owned(job_id,user_id):
    job=read_job(job_id)
    if job['user_id']!=user_id: raise PermissionError('無權存取此面具')
    if job['status']!='ready': raise ValueError('面具尚未訓練完成')
    return job

def launch(job_id,user_id):
    with PHOTO_JOB_LOCK:
        job=owned(job_id,user_id)
        if any(j.get('base_status')=='generating' or j['status'] in ('queued','training') for p in JOBS.glob('*.json') for j in [json.loads(p.read_text(encoding='utf-8'))]):
            raise ValueError('GPU 正忙碌，請稍後再試')
        job['base_status']='generating';job['base_error']=None;save(job)
        try:
            with (JOBS/(job_id+'.bases.log')).open('w',encoding='utf-8') as log:
                subprocess.Popen([sys.executable,'-m','src.services.mask_bases','--job',job_id],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        except Exception as exc:
            job['base_status']='failed';job['base_error']=str(exc);save(job);raise

def generate(job_id):
    import torch
    from diffusers import StableDiffusionPipeline
    from src.generation.diffusion_engine import SD_MODEL_ID
    job=read_job(job_id)
    try:
        job['base_status']='generating';save(job)
        pipe=StableDiffusionPipeline.from_pretrained(SD_MODEL_ID,torch_dtype=torch.float16,local_files_only=True).to('cuda')
        weight=Path(checkpoint_path(job,300))
        pipe.load_lora_weights(str(weight.parent),weight_name=weight.name,adapter_name='person')
        pipe.set_adapters(['person'],adapter_weights=[.7])
        folder=ROOT/'models'/'user_loras'/job_id/'bases';folder.mkdir(parents=True,exist_ok=True)
        import uuid
        batch=uuid.uuid4().hex[:12];candidates=[]
        for index,seed in enumerate((42,123,314159)):
            result=pipe(prompt=f"photo of {job['trigger']} person, front facing centered portrait, neutral expression, relaxed closed mouth, full head visible, plain light studio background, natural skin, soft light",negative_prompt='side view, tilted head, cropped head, multiple people, busy background, text, blurry, deformed',width=512,height=512,num_inference_steps=30,guidance_scale=7,generator=torch.Generator('cuda').manual_seed(seed))
            if any(result.nsfw_content_detected or []): raise ValueError('候選圖片未通過生成檢查，請重新產生')
            path=folder/f'{batch}-{index}.png';result.images[0].save(path)
            candidates.append(str(path))
        job=read_job(job_id);job.update(base_candidates=candidates,base_status='ready',base_error=None);save(job)
    except Exception as exc:
        job=read_job(job_id);job.update(base_status='failed',base_error=str(exc));save(job);raise

def choose(job_id,user_id,index):
    with PHOTO_JOB_LOCK:
        job=owned(job_id,user_id)
        if job.get('base_status')!='ready':raise ValueError('候選圖片尚未完成')
        candidates=job.get('base_candidates',[])
        if not 0<=index<len(candidates) or not Path(candidates[index]).is_file(): raise ValueError('候選圖片不存在')
        job['selected_base']=candidates[index];save(job)
        return {'selected':index}

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--job',required=True);generate(parser.parse_args().job)
