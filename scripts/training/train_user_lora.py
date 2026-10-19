"""SD1.5 UNet attention LoRA, frozen text encoder; fixed three-photo preset."""
import argparse
import json
import random
from pathlib import Path

def train(job):
    import torch
    import torch.nn.functional as F
    from PIL import Image, ImageOps
    from diffusers import StableDiffusionPipeline, DDPMScheduler
    from diffusers.utils import convert_state_dict_to_diffusers
    from peft import LoraConfig
    from peft.utils import get_peft_model_state_dict
    from src.generation.diffusion_engine import SD_MODEL_ID
    from src.services.lora_training import ROOT, save
    if not torch.cuda.is_available(): raise RuntimeError('需要可用的 NVIDIA CUDA GPU')
    p = job['parameters']; torch.manual_seed(p['seed']); random.seed(p['seed'])
    output = ROOT / 'models' / 'user_loras' / job['id']; output.mkdir(parents=True, exist_ok=True)
    pipe = StableDiffusionPipeline.from_pretrained(SD_MODEL_ID, local_files_only=True, torch_dtype=torch.float16).to('cuda')
    vae, unet, encoder = pipe.vae, pipe.unet, pipe.text_encoder
    vae.requires_grad_(False); encoder.requires_grad_(False); unet.requires_grad_(False)
    unet.add_adapter(LoraConfig(r=p['rank'], lora_alpha=p['alpha'], init_lora_weights='gaussian', target_modules=['to_q','to_k','to_v','to_out.0']))
    unet_params = [v for v in unet.parameters() if v.requires_grad]
    text_lr = p.get('text_encoder_learning_rate',0)
    text_params = []
    if text_lr:
        encoder.add_adapter(LoraConfig(r=8,lora_alpha=8,init_lora_weights='gaussian',target_modules=['q_proj','k_proj','v_proj','out_proj']))
        text_params = [v for v in encoder.parameters() if v.requires_grad]
        encoder.train()
    params = unet_params + text_params
    for v in params: v.data = v.data.float()
    unet.enable_gradient_checkpointing(); unet.train()
    groups=[{'params':unet_params,'lr':p['learning_rate']}]
    if text_params: groups.append({'params':text_params,'lr':text_lr})
    optimizer = torch.optim.AdamW(groups)
    scaler = torch.amp.GradScaler('cuda')
    scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    trigger = job.get('trigger') or 'vms' + job['id'][:12]
    tokens = pipe.tokenizer('photo of ' + trigger + ' person', padding='max_length', max_length=77, truncation=True, return_tensors='pt').input_ids.to('cuda')
    with torch.no_grad():
        hidden = encoder(tokens)[0]
        latents = []
        for path in job['paths']:
            with Image.open(path) as im:
                im = ImageOps.fit(ImageOps.exif_transpose(im).convert('RGB'), (512,512))
                import numpy as np
                pixels = torch.from_numpy(np.array(im).copy()).permute(2,0,1).unsqueeze(0).to('cuda', dtype=torch.float16) / 127.5 - 1
                latents.append(vae.encode(pixels).latent_dist.sample() * vae.config.scaling_factor)
    for step in range(1, p['steps']+1):
        latent = latents[(step-1) % len(latents)]
        noise = torch.randn_like(latent)
        ts = torch.randint(0, scheduler.config.num_train_timesteps, (1,), device='cuda').long()
        noisy = scheduler.add_noise(latent, noise, ts)
        with torch.autocast('cuda', dtype=torch.float16):
            if text_params:
                hidden = encoder(tokens)[0]
            prediction = unet(noisy, ts, hidden).sample
            target = scheduler.get_velocity(latent, noise, ts) if scheduler.config.prediction_type == 'v_prediction' else noise
            loss = F.mse_loss(prediction.float(), target.float())
        scaler.scale(loss).backward(); scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
        job.update(step=step, loss=float(loss.detach()), status='training'); save(job)
        if job.get('experiment'):
            with (output/'training.jsonl').open('a',encoding='utf-8') as log:
                log.write(json.dumps({'step':step,'loss':job['loss'],'photo_index':(step-1)%len(latents),'amp_scale':scaler.get_scale()})+'\n')
        if step % 100 == 0:
            folder = output / str(step)
            state = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
            text_state=convert_state_dict_to_diffusers(get_peft_model_state_dict(encoder)) if text_params else None
            StableDiffusionPipeline.save_lora_weights(str(folder), unet_lora_layers=state, text_encoder_lora_layers=text_state, safe_serialization=True)
    if job.get('smoke_test'):
        return
    if job.get('experiment'):
        job.update(status='ready',trigger=trigger,lora_path=str(output/str(p['steps'])/'pytorch_lora_weights.safetensors'))
        save(job)
        return
    from db.connection import get_connection
    key = 'lora_' + job['id']
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute('INSERT INTO user_loras (username,lora_key,display_name,is_active) VALUES (%s,%s,%s,0)', (job['username'],key,job['mask_name']))
        conn.commit()
    job.update(status='ready', mask_id=key, trigger=trigger, lora_path=str(output/str(p['steps'])/'pytorch_lora_weights.safetensors')); save(job)
    # Release training tensors before the text-to-image candidate pipeline.
    del pipe,vae,unet,encoder,optimizer,params,latents,hidden,prediction,loss,latent,noisy
    import gc
    gc.collect();torch.cuda.empty_cache()
    # Candidate-base generation is retired; training ends after saving weights.

if __name__ == '__main__':
    from src.services.lora_training import JOBS, save
    ap=argparse.ArgumentParser(); ap.add_argument('--job',required=True); args=ap.parse_args()
    job=json.loads((JOBS/(args.job+'.json')).read_text(encoding='utf-8'))
    try:
        job['status']='training'; save(job); train(job)
    except Exception as exc:
        job.update(status='failed',error=str(exc)); save(job)
        raise
