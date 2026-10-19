"""U2-Net background normalization on E inputs; prepare and inspect before training."""
import argparse
import hashlib
import json
import sys
import uuid
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def main():
    import numpy as np
    import onnxruntime as ort
    from PIL import Image, ImageDraw
    from src.services.lora_training import JOBS,save
    from src.services.mask_generation_settings import read_job
    ap=argparse.ArgumentParser()
    ap.add_argument('--source-job')
    ap.add_argument('--train-job')
    args=ap.parse_args()
    if bool(args.source_job)==bool(args.train_job):
        ap.error('Choose --source-job (prepare) or --train-job (train inspected inputs)')
    if args.train_job:
        for p in JOBS.glob('*.json'):
            j=json.loads(p.read_text(encoding='utf-8'))
            if j['status'] in ('training','queued') or j.get('base_status')=='generating':
                raise RuntimeError('GPU already has an active job')
        from scripts.training.train_user_lora import train
        job=read_job(args.train_job)
        if job.get('preprocessing')!='u2net_gray_background' or job['status']!='prepared':
            raise ValueError('Expected prepared background experiment')
        for path,digest in zip(job['paths'],job['input_hashes']):
            if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:
                raise ValueError('Prepared inputs changed')
        job['status']='training';save(job)
        try:train(job)
        except Exception as exc:job.update(status='failed',error=str(exc));save(job);raise
        print('TRAINING_COMPLETE',job['id'],flush=True)
        return
    original=read_job(args.source_job)
    if not original.get('preprocessing_audit'):
        raise ValueError('Expected normalized framing experiment as source')
    model=ROOT/'models/background_removal/u2net.onnx'
    if hashlib.md5(model.read_bytes()).hexdigest()!='60024c5c889badc19c04ad937298a77b':
        raise ValueError('U2-Net checksum mismatch')
    session=ort.InferenceSession(str(model),providers=['CPUExecutionProvider'])
    job_id=uuid.uuid4().hex
    out=ROOT/'output/current/lora_background_experiment'/job_id
    out.mkdir(parents=True)
    sheet=Image.new('RGB',(1536,550*len(original['paths'])),'#fff3e3')
    draw=ImageDraw.Draw(sheet)
    paths,audit=[],[]
    for i,source in enumerate(original['paths']):
        with Image.open(source) as raw:im=raw.convert('RGB')
        # Preprocessing and output normalization follow rembg's U2netSession.
        arr=np.asarray(im.resize((320,320),Image.Resampling.LANCZOS)).astype(np.float32)
        arr/=max(float(arr.max()),1e-6)
        arr=(arr-np.array([.485,.456,.406],dtype=np.float32))/np.array([.229,.224,.225],dtype=np.float32)
        tensor=np.transpose(arr,(2,0,1))[None].astype(np.float32)
        pred=session.run(None,{session.get_inputs()[0].name:tensor})[0][0,0]
        pred=(pred-pred.min())/max(float(pred.max()-pred.min()),1e-6)
        mask=Image.fromarray((pred.clip(0,1)*255).astype(np.uint8)).resize(im.size,Image.Resampling.LANCZOS)
        mask.save(out/f'mask-{i}.png')
        normalized=Image.composite(im,Image.new('RGB',im.size,(127,127,127)),mask)
        path=out/f'photo-{i}.png';normalized.save(path);paths.append(str(path))
        audit.append(dict(source=source,source_sha256=hashlib.sha256(Path(source).read_bytes()).hexdigest(),path=str(path)))
        for col,img in enumerate((im,mask.convert('RGB'),normalized)):
            sheet.paste(img.resize((512,512)),(col*512,i*550+38))
            draw.text((col*512+10,i*550+10),('E input','U2-Net mask','F gray background')[col],fill='black')
    sheet.save(out/'preprocessing_comparison.png')
    job=dict(id=job_id,user_id=original['user_id'],username=original['username'],
             mask_name=original['mask_name']+' experiment F background',paths=paths,
             photo_ids=original.get('photo_ids',[]),parameters=dict(original['parameters']),
             trigger=original['trigger'],source_job=original['id'],experiment=True,
             preprocessing='u2net_gray_background',preprocessing_audit=audit,
             reference_paths=[a['source'] for a in original['preprocessing_audit']],
             input_hashes=[hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths],
             status='prepared',step=0,error=None)
    save(job)
    print('PREPARED_JOB',job_id,flush=True)
    print('PREVIEW',out/'preprocessing_comparison.png',flush=True)


if __name__=='__main__':main()
