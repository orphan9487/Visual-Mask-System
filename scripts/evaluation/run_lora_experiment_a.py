"""Independent 600-step replay of an existing three-photo training job."""
import argparse
import hashlib
import json
import sys
import uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))

def main():
    from src.services.mask_generation_settings import read_job
    from src.services.lora_training import JOBS,save
    from scripts.training.train_user_lora import train
    parser=argparse.ArgumentParser();parser.add_argument('--source-job',required=True);parser.add_argument('--rank',type=int,default=8);parser.add_argument('--steps',type=int,default=600);parser.add_argument('--learning-rate',type=float);parser.add_argument('--text-encoder-learning-rate',type=float,default=0);args=parser.parse_args()
    if args.rank not in (8,16) or args.steps not in (500,600):raise ValueError('Supported experiments: rank 8/16, steps 500/600')
    original=read_job(args.source_job)
    for path in JOBS.glob('*.json'):
        existing=json.loads(path.read_text(encoding='utf-8'))
        if existing['status'] in ('queued','training') or existing.get('base_status')=='generating':raise RuntimeError('GPU already has an active job')
    group='C' if args.learning_rate is not None else ('A' if args.rank==8 else 'B')
    if args.text_encoder_learning_rate:
        if args.text_encoder_learning_rate!=5e-6:raise ValueError('Text encoder experiment LR must be 5e-6')
        group='D'
    job=dict(id=uuid.uuid4().hex,user_id=original['user_id'],username=original['username'],mask_name=original['mask_name']+' experiment '+group,paths=list(original['paths']),photo_ids=original.get('photo_ids',[]),parameters={**original['parameters'],'steps':args.steps,'rank':args.rank,'alpha':args.rank},trigger=original['trigger'],source_job=original['id'],experiment=True,status='training',step=0,error=None)
    job['input_hashes']=[hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in job['paths']]
    if args.text_encoder_learning_rate:job['parameters']['text_encoder_learning_rate']=args.text_encoder_learning_rate
    if args.learning_rate is not None:
        if args.learning_rate not in (2e-5,5e-5):raise ValueError('Supported learning rates: 2e-5/5e-5')
        job['parameters']['learning_rate']=args.learning_rate
    save(job);print('EXPERIMENT_JOB',job['id'],flush=True)
    try:train(job)
    except Exception as exc:job.update(status='failed',error=str(exc));save(job);raise
    print('TRAINING_COMPLETE',job['id'],flush=True)

if __name__=='__main__':main()
