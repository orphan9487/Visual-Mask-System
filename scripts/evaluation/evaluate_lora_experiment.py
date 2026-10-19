"""Six checkpoints, three fixed seeds; no FaceID conditioning or prompt tuning."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))

def main():
    import torch
    from PIL import Image,ImageDraw
    from diffusers import StableDiffusionPipeline
    from src.services.mask_generation_settings import read_job,checkpoint_path
    from src.generation.diffusion_engine import SD_MODEL_ID
    parser=argparse.ArgumentParser();parser.add_argument('--job',required=True);parser.add_argument('--compare-job');args=parser.parse_args()
    job=read_job(args.job);out=ROOT/'output/current'/('lora_rank_comparison' if args.compare_job else 'lora_experiment_a')/job['id'];out.mkdir(parents=True,exist_ok=True)
    pipe=StableDiffusionPipeline.from_pretrained(SD_MODEL_ID,local_files_only=True,torch_dtype=torch.float16).to('cuda')
    prompt=f"photo of {job['trigger']} person, front facing centered portrait, neutral expression, relaxed closed mouth, full head visible, plain light studio background, natural skin, soft light"
    negative='side view, tilted head, cropped head, multiple people, busy background, text, blurry, deformed'
    seeds=(42,123,314159);steps=(100,200,300,400,500,600);records=[]
    cases=[(job,step) for step in steps]
    if args.compare_job:
        baseline=read_job(args.compare_job)
        same_inputs=baseline['paths']==job['paths']
        if job.get('preprocessing_audit'):
            import hashlib
            audit=job['preprocessing_audit']
            same_inputs=([a['source'] for a in audit]==baseline['paths'] and all(hashlib.sha256(Path(a['source']).read_bytes()).hexdigest()==a['source_sha256'] for a in audit))
        if baseline['trigger']!=job['trigger'] or not same_inputs:raise ValueError('Comparison inputs differ')
        cases=[(baseline,500),(job,500)]
    sheet=Image.new('RGB',(512*len(cases),550*3),'#fff3e3');draw=ImageDraw.Draw(sheet)
    for col,(case,step) in enumerate(cases):
        pipe.unload_lora_weights();weight=Path(checkpoint_path(case,step))
        pipe.load_lora_weights(str(weight.parent),weight_name=weight.name,adapter_name='experiment');pipe.set_adapters(['experiment'],adapter_weights=[1.])
        for row,seed in enumerate(seeds):
            result=pipe(prompt=prompt,negative_prompt=negative,width=512,height=512,num_inference_steps=30,guidance_scale=7,generator=torch.Generator('cuda').manual_seed(seed))
            rejected=any(result.nsfw_content_detected or [])
            rank=case['parameters']['rank']
            lr=case['parameters']['learning_rate']
            text_lr=case['parameters'].get('text_encoder_learning_rate',0)
            path=out/f'{case["id"]}_step_{step}_seed_{seed}.png';result.images[0].save(path)
            framing='GRAY BG' if case.get('preprocessing')=='u2net_gray_background' else ('HEAD CROP' if case.get('preprocessing_audit') else 'ORIGINAL')
            sheet.paste(result.images[0],(512*col,550*row+38));draw.text((512*col+12,550*row+12),f'{framing} r{rank} LR{lr} TE{text_lr} | {step} | seed {seed}'+(' | REJECTED' if rejected else ''),fill='black')
            records.append(dict(job_id=case['id'],rank=rank,learning_rate=lr,text_encoder_learning_rate=text_lr,step=step,seed=seed,path=str(path),rejected=rejected));print('DONE',rank,lr,text_lr,step,seed,flush=True)
    sheet.save(out/'comparison.png')
    del pipe;torch.cuda.empty_cache()
    import cv2,numpy as np
    from insightface.app import FaceAnalysis
    analyzer=FaceAnalysis(name='buffalo_l',root=str(ROOT/'models/insightface'),providers=['CPUExecutionProvider']);analyzer.prepare(ctx_id=-1,det_size=(640,640))
    def embedding(path):
        image=cv2.imdecode(np.fromfile(path,dtype=np.uint8),1);faces=analyzer.get(image)
        if not faces:
            pad=int(max(image.shape[:2])*.25);faces=analyzer.get(cv2.copyMakeBorder(image,pad,pad,pad,pad,cv2.BORDER_CONSTANT,value=(127,127,127)))
        if len(faces)!=1:return None
        return faces[0].normed_embedding
    reference_paths=job.get('reference_paths') or ([a['source'] for a in job['preprocessing_audit']] if job.get('preprocessing_audit') else job['paths'])
    references=[embedding(p) for p in reference_paths];references=[e for e in references if e is not None]
    for record in records:
        e=None if record['rejected'] else embedding(record['path'])
        record['cosine_to_references']=None if e is None else [float(np.dot(e,r)) for r in references]
        record['mean_cosine']=None if e is None or not references else float(np.mean(record['cosine_to_references']))
    summary={f"{case['id']}_rank{case['parameters']['rank']}_lr{case['parameters']['learning_rate']}_textlr{case['parameters'].get('text_encoder_learning_rate',0)}_step{step}":[r['mean_cosine'] for r in records if r['step']==step and r['job_id']==case['id']] for case,step in cases}
    (out/'report.json').write_text(json.dumps(dict(prompt=prompt,negative=negative,lora_weight=1.,source_job=job.get('source_job'),records=records,summary=summary),indent=2),encoding='utf-8')
    print('SUMMARY',json.dumps(summary),flush=True);print('REPORT',out,flush=True)

if __name__=='__main__':main()
