"""Same prompt and noise, with and without the personal LoRA."""
import argparse
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))

def main():
    import torch
    from diffusers import StableDiffusionPipeline
    from src.services.mask_generation_settings import read_job,checkpoint_path
    from src.generation.diffusion_engine import SD_MODEL_ID
    parser=argparse.ArgumentParser();parser.add_argument('--job',required=True);parser.add_argument('--checkpoints',action='store_true');parser.add_argument('--appearance',action='store_true');args=parser.parse_args()
    job=read_job(args.job)
    pipe=StableDiffusionPipeline.from_pretrained(SD_MODEL_ID,torch_dtype=torch.float16,local_files_only=True).to('cuda')
    prompt=f"photo of {job['trigger']} person, front facing centered portrait, neutral expression, relaxed closed mouth, full head visible, plain light studio background, natural skin, soft light"
    if args.appearance:
        prompt=f"photo of {job['trigger']} person, young adult man, short straight black hair, rectangular glasses, light stubble, front facing neutral portrait, closed mouth, full head visible, plain light studio background, natural skin"
    negative='side view, tilted head, cropped head, multiple people, busy background, text, blurry, deformed'
    out=ROOT/'output'/'current'/('lora_appearance' if args.appearance else ('lora_checkpoints' if args.checkpoints else 'lora_effect'))/job['id'];out.mkdir(parents=True,exist_ok=True)
    weight=Path(checkpoint_path(job,300));records=[]
    cases=[(step,1.) for step in (100,200,300)] if args.checkpoints else [(300,s) for s in (0,.7,1.)]
    if args.appearance:cases=[(0,0),(100,1.),(200,1.),(300,1.)]
    for step,scale in cases:
        if scale and (args.appearance or args.checkpoints or scale==.7):
            if args.checkpoints or args.appearance:pipe.unload_lora_weights()
            weight=Path(checkpoint_path(job,step))
            pipe.load_lora_weights(str(weight.parent),weight_name=weight.name,adapter_name='test_person')
            print('ADAPTERS',pipe.get_active_adapters(),flush=True)
        if scale:pipe.set_adapters(['test_person'],adapter_weights=[scale])
        result=pipe(prompt=prompt,negative_prompt=negative,width=512,height=512,num_inference_steps=30,guidance_scale=7,generator=torch.Generator('cuda').manual_seed(42))
        if any(result.nsfw_content_detected or []):raise RuntimeError('Image rejected by safety checker')
        path=out/(f'steps_{step}.png' if args.checkpoints or args.appearance else f'scale_{scale}.png');result.images[0].save(path);records.append(dict(step=step,scale=scale,path=str(path)));print('SAVED',path,flush=True)
    from PIL import Image,ImageDraw
    sheet=Image.new('RGB',(512*len(records),550),'#fff3e3');draw=ImageDraw.Draw(sheet)
    for i,r in enumerate(records):
        sheet.paste(Image.open(r['path']),(512*i,38));draw.text((512*i+12,12),(str(r['step'])+' steps / ' if args.checkpoints or args.appearance else '')+'LoRA '+str(r['scale']),fill='black')
    sheet.save(out/'comparison.png')
    (out/'settings.json').write_text(json.dumps(dict(seed=42,prompt=prompt,negative_prompt=negative,checkpoint=str(weight),results=records),indent=2),encoding='utf-8')
    print('COMPARISON',out/'comparison.png',flush=True)

if __name__=='__main__':main()
