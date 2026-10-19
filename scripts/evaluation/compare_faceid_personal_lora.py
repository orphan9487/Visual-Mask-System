"""Offline three-reference FaceID versus FaceID plus personal LoRA."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))

def main():
    import argparse,json
    import torch,cv2,numpy as np
    from PIL import Image,ImageDraw
    from src.services.mask_generation_settings import read_job,checkpoint_path
    from src.generation.renderers.faceid_sticker_renderer import FaceIDStickerRenderer
    parser=argparse.ArgumentParser();parser.add_argument('--job',required=True);args=parser.parse_args()
    job=read_job(args.job);renderer=FaceIDStickerRenderer();analyzer=renderer._analyzer();embeds=[]
    for p in job['paths']:
        im=cv2.imdecode(np.fromfile(p,dtype=np.uint8),1);faces=analyzer.get(im)
        if not faces:
            pad=int(max(im.shape[:2])*.25);faces=analyzer.get(cv2.copyMakeBorder(im,pad,pad,pad,pad,cv2.BORDER_CONSTANT,value=(127,127,127)))
        if len(faces)!=1:raise RuntimeError('Expected one face in '+p)
        embeds.append(torch.from_numpy(faces[0].normed_embedding).reshape(1,1,-1))
    adapter=renderer._adapter_for(ROOT/'models/ip_adapter_faceid/ip-adapter-faceid-portrait-v11_sd15.bin',len(embeds))
    embedding=torch.cat(embeds,dim=1)
    # Explicit appearance cues are held constant for both cases.
    prompt=f"{job['trigger']}, young adult man, short black hair, glasses, front facing neutral portrait, full head visible, plain light studio background, realistic photo"
    negative='long hair, curly hair, side view, cropped head, multiple faces, deformed, blurry'
    out=ROOT/'output/current/faceid_lora_comparison'/job['id'];out.mkdir(parents=True,exist_ok=True);records=[]
    for scale in (0,.7):
        if scale:
            weight=Path(checkpoint_path(job,300));adapter.pipe.load_lora_weights(str(weight.parent),weight_name=weight.name,adapter_name='personal');adapter.pipe.set_adapters(['personal'],adapter_weights=[scale])
        img=adapter.generate(prompt=prompt,negative_prompt=negative,faceid_embeds=embedding,scale=.55,num_samples=1,width=512,height=512,num_inference_steps=30,guidance_scale=7.,seed=42)[0]
        path=out/f'lora_{scale}.png';img.save(path);records.append(dict(scale=scale,path=str(path)));print('SAVED',path,flush=True)
    sheet=Image.new('RGB',(1024,550),'#fff3e3');draw=ImageDraw.Draw(sheet)
    for i,item in enumerate(records):
        sheet.paste(Image.open(item['path']),(i*512,38));draw.text((i*512+12,12),'FaceID 0.55 + LoRA '+str(item['scale']),fill='black')
    sheet.save(out/'comparison.png');(out/'settings.json').write_text(json.dumps(dict(prompt=prompt,negative=negative,seed=42,faceid_scale=.55,results=records),indent=2),encoding='utf-8');print('DONE',out/'comparison.png',flush=True)

if __name__=='__main__':main()
