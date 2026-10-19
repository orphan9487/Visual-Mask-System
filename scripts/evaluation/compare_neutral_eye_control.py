"""Run with external/LivePortrait/.venv Python; experiment-only eye delta control."""
import json
import os
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
LP=ROOT/'external/LivePortrait'
sys.path.insert(0,str(LP))
os.chdir(LP)
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')


def main():
    import numpy as np
    import torch
    from PIL import Image,ImageDraw,ImageOps
    from src.config.argument_config import ArgumentConfig
    from src.config.inference_config import InferenceConfig
    from src.config.crop_config import CropConfig
    from src.live_portrait_pipeline import LivePortraitPipeline
    from src.utils.retargeting_utils import calc_eye_close_ratio
    out=ROOT/'output/current/neutral_eye_control'
    out.mkdir(parents=True,exist_ok=True)
    sources=[Path(r'C:\Users\User\OneDrive\桌面\Li_train\PXL_20251222_171751462.jpg'),Path(r'C:\Users\User\OneDrive\桌面\henry_train\PXL_20260527_083654253.RAW-01.jpg')]
    args=ArgumentConfig(driving=str(ROOT/'resources/henry_liveportrait_v1/templates/neutral.pkl'),animation_region='lip',flag_relative_motion=False,flag_normalize_lip=False,driving_multiplier=1.,flag_do_torch_compile=False)
    cfg=InferenceConfig(**{k:v for k,v in vars(args).items() if hasattr(InferenceConfig,k)})
    crop_cfg=CropConfig(**{k:v for k,v in vars(args).items() if hasattr(CropConfig,k)})
    pipeline=LivePortraitPipeline(inference_cfg=cfg,crop_cfg=crop_cfg)
    wrapper=pipeline.live_portrait_wrapper
    original_crop=pipeline.cropper.crop_source_image
    original_stitch=wrapper.stitching
    state={}
    def capture_crop(*a,**kw):
        result=original_crop(*a,**kw)
        if result is not None:state['landmarks']=result['lmk_crop']
        return result
    pipeline.cropper.crop_source_image=capture_crop
    def adjusted_stitch(source,driving):
        amount=state['amount']
        if amount:
            lmk=state['landmarks']
            ratios=calc_eye_close_ratio(lmk[None])
            mean=float(ratios.mean())
            # The pretrained eye network accepts two source ratios but just ONE
            # target scalar. This is not independent left/right eye correction.
            baseline=wrapper.calc_combined_eye_ratio([[mean]],lmk)
            target=wrapper.calc_combined_eye_ratio([[min(mean+amount,.4)]],lmk)
            with torch.no_grad():
                correction=wrapper.retarget_eye(source,target)-wrapper.retarget_eye(source,baseline)
            driving=driving+.5*correction
            state['source_ratios']=ratios.tolist()
        return original_stitch(source,driving)
    wrapper.stitching=adjusted_stitch
    records=[]
    sheet=Image.new('RGB',(1200,850),'#fff3e3');draw=ImageDraw.Draw(sheet)
    for row,source in enumerate(sources):
        staged=out/f'person{row+1}.png'
        with Image.open(source) as im:ImageOps.exif_transpose(im).convert('RGB').save(staged)
        for col,amount in enumerate((0.,.02,.04)):
            folder=out/f'person{row+1}_eye{amount:.2f}';folder.mkdir(exist_ok=True)
            args.source=str(staged);args.output_dir=str(folder)
            state.clear();state['amount']=amount
            pipeline.execute(args)
            result=folder/f'{staged.stem}--neutral.jpg'
            with Image.open(result) as im:
                thumb=ImageOps.contain(im.convert('RGB'),(400,395));sheet.paste(thumb,(col*400+(400-thumb.width)//2,row*425+30))
            draw.text((col*400+10,row*425+10),f'P{row+1} lip only | eye increment {amount} blend .5',fill='black')
            records.append({'person':row+1,'increment':amount,'blend':.5,'source_eye_ratios':state.get('source_ratios'),'output':str(result)})
            print('DONE',row+1,amount,flush=True)
    sheet.save(out/'comparison.png')
    (out/'report.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    print('REPORT',out,flush=True)


if __name__=='__main__':main()
