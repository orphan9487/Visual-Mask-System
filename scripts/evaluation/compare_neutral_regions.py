"""Isolated full-expression vs lips-only neutralization; no profile writes."""
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def main():
    from PIL import Image, ImageDraw, ImageOps
    from src.integrations.liveportrait_expression import generate_from_expression_profile
    out=ROOT/'output/current/neutral_region_comparison'
    out.mkdir(parents=True,exist_ok=True)
    sources=[Path(r'C:\Users\User\OneDrive\桌面\Li_train\PXL_20251222_171751462.jpg'),
             Path(r'C:\Users\User\OneDrive\桌面\henry_train\PXL_20260527_083654253.RAW-01.jpg')]
    run=subprocess.run
    records=[]
    sheet=Image.new('RGB',(1200,850),'#fff3e3')
    draw=ImageDraw.Draw(sheet)
    for row,source in enumerate(sources):
        images=[source]
        for region in ('exp','lip'):
            def region_run(command,*args,**kwargs):
                command=list(command)
                command[command.index('--animation-region')+1]=region
                return run(command,*args,**kwargs)
            with patch('src.integrations.liveportrait_expression.subprocess.run',side_effect=region_run):
                result=generate_from_expression_profile({'emotion':'neutral','identity':{'source_override':str(source),'neutralize_source':True}})
            path=out/f'person{row+1}_{region}.png'
            with Image.open(result) as image:image.convert('RGB').save(path)
            images.append(path)
            records.append({'source':str(source),'region':region,'output':str(path)})
            print('DONE',row+1,region,flush=True)
        for col,(label,path) in enumerate(zip(('Original','Absolute full expression','Absolute lips only'),images)):
            with Image.open(path) as image:
                # Display full oriented image, with no stretching or new face crop.
                thumb=ImageOps.contain(ImageOps.exif_transpose(image).convert('RGB'),(400,395))
                sheet.paste(thumb,(400*col+(400-thumb.width)//2,425*row+30))
            draw.text((400*col+10,425*row+10),f'Person {row+1}: {label}',fill='black')
    sheet.save(out/'comparison.png')
    (out/'report.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    print('REPORT',out,flush=True)


if __name__=='__main__':main()
