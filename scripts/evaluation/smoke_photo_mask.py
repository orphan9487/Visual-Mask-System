"""Real LivePortrait smoke test in isolated temporary mask storage, no database."""
import argparse
import asyncio
import io
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def main():
    from PIL import Image,ImageOps
    from src.services import portrait_masks
    from src.services.pipeline_service import PipelineService,PipelineAnalysis
    from src.reasoning.identity_db import identity_db
    parser=argparse.ArgumentParser();parser.add_argument('--source',required=True);args=parser.parse_args()
    with Image.open(args.source) as raw:im=ImageOps.exif_transpose(raw).convert('RGB')
    stream=io.BytesIO();im.save(stream,format='PNG')
    side=min(im.size);crop=json.dumps(dict(x=(im.width-side)//2,y=(im.height-side)//2,width=side,height=side))
    (ROOT/'tmp').mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as directory,patch.object(portrait_masks,'STORE',Path(directory)/'masks'):
        result=portrait_masks.create(4,'TEST','Isolated square smoke test',stream.getvalue(),'image/png',crop,'natural')
        portrait_masks.confirm(result['key'],4,True)
        identity=identity_db.resolve(user_id='TEST',mask_id=result['key'])
        service=PipelineService()
        for emotion in ('neutral','sadness'):
            analysis=PipelineAnalysis({'emotion':emotion,'instruction':{'emotion':emotion,'identity':{'mask_id':identity.mask_id},'intensity':.5}},emotion,[],'TEST')
            output=asyncio.run(service.generate(analysis))
            with Image.open(output.output_path) as image:
                assert image.width==image.height,image.size
                print('PASS',emotion,image.size,output.output_path,flush=True)


if __name__=='__main__':main()
