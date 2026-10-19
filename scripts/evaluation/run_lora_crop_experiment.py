"""Compare normalized head framing against existing inputs; no production writes."""
import hashlib
import json
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def head_box(bbox):
    x1, y1, x2, y2 = map(float, bbox)
    side = max(x2-x1, y2-y1)*1.5
    cx, cy = (x1+x2)/2, (y1+y2)/2 - (y2-y1)*.08
    return tuple(round(v) for v in (cx-side/2, cy-side/2, cx+side/2, cy+side/2))


def main():
    import argparse
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw, ImageOps
    from insightface.app import FaceAnalysis
    from src.services.mask_generation_settings import read_job
    from src.services.lora_training import JOBS, save
    from scripts.training.train_user_lora import train

    parser = argparse.ArgumentParser()
    parser.add_argument('--source-job', required=True)
    args = parser.parse_args()
    original = read_job(args.source_job)
    for p in JOBS.glob('*.json'):
        j = json.loads(p.read_text(encoding='utf-8'))
        if j['status'] in ('queued', 'training') or j.get('base_status') == 'generating':
            raise RuntimeError('GPU already has an active job')
    job_id = uuid.uuid4().hex
    folder = ROOT/'output/current/lora_crop_experiment'/job_id
    folder.mkdir(parents=True)
    analyzer = FaceAnalysis(name='buffalo_l', root=str(ROOT/'models/insightface'), providers=['CPUExecutionProvider'])
    analyzer.prepare(ctx_id=-1, det_size=(640,640))
    paths, audit = [], []
    sheet = Image.new('RGB', (1024, 550*len(original['paths'])), '#fff3e3')
    draw = ImageDraw.Draw(sheet)
    for i, source in enumerate(original['paths']):
        with Image.open(source) as raw:
            im = ImageOps.exif_transpose(raw).convert('RGB')
        bgr = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
        faces = analyzer.get(bgr)
        offset = 0
        if not faces:
            offset = int(max(im.size)*.25)
            faces = analyzer.get(cv2.copyMakeBorder(bgr,offset,offset,offset,offset,cv2.BORDER_CONSTANT,value=(127,127,127)))
        if len(faces) != 1:
            raise ValueError(f'Expected exactly one face: {source}')
        bbox = [float(v)-offset for v in faces[0].bbox]
        box = head_box(bbox)
        # Extend missing edge pixels with neutral gray; never stretch the face.
        canvas = Image.new('RGB', (box[2]-box[0],box[3]-box[1]), (127,127,127))
        canvas.paste(im, (-box[0],-box[1]))
        cropped = canvas.resize((512,512), Image.Resampling.LANCZOS)
        path = folder/f'photo-{i}.png'
        cropped.save(path)
        paths.append(str(path))
        audit.append(dict(source=source, source_sha256=hashlib.sha256(Path(source).read_bytes()).hexdigest(),
                          face_bbox=bbox, crop_box=box, path=str(path)))
        sheet.paste(ImageOps.fit(im,(512,512)),(0,i*550+38))
        sheet.paste(cropped,(512,i*550+38))
        draw.text((12,i*550+12),f'Original training input {i+1}',fill='black')
        draw.text((524,i*550+12),f'Normalized head framing {i+1}',fill='black')
    sheet.save(folder/'preprocessing_comparison.png')
    job = dict(id=job_id,user_id=original['user_id'],username=original['username'],
               mask_name=original['mask_name']+' experiment E framing', paths=paths,
               photo_ids=original.get('photo_ids',[]),
               parameters={**original['parameters'],'steps':500,'rank':8,'alpha':8,
                           'learning_rate':5e-5,'text_encoder_learning_rate':0},
               trigger=original['trigger'],source_job=original['id'],experiment=True,
               preprocessing='normalized_head_1.5_shift_up_0.08_gray_padding',
               preprocessing_audit=audit,status='training',step=0,error=None)
    job['input_hashes']=[hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths]
    save(job)
    print('EXPERIMENT_JOB',job_id,flush=True)
    try:
        train(job)
    except Exception as exc:
        job.update(status='failed',error=str(exc));save(job);raise
    print('TRAINING_COMPLETE',job_id,flush=True)


if __name__ == '__main__':
    main()
