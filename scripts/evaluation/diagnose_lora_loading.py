"""Controlled local diagnostic; never changes active masks or database records."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    import torch
    from PIL import Image, ImageDraw
    from diffusers import StableDiffusionPipeline
    from safetensors.torch import load_file
    from src.services.mask_generation_settings import read_job, checkpoint_path
    from src.generation.diffusion_engine import SD_MODEL_ID

    parser = argparse.ArgumentParser()
    parser.add_argument('--job', required=True)
    args = parser.parse_args()
    job = read_job(args.job)
    out = ROOT / 'output/current/lora_loading_diagnostic' / job['id']
    out.mkdir(parents=True, exist_ok=True)
    prompts = {
        'training_caption': f"photo of {job['trigger']} person",
        'studio': f"photo of {job['trigger']} person, front facing centered portrait, neutral expression, full head visible, plain light studio background",
    }
    seeds = (42, 123)
    cases = [(None, 0)] + [(step, weight) for step in (300, 500) for weight in (.5, 1., 1.5)]
    pipe = StableDiffusionPipeline.from_pretrained(
        SD_MODEL_ID, local_files_only=True, torch_dtype=torch.float16).to('cuda')
    pipe.set_progress_bar_config(disable=True)
    records, audits = [], []
    for column, (step, weight) in enumerate(cases):
        pipe.unload_lora_weights()
        if step is not None:
            path = Path(checkpoint_path(job, step)).resolve()
            state = load_file(str(path))
            assert state and all(torch.isfinite(t).all() for t in state.values())
            up = [v for k, v in state.items() if '.up.' in k or 'lora_B' in k]
            assert up and any(torch.count_nonzero(t).item() for t in up), 'Empty LoRA updates'
            pipe.load_lora_weights(str(path.parent), weight_name=path.name, adapter_name='diagnostic')
            pipe.set_adapters(['diagnostic'], adapter_weights=[weight])
            loaded = {k: v.detach().cpu().float() for k, v in pipe.unet.named_parameters() if 'lora_' in k}
            assert loaded and any('lora_B' in k and torch.count_nonzero(v).item() for k, v in loaded.items())
            audits.append(dict(step=step, weight=weight, path=str(path),
                               sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                               saved_tensor_count=len(state), loaded_unet_tensor_count=len(loaded),
                               active_adapters=pipe.get_active_adapters()))
        for prompt_name, prompt in prompts.items():
            for seed in seeds:
                result = pipe(prompt=prompt, width=512, height=512, num_inference_steps=30,
                              guidance_scale=7, generator=torch.Generator('cuda').manual_seed(seed))
                name = f'{prompt_name}_seed{seed}_step{step or 0}_weight{weight}.png'
                result.images[0].save(out / name)
                records.append(dict(prompt_name=prompt_name, seed=seed, step=step, weight=weight,
                                    column=column, path=str(out / name),
                                    rejected=any(result.nsfw_content_detected or [])))
        print('CASE_DONE', step, weight, flush=True)
    del pipe
    torch.cuda.empty_cache()
    import cv2
    import numpy as np
    from insightface.app import FaceAnalysis
    analyzer = FaceAnalysis(name='buffalo_l', root=str(ROOT/'models/insightface'), providers=['CPUExecutionProvider'])
    analyzer.prepare(ctx_id=-1, det_size=(640, 640))

    def embed(path):
        im = cv2.imdecode(np.fromfile(path, dtype=np.uint8), 1)
        faces = analyzer.get(im)
        if not faces:
            pad = int(max(im.shape[:2])*.25)
            faces = analyzer.get(cv2.copyMakeBorder(im, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(127,127,127)))
        return faces[0].normed_embedding if len(faces) == 1 else None

    refs = [embed(p) for p in job['paths']]
    reference_audit = [dict(path=p, detected=e is not None) for p, e in zip(job['paths'], refs)]
    refs = [e for e in refs if e is not None]
    for r in records:
        e = None if r['rejected'] else embed(r['path'])
        r['mean_cosine'] = float(np.mean([np.dot(e, ref) for ref in refs])) if e is not None and refs else None
        baseline = next(x for x in records if x['step'] is None and x['seed'] == r['seed'] and x['prompt_name'] == r['prompt_name'])
        a = np.asarray(Image.open(r['path'])).astype(float)
        b = np.asarray(Image.open(baseline['path'])).astype(float)
        r['pixel_mae_vs_no_lora'] = float(np.abs(a-b).mean())
    for prompt_name in prompts:
        sheet = Image.new('RGB', (320*len(cases), 350*len(seeds)), '#fff3e3')
        draw = ImageDraw.Draw(sheet)
        for r in records:
            if r['prompt_name'] != prompt_name:
                continue
            x, y = r['column']*320, seeds.index(r['seed'])*350
            with Image.open(r['path']) as im:
                sheet.paste(im.resize((320,320)), (x,y+30))
            draw.text((x+5,y+7), f"step {r['step'] or 0} w{r['weight']} seed{r['seed']}", fill='black')
        sheet.save(out / f'{prompt_name}_comparison.png')
    report = dict(job_id=job['id'], model=SD_MODEL_ID, prompts=prompts, negative_prompt='',
                  guidance_scale=7, inference_steps=30, references=reference_audit, audits=audits, records=records,
                  caveat='Training-reference cosine is not held-out accuracy. Pixel difference proves effect, not identity quality.')
    (out/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('REPORT', out, flush=True)


if __name__ == '__main__':
    main()
