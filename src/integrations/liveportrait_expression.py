"""Optional LivePortrait expression backend with cache fallback."""
from __future__ import annotations
import json, os, subprocess, time
from pathlib import Path
from PIL import Image
ROOT=Path(__file__).resolve().parents[2]; CONFIG=ROOT/'configs'/'henry_liveportrait_v1.json'; OUT=ROOT/'output'/'current'/'instruction_runs'
# LivePortrait 的暫存區會放使用者原始照片的複本，必須在不對外提供的 data/ 底下，
# 不能放在 output/（/output 是不需登入的靜態目錄）；每次用完即刪。
RUNTIME_ROOT=ROOT/'data'/'liveportrait_runtime'
def _path(v):
    p=Path(v); return p if p.is_absolute() else ROOT/p
def generate_from_expression_profile(instruction:dict)->str:
    profile=json.loads(CONFIG.read_text(encoding='utf-8')); emotion=str(instruction.get('emotion','neutral')).lower(); item=profile.get('profiles',{}).get(emotion)
    if not item: raise RuntimeError(f'LivePortrait profile missing: {emotion}')
    identity = instruction.get('identity', {})
    source_override = identity.get('source_override')
    references = identity.get('faceid_reference_paths', identity.get('reference_paths', []))
    source = _path(source_override) if source_override else (_path(references[0]) if references else _path(profile['source']))
    template=_path(item['template']); OUT.mkdir(parents=True,exist_ok=True)
    import uuid
    root=_path(os.getenv('VMS_LIVEPORTRAIT_ROOT','external/LivePortrait')); python=root/'.venv'/'Scripts'/'python.exe'; inference=root/'inference.py'; runtime=RUNTIME_ROOT/uuid.uuid4().hex; runtime.mkdir(parents=True,exist_ok=True)
    try:
        return _run_in(runtime, source, template, profile, item, identity, emotion, source_override, references, python, inference, root)
    finally:
        import shutil
        shutil.rmtree(runtime, ignore_errors=True)


def _run_in(runtime, source, template, profile, item, identity, emotion, source_override, references, python, inference, root):
    result = None
    if python.is_file() and inference.is_file():
        try:
            # OpenCV's image reader cannot reliably handle Chinese Windows paths.
            # Stage a fresh, oriented image under the ASCII runtime directory.
            from PIL import ImageOps
            with Image.open(source) as original:
                ImageOps.exif_transpose(original).convert('RGB').save(runtime/'source.png')
            source=runtime/'source.png'
            environment=os.environ.copy(); environment['PYTHONUTF8']='1'; environment['PYTHONIOENCODING']='utf-8'
            neutralize=bool(identity.get('neutralize_source'))
            command=[str(python),'inference.py','--source',str(source),'--driving',str(template),'--output-dir',str(runtime),'--animation-region',str(profile.get('animation_region','exp')),'--driving-option',str(profile.get('driving_option','expression-friendly')),'--driving-multiplier',str(1.0 if neutralize else item.get('multiplier',1.0)),'--flag-use-half-precision','--no-flag-do-torch-compile']
            if neutralize:command+=['--no-flag-relative-motion','--no-flag-normalize-lip']
            subprocess.run(command,cwd=root,check=True,timeout=120,env=environment)
            candidate=runtime/f'{source.stem}--{template.stem}.jpg'
            if candidate.is_file(): result=candidate
        except (OSError, subprocess.SubprocessError) as error:
            print(f'[LivePortrait] runtime failed; using cache: {error}', flush=True)
    if result is None and (source_override or references):
        raise RuntimeError('此人物的 LivePortrait 生成失敗，已停止，避免誤用 Henry 快取')
    if result is None: result=_path(item['cache'])
    if not result.is_file(): raise RuntimeError('LivePortrait runtime and cache are unavailable')
    destination=OUT/f'liveportrait_{emotion}_{time.time_ns()}.png'
    with Image.open(result) as rendered:   # 用 with 關檔，暫存區才刪得掉（Windows 會鎖住開著的檔）
        rendered.convert('RGB').save(destination)
    return str(destination)

_sd15_img2img = None
def stylize_with_sd15(image_path: str, emotion: str, strength: float = 0.22, identity=None) -> str:
    """Apply a restrained SD1.5 img2img pass while preserving expression."""
    global _sd15_img2img
    import torch
    from diffusers import AutoencoderKL, DDIMScheduler, StableDiffusionImg2ImgPipeline
    from src.generation.diffusion_engine import SD_MODEL_ID, VAE_DIR
    if _sd15_img2img is None:
        _sd15_img2img = StableDiffusionImg2ImgPipeline.from_pretrained(
            SD_MODEL_ID, torch_dtype=torch.float16, local_files_only=True,
            safety_checker=None, feature_extractor=None, requires_safety_checker=False,
        ).to('cuda')
        if VAE_DIR.is_dir():
            _sd15_img2img.vae = AutoencoderKL.from_pretrained(str(VAE_DIR), torch_dtype=torch.float16, local_files_only=True).to('cuda')
        _sd15_img2img.scheduler = DDIMScheduler.from_config(_sd15_img2img.scheduler.config)
    identity = identity or {}
    _sd15_img2img.unload_lora_weights()
    if identity.get('lora_path'):
        _sd15_img2img.load_lora_weights(str(_path(identity['lora_path']).parent), weight_name=_path(identity['lora_path']).name, adapter_name='personal')
        _sd15_img2img.set_adapters(['personal'],adapter_weights=[float(identity.get('lora_weight',0.7))])
    image = Image.open(image_path).convert('RGB').resize((512, 512))
    result = _sd15_img2img(
        prompt=f"{identity.get('trigger', '')}, realistic portrait, {emotion} facial expression, natural skin",
        negative_prompt='blurry, deformed, changed identity, extra face, cropped face',
        image=image, strength=float(strength), guidance_scale=5.5,
        num_inference_steps=25,
        generator=torch.Generator('cuda').manual_seed(int(identity.get('seed',42))),
    ).images[0]
    destination=OUT/f'hybrid_{emotion}_{time.time_ns()}.png'; result.save(destination); return str(destination)

