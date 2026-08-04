# Talking Face black-box adapter

Visual Mask System does not import or modify the partner team's Wav2Lip code.
The integration boundary is an HTTP service so either team can replace its
implementation without changing the emotion, LoRA, WebSocket, or LINE layers.

## Service contract

The external service must expose:

- `GET /health`: return HTTP 200 when its models are loaded and ready.
- `POST /generate`: accept `multipart/form-data` fields `face` and `audio`.
- `face`: a PNG or JPEG containing one detectable face.
- `audio`: WAV, MP3, M4A, or MP4 speech audio. The current partner code resamples
  audio to 16 kHz before Wav2Lip inference.
- Successful response: HTTP 200, `Content-Type: video/mp4`, body containing MP4
  bytes.
- Failed response: non-200 response with a short diagnostic message.
- Optional authentication: `Authorization: Bearer <token>`.

Configure the adapter in `.env`:

```dotenv
VMS_TALKING_FACE_URL=http://127.0.0.1:8020
VMS_TALKING_FACE_TOKEN=
VMS_TALKING_FACE_TIMEOUT=300
VMS_TALKING_FACE_MAX_OUTPUT_MB=100
```

## Current partner-package gap

`talking-face-linebot--main.zip` contains `process_wav2lip(face_path,
audio_path, final_out_path)`, but its FastAPI app only exposes a LINE callback.
It does not currently expose `/health` or `/generate`, and the archive does not
contain `checkpoints/wav2lip_gan.pth`.

The ZIP alone therefore cannot start. After the two weights are placed in the
paths below, the Sidecar supplies the missing contract without changing any
partner source file.

## Sidecar for the current partner package

`src.interfaces.talking_face_sidecar` provides the missing HTTP contract while
leaving every file in the partner package unchanged. It loads the partner
`api_server.py` in an isolated process, calls its existing `process_wav2lip`,
serializes GPU jobs, and removes per-request input/output files afterward.

The extracted partner directory must contain:

```text
api_server.py
checkpoints/wav2lip_gan.pth
face_detection/detection/sfd/s3fd.pth
```

Set the absolute extracted directory and start the Sidecar with the partner's
own Python environment (not the Visual Mask environment):

```dotenv
VMS_TALKING_FACE_ROOT=C:\path\to\talking-face-linebot--main
VMS_TALKING_FACE_URL=http://127.0.0.1:8020
```

```powershell
python -m uvicorn src.interfaces.talking_face_sidecar:app `
  --host 127.0.0.1 --port 8020
```

Port 8020 avoids conflicts with the main Visual Mask API on port 8000 and the
emotion inference service on port 8010.

The validated Windows setup uses an isolated virtual environment created from
the existing CUDA-enabled `mask_env`, with NumPy 1.23.5, OpenCV 4.8.1,
MediaPipe 0.10.14, and Librosa 0.9.2. The legacy Wav2Lip audio code is not
compatible with current Librosa's keyword-only mel-filter API.

Validated model hashes:

```text
s3fd.pth SHA256
619a31681264d3f7f7fc7a16a42cbbe8b23f31a256f75a366e5a1bcd59b33543

wav2lip_gan.pth SHA256
ca9ab7b7b812c0e80a6e70a5977c545a1e8a365a6c49d5e533023c034d7ac3d8
```

## Intended orchestration

1. Emotion inference produces a `VisualInstruction`.
2. SD1.5 plus `henrymask_v3` produces the face PNG.
3. Edge TTS produces an MP3 with the configured Traditional Chinese voice.
4. `HttpTalkingFaceAdapter.generate()` uploads the PNG and audio.
5. The returned MP4 is stored under `output/current` and can then be pushed to
   LINE with its preview image.

## LINE orchestration

Set `LINE_TALKING_FACE_ENABLED=1` to enable the optional video stage. The LINE
pipeline checks `/health` before calling TTS, so an unavailable Sidecar does not
waste a synthesis request. If TTS or Talking Face fails, LINE falls back to the
already generated Henry PNG instead of failing the whole message.

Emotion inference receives the original text, while TTS receives a sanitized
copy. Unicode emoji, LINE emoji alternative names and sticker labels are removed
before speech synthesis. An emoji-only or sticker-only message falls back to the
generated emotion image instead of speaking the visual label.

```dotenv
LINE_TALKING_FACE_ENABLED=1
VMS_TTS_VOICE=zh-TW-YunJheNeural
VMS_TTS_RATE=+0%
VMS_TTS_VOLUME=+0%
VMS_TTS_PITCH=+0Hz
VMS_TTS_TIMEOUT=60
VMS_TTS_MAX_TEXT_CHARS=500
```
