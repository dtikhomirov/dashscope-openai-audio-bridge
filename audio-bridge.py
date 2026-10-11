"""OpenAI-compatible audio bridge for Aliyun Model Studio (DashScope Token Plan).

Exposes:
  POST /v1/audio/speech          -> DashScope SpeechSynthesizer (TTS)
  POST /v1/audio/transcriptions  -> DashScope multimodal generation (STT)
  GET  /health                   -> service status

All configuration comes from environment variables (see .env.example).
There are no built-in defaults: the service validates its configuration at
startup and exits with a meaningful log message if anything is missing or
invalid.
"""

import base64
import logging
import os
from pathlib import Path

import requests
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

LOG_FORMAT = "%(asctime)s %(levelname)-8s [audio-bridge] %(message)s"
VALID_LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
log = logging.getLogger("audio-bridge")


def mask_key(key: str) -> str:
    """Never log the full API key - show a prefix and length only."""
    return f"{key[:6]}*** ({len(key)} chars)" if key else "<empty>"


class Settings:
    """Validated runtime configuration. Every value is required via environment."""

    def __init__(self) -> None:
        problems: list[str] = []

        def raw(name: str) -> str:
            value = (os.getenv(name) or "").strip()
            if not value:
                problems.append(f"{name}: required but missing or empty (set it in .env)")
            return value

        def as_int(name: str, lo: int, hi: int) -> int:
            value = raw(name)
            if not value:
                return 0
            try:
                number = int(value)
            except ValueError:
                problems.append(f"{name}: must be an integer, got {value!r}")
                return 0
            if not lo <= number <= hi:
                problems.append(f"{name}: must be between {lo} and {hi}, got {number}")
            return number

        def as_positive_float(name: str) -> float:
            value = raw(name)
            if not value:
                return 0.0
            try:
                number = float(value)
            except ValueError:
                problems.append(f"{name}: must be a number of seconds, got {value!r}")
                return 0.0
            if number <= 0:
                problems.append(f"{name}: must be greater than 0, got {number}")
            return number

        def as_url(name: str) -> str:
            value = raw(name)
            if value and not value.startswith(("http://", "https://")):
                problems.append(f"{name}: must be an http(s) URL, got {value!r}")
            return value

        self.ali_api_key = raw("ALI_API_KEY")
        self.bridge_host = raw("BRIDGE_HOST")
        self.bridge_port = as_int("BRIDGE_PORT", 1, 65535)
        self.ali_tts_url = as_url("ALI_TTS_URL")
        self.ali_stt_url = as_url("ALI_STT_URL")
        self.tts_format = raw("TTS_FORMAT")
        self.tts_sample_rate = as_int("TTS_SAMPLE_RATE", 1000, 192000)
        self.request_timeout = as_positive_float("REQUEST_TIMEOUT")
        self.max_base64_bytes = as_int("MAX_BASE64_BYTES", 1, 100 * 1024 * 1024)

        log_level = raw("LOG_LEVEL").upper()
        if log_level and log_level not in VALID_LOG_LEVELS:
            problems.append(
                f"LOG_LEVEL: must be one of {', '.join(VALID_LOG_LEVELS)}, got {log_level!r}"
            )
        self.log_level = log_level or "INFO"

        if problems:
            log.error("Invalid configuration - the bridge cannot start:")
            for problem in problems:
                log.error("  - %s", problem)
            log.error("Fix the environment (see .env.example) and restart.")
            raise SystemExit(2)


settings = Settings()
log.setLevel(settings.log_level)

app = FastAPI(title="DashScope OpenAI Audio Bridge")

# Extension -> (DashScope format name, MIME type). Static protocol mapping, not configuration.
AUDIO_FORMATS = {
    ".wav": ("wav", "audio/wav"),
    ".mp3": ("mp3", "audio/mpeg"),
    ".aac": ("aac", "audio/aac"),
    ".amr": ("amr", "audio/amr"),
    ".flac": ("flac", "audio/flac"),
    ".m4a": ("m4a", "audio/mp4"),
    ".ogg": ("ogg", "audio/ogg"),
    ".opus": ("opus", "audio/ogg"),
    ".wma": ("wma", "audio/x-ms-wma"),
}


def auth_headers() -> dict:
    return {
        "Authorization": "Bearer " + settings.ali_api_key,
        "Content-Type": "application/json",
    }


def extract_asr_text(output: dict) -> str:
    """Pull recognized text out of the DashScope ASR response (several shapes seen in the wild)."""
    sentences = output.get("sentences")
    if isinstance(sentences, list) and sentences:
        return " ".join(s.get("text", "") for s in sentences).strip()

    sentence = output.get("sentence")
    if isinstance(sentence, dict):
        return sentence.get("text", "").strip()
    if isinstance(sentence, list):
        return " ".join(s.get("text", "") for s in sentence).strip()

    # Chat-completions style fallback
    choices = output.get("choices")
    if choices:
        content = choices[0].get("message", {}).get("content")
        if isinstance(content, list):
            return " ".join(c.get("text", "") for c in content if isinstance(c, dict)).strip()
        if isinstance(content, str):
            return content.strip()

    return ""


@app.get("/health")
async def health():
    log.debug("Health check requested")
    return {
        "status": "ok",
        "api_key": mask_key(settings.ali_api_key),
    }


# ---------------------------------------------------------
# 1. TTS: OpenAI /v1/audio/speech -> DashScope SpeechSynthesizer
# ---------------------------------------------------------
@app.post("/v1/audio/speech")
async def openai_tts(payload: dict):
    text = payload.get("input")
    if not isinstance(text, str) or not text.strip():
        log.warning("TTS rejected: request body has no non-empty 'input' string")
        raise HTTPException(status_code=400, detail="'input' text is required")

    # OpenAI spec: 'model' and 'voice' are required fields of POST /v1/audio/speech
    model = payload.get("model")
    if not isinstance(model, str) or not model.strip():
        log.warning("TTS rejected: request body has no non-empty 'model' string")
        raise HTTPException(status_code=400, detail="'model' is required")
    model = model.strip()

    voice = payload.get("voice")
    if not isinstance(voice, str) or not voice.strip():
        log.warning("TTS rejected: request body has no non-empty 'voice' string")
        raise HTTPException(status_code=400, detail="'voice' is required")
    voice = voice.strip()
    fmt = payload.get("response_format") or settings.tts_format
    try:
        sample_rate = int(payload.get("sample_rate") or settings.tts_sample_rate)
    except (TypeError, ValueError):
        log.warning("TTS rejected: invalid sample_rate %r in request", payload.get("sample_rate"))
        raise HTTPException(status_code=400, detail="sample_rate must be an integer")

    log.info(
        "TTS request: model=%s voice=%s format=%s sample_rate=%s text_chars=%d",
        model, voice, fmt, sample_rate, len(text),
    )

    # qwen-audio-3.0-tts takes voice/format/sample_rate inside "input"
    ali_payload = {
        "model": model,
        "input": {
            "text": text,
            "voice": voice,
            "format": fmt,
            "sample_rate": sample_rate,
        },
    }

    try:
        resp = requests.post(
            settings.ali_tts_url, json=ali_payload, headers=auth_headers(),
            timeout=settings.request_timeout,
        )
    except requests.RequestException as exc:
        log.error("TTS upstream unreachable (%s): %s", settings.ali_tts_url, exc)
        raise HTTPException(status_code=502, detail=f"Upstream TTS request failed: {exc}")

    if resp.status_code != 200:
        log.error("TTS upstream returned %s: %s", resp.status_code, resp.text[:500])
        raise HTTPException(status_code=resp.status_code, detail=resp.text)

    data = resp.json()
    audio = data.get("output", data).get("audio") or {}

    if audio.get("data"):
        content = base64.b64decode(audio["data"])
    elif audio.get("url"):
        # Non-streaming mode returns a download URL (valid 24h)
        try:
            dl = requests.get(audio["url"], timeout=settings.request_timeout)
        except requests.RequestException as exc:
            log.error("TTS audio download failed: %s", exc)
            raise HTTPException(status_code=502, detail=f"Failed to download synthesized audio: {exc}")
        if dl.status_code != 200:
            log.error("TTS audio download returned %s", dl.status_code)
            raise HTTPException(
                status_code=502, detail=f"Failed to download synthesized audio: {dl.status_code}"
            )
        content = dl.content
    else:
        log.error("TTS upstream response contained no audio: %s", str(data)[:500])
        raise HTTPException(status_code=502, detail=f"No audio in DashScope response: {data}")

    media_type = "audio/mpeg" if fmt == "mp3" else f"audio/{fmt}"
    log.info("TTS success: returned %d bytes of %s", len(content), media_type)
    return Response(content=content, media_type=media_type)


# ---------------------------------------------------------
# 2. STT: OpenAI /v1/audio/transcriptions -> DashScope ASR (sync, base64 input)
# ---------------------------------------------------------
@app.post("/v1/audio/transcriptions")
async def openai_transcriptions(
    file: UploadFile = File(...),
    model: str = Form(None),
    language: str = Form(None),
):
    # OpenAI spec: 'model' is a required field of POST /v1/audio/transcriptions
    if not model or not model.strip():
        log.warning("STT rejected: multipart form has no non-empty 'model' field")
        raise HTTPException(status_code=400, detail="'model' form field is required")
    model = model.strip()

    content = await file.read()
    if not content:
        log.warning("STT rejected: uploaded file %r is empty", file.filename)
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    ext = Path(file.filename or "").suffix.lower()
    fmt, mime = AUDIO_FORMATS.get(ext, ("wav", "audio/wav"))
    if ext not in AUDIO_FORMATS:
        log.warning(
            "STT: unknown extension %r in filename %r - sending to ASR as wav", ext, file.filename
        )

    b64 = base64.b64encode(content).decode()
    if len(b64) > settings.max_base64_bytes:
        log.warning(
            "STT rejected: file %r encodes to %d base64 bytes, limit is %d",
            file.filename, len(b64), settings.max_base64_bytes,
        )
        raise HTTPException(
            status_code=413,
            detail=f"Audio too large: base64 size {len(b64)} exceeds limit {settings.max_base64_bytes}",
        )

    parameters = {"format": fmt}
    if language:
        parameters["language_hints"] = [language]

    log.info(
        "STT request: model=%s file=%s bytes=%d format=%s language=%s",
        model, file.filename, len(content), fmt, language or "auto",
    )

    ali_payload = {
        "model": model,
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_audio", "input_audio": {"data": f"data:{mime};base64,{b64}"}}
                    ],
                }
            ]
        },
        "parameters": parameters,
    }

    headers = auth_headers()
    headers["X-DashScope-SSE"] = "disable"

    try:
        resp = requests.post(
            settings.ali_stt_url, json=ali_payload, headers=headers,
            timeout=settings.request_timeout,
        )
    except requests.RequestException as exc:
        log.error("STT upstream unreachable (%s): %s", settings.ali_stt_url, exc)
        raise HTTPException(status_code=502, detail=f"Upstream STT request failed: {exc}")

    if resp.status_code != 200:
        log.error("STT upstream returned %s: %s", resp.status_code, resp.text[:500])
        raise HTTPException(status_code=resp.status_code, detail=resp.text)

    data = resp.json()
    output = data.get("output", data)
    text = extract_asr_text(output)
    if not text:
        log.warning(
            "STT: upstream returned no recognizable text (file=%s, language=%s). "
            "Audio may be silent or unintelligible.", file.filename, language or "auto",
        )
    else:
        log.info("STT success: recognized %d chars", len(text))
        log.debug("STT text: %s", text)
    return {"text": text}


if __name__ == "__main__":
    import uvicorn

    log.info(
        "Starting bridge on %s:%s | TTS defaults: %s @ %s Hz | key %s",
        settings.bridge_host, settings.bridge_port,
        settings.tts_format, settings.tts_sample_rate,
        mask_key(settings.ali_api_key),
    )
    uvicorn.run(app, host=settings.bridge_host, port=settings.bridge_port)
