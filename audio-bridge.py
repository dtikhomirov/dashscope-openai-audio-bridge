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

import asyncio
import base64
import io
import logging
import math
import os
import re
import shutil
import wave
from array import array
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import httpx
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import Response

LOG_FORMAT = "%(asctime)s %(levelname)-8s [audio-bridge] %(message)s"
VALID_LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
log = logging.getLogger("audio-bridge")


def mask_key(key: str) -> str:
    """Never log the full API key - show a prefix and length only."""
    return f"{key[:6]}*** ({len(key)} chars)" if key else "<empty>"


# Allowed TTS output formats - the set qwen-audio-tts supports (see
# docs/qwen-audio-tts-reference.md) - mapped to the Content-Type returned to
# the client. Validates TTS_FORMAT at startup and response_format per request.
TTS_MEDIA_TYPES = {
    "wav": "audio/wav",
    "mp3": "audio/mpeg",
    "opus": "audio/ogg",
    "pcm": "audio/pcm",
}

# Accepted TTS sample-rate bounds in Hz (env default and per-request override).
TTS_SAMPLE_RATE_RANGE = (1000, 192000)

# Formats delivered via a local decode -> re-encode pipeline (see
# postprocess_compressed_audio). DashScope's own opus/ogg muxer writes a 5:3
# inflated granule clock (RFC 7845 violation), which makes granulepos-based
# players (e.g. Telegram voice notes) drop ~40 ms of audio every 100 ms.
REENCODE_FORMATS = ("opus", "mp3")

# ffmpeg is optional at runtime: without it, compressed formats fall back to
# passing the upstream audio through unchanged (logged as a warning).
FFMPEG = shutil.which("ffmpeg")
FFMPEG_TIMEOUT_S = 30

# httpx.InvalidURL is not a RequestError subclass, but a malformed URL (from
# config or an upstream response) must surface as a 502, not an unhandled 500.
UPSTREAM_HTTP_ERRORS = (httpx.RequestError, httpx.InvalidURL)


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

        def opt_int(name: str, lo: int, hi: int, default: int) -> int:
            """Optional integer with a built-in default (unset/empty -> default)."""
            value = (os.getenv(name) or "").strip()
            if not value:
                return default
            try:
                number = int(value)
            except ValueError:
                problems.append(f"{name}: must be an integer, got {value!r}")
                return default
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
        if self.tts_format == "ogg":
            self.tts_format = "opus"  # colloquial OpenAI-ecosystem alias
        if self.tts_format and self.tts_format not in TTS_MEDIA_TYPES:
            problems.append(
                f"TTS_FORMAT: must be one of {', '.join(sorted(TTS_MEDIA_TYPES))} "
                f"(or 'ogg' as an alias of 'opus'), got {self.tts_format!r}"
            )
        self.tts_sample_rate = as_int("TTS_SAMPLE_RATE", *TTS_SAMPLE_RATE_RANGE)
        # Bitrate (kbps) for locally re-encoded opus output. Default 32 matches
        # DashScope's documented opus bit_rate default (valid range 6-510).
        self.tts_opus_bitrate = opt_int("TTS_OPUS_BITRATE", 6, 510, 32)
        self.request_timeout = as_positive_float("REQUEST_TIMEOUT")
        self.max_base64_bytes = as_int("MAX_BASE64_BYTES", 1, 100 * 1024 * 1024)
        # 0 disables silence smoothing; otherwise internal silences in TTS audio
        # longer than this cap are shortened (all formats; opus/mp3 need ffmpeg).
        self.tts_silence_cap_ms = as_int("TTS_SILENCE_CAP_MS", 0, 10000)

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

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Shared async HTTP client: non-blocking upstream calls + connection pooling.
    # Using async handlers with a sync client (requests) would block the event
    # loop and serialize every request; httpx keeps the bridge concurrent.
    # follow_redirects=True: requests (the previous client) followed redirects
    # by default, and DashScope download URLs may redirect (OSS/CDN, http->https).
    app.state.http = httpx.AsyncClient(
        timeout=settings.request_timeout, follow_redirects=True
    )
    if FFMPEG:
        log.info("ffmpeg found (%s) - opus/mp3 responses are locally re-encoded", FFMPEG)
    else:
        log.warning(
            "ffmpeg not found - opus/mp3 responses pass through unchanged "
            "(no silence smoothing, upstream granule clock kept)"
        )
    try:
        yield
    finally:
        await app.state.http.aclose()


app = FastAPI(title="DashScope OpenAI Audio Bridge", lifespan=lifespan)

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


EMOJI_RE = re.compile("[\U0001F000-\U0001FAFF\u2600-\u27BF\uFE0F\u2B00-\u2BFF\u2190-\u21FF]")
BULLET_RE = re.compile(r"^[ \t]*(?:[-*+\u2022\u00b7>]+|\d+\.)[ \t]+", re.M)
MARKDOWN_RE = re.compile(r"[#*_>`|~]+")
# Numeric ranges like "10-15" or "10–15": spoken as "10 to 15" because the
# synthesizer inserts a comma pause at hyphen/space-grouped digit strings
# (documented in the SSML <say-as> reference). Guards exclude dates (2026-01-15)
# and longer digit strings (phone numbers).
NUM_RANGE_RE = re.compile(r"(?<![\d.\-\u2013\u2014])(\d{1,3})\s*[-\u2013\u2014]\s*(\d{1,3})(?![\d.\-\u2013\u2014])")
TILDE_RANGE_RE = re.compile(r"(?<=\w)\s*~\s*(?=\w)")
UNIT_WORDS = (
    (re.compile(r"\bkm/h\b", re.I), " kilometers per hour"),
    (re.compile(r"\bmph\b", re.I), " miles per hour"),
    (re.compile(r"\bm/s\b"), " meters per second"),
)


def normalize_tts_text(text: str) -> str:
    """Make text speakable: the synthesizer inserts audible pauses or stumbles at
    markdown decorations, bullets, emoji, hard line breaks, dashes and unit symbols
    (common in LLM output like weather forecasts). Strip those, verbalize numeric
    ranges and expand a few unit symbols."""
    text = EMOJI_RE.sub(" ", text)
    text = BULLET_RE.sub(" ", text)
    text = TILDE_RANGE_RE.sub(" to ", text)  # before MARKDOWN_RE strips "~"
    text = NUM_RANGE_RE.sub(r"\1 to \2", text)
    text = MARKDOWN_RE.sub(" ", text)
    text = text.replace("\u2013", " ").replace("\u2014", " ")  # en/em dashes
    text = text.replace("\u2026", ".").replace("...", ".")

    def newline_to_break(m: re.Match) -> str:
        prev = text[m.start() - 1] if m.start() else ""
        return " " if prev in ".!?,;:" else ". "

    text = re.sub(r"\r?\n+", newline_to_break, text)
    text = text.replace("\u00b0C", " degrees Celsius").replace("\u00b0F", " degrees Fahrenheit")
    text = re.sub(r"(\d+)\s*%", r"\1 percent", text)
    for pattern, words in UNIT_WORDS:
        text = pattern.sub(words, text)
    text = text.replace(";", ",")  # semicolons trigger long prosodic pauses
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"(?<=[.!?,;:])\s*\.", "", text)  # collapse ". ." duplicates
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)    # no space before punctuation
    text = re.sub(r"\(\s+", "(", text).replace(" )", ")")
    return text.strip()


# ---------------------------------------------------------
# Silence smoothing: the CosyVoice engine behind Qwen-Audio-TTS inserts
# stochastic prosodic silences (250-900 ms) at punctuation that no upstream
# parameter removes (see docs/qwen-audio-tts-reference.md). The maintainer-
# endorsed fix is postprocessing: detect long internal silences and shorten
# them. Applied to wav/pcm responses only; compressed formats pass through.
# ---------------------------------------------------------
SILENCE_RMS_THRESHOLD = 300   # 16-bit mono frames quieter than this count as silence
SILENCE_FRAME_MS = 10
CROSSFADE_MS = 4              # short fades at each cut to avoid clicks


def _silence_mask(samples: array, sample_rate: int) -> list:
    """Per-frame (10 ms) boolean: True where the frame is silence."""
    frame = max(1, sample_rate * SILENCE_FRAME_MS // 1000)
    mask = []
    for i in range(0, len(samples), frame):
        chunk = samples[i:i + frame]
        rms = math.sqrt(sum(s * s for s in chunk) / len(chunk))
        mask.append(rms < SILENCE_RMS_THRESHOLD)
    return mask


def compress_silence(samples: array, sample_rate: int, cap_ms: int) -> array:
    """Shorten every silence longer than cap_ms to cap_ms (leading/trailing
    included), splicing regions with a short crossfade. Returns new samples."""
    if not len(samples):
        return samples
    mask = _silence_mask(samples, sample_rate)
    frame = max(1, sample_rate * SILENCE_FRAME_MS // 1000)
    cap_frames = max(1, cap_ms // SILENCE_FRAME_MS)

    # Group kept frames into contiguous regions
    regions = []  # (start_sample, end_sample)
    run = 0
    start = None
    for idx, silent in enumerate(mask):
        if silent:
            run += 1
            keep = run <= cap_frames
        else:
            run = 0
            keep = True
        if keep and start is None:
            start = idx
        elif not keep and start is not None:
            regions.append((start * frame, min(idx * frame, len(samples))))
            start = None
    if start is not None:
        regions.append((start * frame, len(samples)))

    if len(regions) <= 1:
        return samples

    fade = max(1, sample_rate * CROSSFADE_MS // 1000)
    out = array("h")
    for n, (a, b) in enumerate(regions):
        piece = array("h", samples[a:b])
        if n > 0:  # fade in at the cut
            for i in range(min(fade, len(piece))):
                piece[i] = int(piece[i] * i / fade)
        if n < len(regions) - 1:  # fade out into the cut
            for i in range(min(fade, len(piece))):
                piece[len(piece) - 1 - i] = int(piece[len(piece) - 1 - i] * i / fade)
        out.extend(piece)
    return out


def smooth_tts_audio(content: bytes, fmt: str, sample_rate: int, cap_ms: int) -> bytes:
    """Apply silence compression to a synthesized wav/pcm body. Returns the
    original bytes unchanged for compressed formats or unexpected layouts."""
    try:
        if fmt == "wav":
            with wave.open(io.BytesIO(content)) as w:
                if w.getnchannels() != 1 or w.getsampwidth() != 2:
                    return content
                rate = w.getframerate()
                # WAV header duration is documented as an estimate - read to EOF
                frames = w.readframes(2 ** 30)
            samples = array("h")
            samples.frombytes(frames[: len(frames) // 2 * 2])
        elif fmt == "pcm":
            rate = sample_rate
            samples = array("h")
            samples.frombytes(content[: len(content) // 2 * 2])
        else:
            return content

        smoothed = compress_silence(samples, rate, cap_ms)
        if len(smoothed) == len(samples):
            return content

        if fmt == "pcm":
            return smoothed.tobytes()
        out = io.BytesIO()
        with wave.open(out, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(rate)
            w.writeframes(smoothed.tobytes())
        return out.getvalue()
    except (wave.Error, EOFError, ValueError) as exc:
        log.warning("Silence smoothing skipped (%s) - returning audio unchanged", exc)
        return content


# ---------------------------------------------------------
# Compressed-format postprocessing: DashScope's opus/ogg muxer advances the
# granule position by 4800 per page while each page carries a 2880-sample
# (60 ms) packet - a constant 5:3 inflation that violates RFC 7845 §4
# (measured: 22.2 s claimed vs 13.3 s decoded PCM). Players that build the
# timeline from granulepos (Telegram voice notes) therefore drop ~40 ms every
# 100 ms. Remuxing (-c:a copy) does not repair it; only decode -> re-encode
# rewrites a correct granule clock. The pipeline also applies silence
# smoothing to compressed formats, which smooth_tts_audio() cannot do.
# ---------------------------------------------------------
async def _ffmpeg_run(args: list, stdin_data: bytes) -> bytes:
    """Run ffmpeg with piped stdin/stdout; raise on failure or timeout."""
    proc = await asyncio.create_subprocess_exec(
        FFMPEG, *args,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        out, err = await asyncio.wait_for(
            proc.communicate(stdin_data), timeout=FFMPEG_TIMEOUT_S
        )
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise RuntimeError(f"ffmpeg timed out after {FFMPEG_TIMEOUT_S}s")
    if proc.returncode != 0:
        raise RuntimeError(err.decode(errors="replace").strip()[:200] or f"exit {proc.returncode}")
    return out


async def postprocess_compressed_audio(
    content: bytes, fmt: str, sample_rate: int, cap_ms: int, opus_bitrate: int
) -> bytes:
    """Decode upstream wav -> optional silence compression -> re-encode to the
    requested compressed format (correct granule clock). Falls back to the
    upstream bytes unchanged when ffmpeg is missing or any step fails."""
    if not FFMPEG:
        log.warning("re-encode unavailable (%s) - returning upstream audio unchanged", "ffmpeg not found")
        return content
    try:
        pcm = await _ffmpeg_run(
            ["-v", "error", "-i", "pipe:0",
             "-f", "s16le", "-ac", "1", "-ar", str(sample_rate), "pipe:1"],
            content,
        )
        samples = array("h")
        samples.frombytes(pcm[: len(pcm) // 2 * 2])
        if cap_ms > 0 and len(samples):
            smoothed = compress_silence(samples, sample_rate, cap_ms)
            if len(smoothed) != len(samples):
                log.info(
                    "Silence smoothing (%s): %d -> %d samples (cap %d ms)",
                    fmt, len(samples), len(smoothed), cap_ms,
                )
            samples = smoothed
        if fmt == "opus":
            encoder = ["-c:a", "libopus", "-b:a", f"{opus_bitrate}k", "-ar", str(sample_rate), "-f", "ogg"]
        else:  # mp3
            encoder = ["-c:a", "libmp3lame", "-q:a", "4", "-ar", str(sample_rate), "-f", "mp3"]
        out = await _ffmpeg_run(
            ["-v", "error", "-f", "s16le", "-ar", str(sample_rate), "-ac", "1",
             "-i", "pipe:0", *encoder, "pipe:1"],
            samples.tobytes(),
        )
        if not out:
            raise RuntimeError("encoder produced no output")
        return out
    except (OSError, RuntimeError, ValueError) as exc:
        log.warning("re-encode unavailable (%s) - returning upstream audio unchanged", exc)
        return content


def extract_asr_text(output) -> str:
    """Pull recognized text out of the DashScope ASR response (several shapes seen
    in the wild). Tolerates unexpected types: anything unrecognized yields ""."""
    if not isinstance(output, dict):
        return ""

    def text_of(item) -> str:
        """item['text'] when item is a dict holding a str, else ""."""
        if isinstance(item, dict):
            value = item.get("text")
            if isinstance(value, str):
                return value
        return ""

    sentences = output.get("sentences")
    if isinstance(sentences, list) and sentences:
        return " ".join(t for t in map(text_of, sentences) if t).strip()

    sentence = output.get("sentence")
    if isinstance(sentence, dict):
        return text_of(sentence).strip()
    if isinstance(sentence, list):
        return " ".join(t for t in map(text_of, sentence) if t).strip()

    # Chat-completions style fallback
    choices = output.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        message = choices[0].get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, list):
            return " ".join(t for t in map(text_of, content) if t).strip()
        if isinstance(content, str):
            return content.strip()

    return ""


@app.get("/health")
async def health():
    # Deliberately reveals nothing about the configuration: the endpoint is
    # unauthenticated, so even masked key material stays out of the response
    # (it is logged, masked, at startup instead).
    log.debug("Health check requested")
    return {"status": "ok"}


# ---------------------------------------------------------
# 1. TTS: OpenAI /v1/audio/speech -> DashScope SpeechSynthesizer
# ---------------------------------------------------------
@app.post("/v1/audio/speech")
async def openai_tts(request: Request, payload: dict):
    text = payload.get("input")
    if not isinstance(text, str) or not text.strip():
        log.warning("TTS rejected: request body has no non-empty 'input' string")
        raise HTTPException(status_code=400, detail="'input' text is required")

    spoken = normalize_tts_text(text)
    if not spoken:
        log.warning("TTS rejected: 'input' contains no speakable text after normalization")
        raise HTTPException(status_code=400, detail="'input' contains no speakable text")
    if len(spoken) != len(text):
        log.debug("TTS text normalized: %d -> %d chars", len(text), len(spoken))
    text = spoken

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
    if fmt == "ogg":
        fmt = "opus"  # colloquial OpenAI-ecosystem alias; upstream only knows opus
    if not isinstance(fmt, str) or fmt not in TTS_MEDIA_TYPES:
        log.warning("TTS rejected: unsupported response_format %r", fmt)
        raise HTTPException(
            status_code=400,
            detail=f"response_format must be one of: {', '.join(sorted(TTS_MEDIA_TYPES))} "
                   "('ogg' is accepted as an alias of 'opus')",
        )
    try:
        sample_rate = int(payload.get("sample_rate") or settings.tts_sample_rate)
    except (TypeError, ValueError):
        log.warning("TTS rejected: invalid sample_rate %r in request", payload.get("sample_rate"))
        raise HTTPException(status_code=400, detail="sample_rate must be an integer")
    rate_lo, rate_hi = TTS_SAMPLE_RATE_RANGE
    if not rate_lo <= sample_rate <= rate_hi:
        log.warning("TTS rejected: sample_rate %s out of range", sample_rate)
        raise HTTPException(
            status_code=400,
            detail=f"sample_rate must be between {rate_lo} and {rate_hi}",
        )

    # OpenAI 'speed' (0.25-4.0) maps to the upstream 'rate' parameter (0.5-2.0).
    # A slightly elevated rate also measurably shortens prosodic pauses.
    rate = None
    speed = payload.get("speed")
    if speed is not None:
        try:
            speed = float(speed)
        except (TypeError, ValueError):
            log.warning("TTS rejected: invalid speed %r in request", payload.get("speed"))
            raise HTTPException(status_code=400, detail="speed must be a number")
        if not 0.25 <= speed <= 4.0:
            log.warning("TTS rejected: speed %r out of OpenAI range 0.25-4.0", speed)
            raise HTTPException(status_code=400, detail="speed must be between 0.25 and 4.0")
        rate = min(2.0, max(0.5, speed))

    log.info(
        "TTS request: model=%s voice=%s format=%s sample_rate=%s speed=%s text_chars=%d",
        model, voice, fmt, sample_rate, speed if speed is not None else "-", len(text),
    )

    # qwen-audio-3.0-tts takes voice/format/sample_rate inside "input".
    # Compressed formats are fetched as wav and re-encoded locally: DashScope's
    # opus muxer writes a broken (5:3 inflated) granule clock, and silence
    # smoothing needs PCM anyway (see postprocess_compressed_audio).
    upstream_fmt = "wav" if fmt in REENCODE_FORMATS else fmt
    ali_input = {
        "text": text,
        "voice": voice,
        "format": upstream_fmt,
        "sample_rate": sample_rate,
    }
    if rate is not None:
        ali_input["rate"] = rate
    ali_payload = {"model": model, "input": ali_input}

    client = request.app.state.http
    try:
        resp = await client.post(
            settings.ali_tts_url, json=ali_payload, headers=auth_headers(),
        )
    except UPSTREAM_HTTP_ERRORS as exc:
        log.error("TTS upstream unreachable (%s): %s", settings.ali_tts_url, exc)
        raise HTTPException(status_code=502, detail=f"Upstream TTS request failed: {exc}")

    if resp.status_code != 200:
        log.error("TTS upstream returned %s: %s", resp.status_code, resp.text[:500])
        raise HTTPException(
            status_code=resp.status_code,
            detail=f"Upstream TTS error (HTTP {resp.status_code}): {resp.text[:200]}",
        )

    # Parse defensively: a 200 with a non-JSON or unexpectedly shaped body must
    # surface as a 502, not an unhandled exception (500).
    try:
        data = resp.json()
        audio = (data.get("output") or data).get("audio") or {}
        b64_data = audio.get("data")
        audio_url = audio.get("url")
        content = base64.b64decode(b64_data) if b64_data else None
    except (ValueError, AttributeError, TypeError) as exc:
        log.error("TTS upstream response malformed (%s): %.500s", exc, resp.text)
        raise HTTPException(status_code=502, detail="Upstream TTS response was malformed")

    if not content:
        if isinstance(audio_url, str) and audio_url:
            # Non-streaming mode returns a download URL (valid 24h)
            try:
                dl = await client.get(audio_url)
            except UPSTREAM_HTTP_ERRORS as exc:
                log.error("TTS audio download failed: %s", exc)
                raise HTTPException(status_code=502, detail=f"Failed to download synthesized audio: {exc}")
            if dl.status_code != 200:
                log.error("TTS audio download returned %s", dl.status_code)
                raise HTTPException(
                    status_code=502, detail=f"Failed to download synthesized audio: {dl.status_code}"
                )
            content = dl.content
        if not content:
            log.error("TTS upstream response contained no audio: %s", str(data)[:500])
            raise HTTPException(
                status_code=502, detail="Upstream TTS response contained no audio"
            )

    if fmt in REENCODE_FORMATS:
        reencoded = await postprocess_compressed_audio(
            content, fmt, sample_rate,
            settings.tts_silence_cap_ms, settings.tts_opus_bitrate,
        )
        if reencoded is not content:
            log.info(
                "Re-encoded upstream wav -> %s: %d -> %d bytes", fmt, len(content), len(reencoded)
            )
        content = reencoded
    elif settings.tts_silence_cap_ms > 0:
        smoothed = smooth_tts_audio(content, fmt, sample_rate, settings.tts_silence_cap_ms)
        if smoothed is not content:
            log.info(
                "Silence smoothing: %d -> %d bytes (cap %d ms)",
                len(content), len(smoothed), settings.tts_silence_cap_ms,
            )
            content = smoothed

    media_type = TTS_MEDIA_TYPES[fmt]
    log.info("TTS success: returned %d bytes of %s", len(content), media_type)
    return Response(content=content, media_type=media_type)


# ---------------------------------------------------------
# 2. STT: OpenAI /v1/audio/transcriptions -> DashScope ASR (sync, base64 input)
# ---------------------------------------------------------
@app.post("/v1/audio/transcriptions")
async def openai_transcriptions(
    request: Request,
    file: UploadFile = File(...),
    model: Optional[str] = Form(None),
    language: Optional[str] = Form(None),
):
    # OpenAI spec: 'model' is a required field of POST /v1/audio/transcriptions
    if not model or not model.strip():
        log.warning("STT rejected: multipart form has no non-empty 'model' field")
        raise HTTPException(status_code=400, detail="'model' form field is required")
    model = model.strip()

    # Bound memory: never buffer more than the configured limit. base64 expands
    # data (~4/3x), so any raw upload larger than max_base64_bytes would fail the
    # exact base64 check below regardless - reject it during the read instead of
    # after fully buffering + encoding a potentially huge file.
    limit = settings.max_base64_bytes
    chunks: list[bytes] = []
    received = 0
    while True:
        chunk = await file.read(1024 * 1024)
        if not chunk:
            break
        chunks.append(chunk)
        received += len(chunk)
        if received > limit:
            break
    content = b"".join(chunks)
    if received > limit:
        log.warning(
            "STT rejected: upload %r exceeds the %d-byte limit", file.filename, limit
        )
        raise HTTPException(
            status_code=413,
            detail=f"Audio too large: exceeds the {limit}-byte upload limit",
        )
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

    client = request.app.state.http
    try:
        resp = await client.post(
            settings.ali_stt_url, json=ali_payload, headers=headers,
        )
    except UPSTREAM_HTTP_ERRORS as exc:
        log.error("STT upstream unreachable (%s): %s", settings.ali_stt_url, exc)
        raise HTTPException(status_code=502, detail=f"Upstream STT request failed: {exc}")

    if resp.status_code != 200:
        log.error("STT upstream returned %s: %s", resp.status_code, resp.text[:500])
        raise HTTPException(
            status_code=resp.status_code,
            detail=f"Upstream STT error (HTTP {resp.status_code}): {resp.text[:200]}",
        )

    try:
        data = resp.json()
    except ValueError as exc:
        log.error("STT upstream returned a non-JSON body (%s): %.500s", exc, resp.text)
        raise HTTPException(status_code=502, detail="Upstream STT returned a non-JSON response")
    output = data.get("output", data) if isinstance(data, dict) else {}
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
