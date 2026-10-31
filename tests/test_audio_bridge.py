"""Unit + endpoint tests for audio-bridge.py.

No network access happens: a FakeHTTP object stands in for the shared
httpx.AsyncClient (app.state.http). The bridge module has a hyphen in its
filename, so it is loaded via importlib with a valid test configuration in
the environment (Settings validates at import time and exits otherwise).

Run:  .venv/bin/python -m pytest tests -q
"""
import base64
import importlib.util
import io
import itertools
import json
import math
import os
import shutil
import struct
import subprocess
import wave
from array import array
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

BRIDGE_PATH = Path(__file__).resolve().parent.parent / "audio-bridge.py"

TEST_ENV = {
    "ALI_API_KEY": "sk-test-1234567890",
    "BRIDGE_HOST": "127.0.0.1",
    "BRIDGE_PORT": "4005",
    "ALI_TTS_URL": "https://upstream.test/api/v1/services/audio/tts/SpeechSynthesizer",
    "ALI_STT_URL": "https://upstream.test/api/v1/services/aigc/multimodal-generation/generation",
    "TTS_FORMAT": "wav",
    "TTS_SAMPLE_RATE": "24000",
    "REQUEST_TIMEOUT": "120",
    "MAX_BASE64_BYTES": "10485760",
    "TTS_SILENCE_CAP_MS": "0",
    "LOG_LEVEL": "INFO",
}


def load_fresh_module():
    spec = importlib.util.spec_from_file_location("audio_bridge_under_test", BRIDGE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def bridge():
    os.environ.update(TEST_ENV)
    return load_fresh_module()


@pytest.fixture()
def client(bridge):
    # Entering the TestClient context runs the lifespan (creates a real
    # httpx.AsyncClient); tests below swap app.state.http for FakeHTTP.
    with TestClient(bridge.app) as c:
        yield c


# ---------------------------------------------------------
# Fakes
# ---------------------------------------------------------
class FakeResponse:
    """Mimics the parts of httpx.Response the bridge uses."""

    def __init__(self, status_code=200, json_data=None, body_text=None, content=b""):
        self.status_code = status_code
        self._json = json_data
        self.content = content
        self.text = body_text if body_text is not None else (
            json.dumps(json_data) if json_data is not None else ""
        )

    def json(self):
        if self._json is None:
            raise ValueError("response body is not JSON")
        return self._json


class FakeHTTP:
    """Stands in for httpx.AsyncClient; routes by URL and records calls.
    A response given as an Exception instance is raised instead."""

    def __init__(self, tts=None, stt=None, download=None):
        self._tts = tts
        self._stt = stt
        self._download = download
        self.calls = []

    async def post(self, url, json=None, headers=None):
        self.calls.append({"method": "post", "url": url, "json": json, "headers": headers})
        target = self._tts if "SpeechSynthesizer" in url else self._stt
        if isinstance(target, Exception):
            raise target
        return target

    async def get(self, url):
        self.calls.append({"method": "get", "url": url})
        if isinstance(self._download, Exception):
            raise self._download
        return self._download

    async def aclose(self):
        pass


def use_fake_http(bridge, **kwargs):
    fake = FakeHTTP(**kwargs)
    bridge.app.state.http = fake
    return fake


# ---------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------
def tone(rate, seconds, freq=440, amp=3000):
    return array(
        "h",
        [int(amp * math.sin(2 * math.pi * freq * i / rate)) for i in range(int(rate * seconds))],
    )


def silence(rate, seconds):
    return array("h", [0] * int(rate * seconds))


def make_wav(samples, rate=24000):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(samples.tobytes())
    return buf.getvalue()


# ---------------------------------------------------------
# ffmpeg helpers (re-encode pipeline tests)
# ---------------------------------------------------------
FFMPEG = shutil.which("ffmpeg")
requires_ffmpeg = pytest.mark.skipif(
    FFMPEG is None, reason="ffmpeg not installed - re-encode tests need a real ffmpeg"
)


def ffmpeg_run(args, stdin_data):
    r = subprocess.run(
        [FFMPEG, *args], input=stdin_data, capture_output=True, timeout=60
    )
    assert r.returncode == 0, r.stderr.decode(errors="replace")[:300]
    return r.stdout


def decode_pcm(data, rate):
    """Decode an audio body to mono s16le samples at `rate` via ffmpeg."""
    pcm = ffmpeg_run(
        ["-v", "error", "-i", "pipe:0", "-f", "s16le", "-ac", "1", "-ar", str(rate), "pipe:1"],
        data,
    )
    samples = array("h")
    samples.frombytes(pcm[: len(pcm) // 2 * 2])
    return samples


def ogg_last_granulepos(data):
    """Walk Ogg pages and return the final granulepos (RFC 7845: total PCM
    sample count, in 48 kHz units, up to the end of the last page)."""
    pos = 0
    last = 0
    while pos < len(data) and data[pos:pos + 4] == b"OggS":
        nsegs = data[pos + 26]
        body_len = sum(data[pos + 27:pos + 27 + nsegs])
        last = struct.unpack("<q", data[pos + 6:pos + 14])[0]
        pos += 27 + nsegs + body_len
    return last


TTS_WAV = make_wav(silence(24000, 0.1))
TTS_BODY = {"model": "qwen-audio-3.0-tts-plus", "voice": "v1", "input": "Hello **world**"}


def tts_data_response(raw: bytes) -> FakeResponse:
    return FakeResponse(
        json_data={"output": {"audio": {"data": base64.b64encode(raw).decode()}}}
    )


# ---------------------------------------------------------
# normalize_tts_text
# ---------------------------------------------------------
def test_normalize_plain_text_unchanged(bridge):
    text = "Hello world. It is fine."
    assert bridge.normalize_tts_text(text) == text


def test_normalize_strips_markdown_and_emoji(bridge):
    out = bridge.normalize_tts_text("# Title\n**bold** and \U0001f30d emoji")
    assert "#" not in out
    assert "**" not in out
    assert "\U0001f30d" not in out
    assert "bold" in out and "emoji" in out


def test_normalize_bullets_and_newlines(bridge):
    assert bridge.normalize_tts_text("- item one\n- item two") == "item one. item two"


def test_normalize_numeric_ranges_keeps_dates(bridge):
    out = bridge.normalize_tts_text("Wind 10-15 km/h on 2026-01-15")
    assert "10 to 15" in out
    assert "kilometers per hour" in out
    assert "2026-01-15" in out


def test_normalize_units(bridge):
    out = bridge.normalize_tts_text("23\u00b0C, 50 % rain, 60 mph, 5 m/s")
    assert "23 degrees Celsius" in out
    assert "50 percent" in out
    assert "miles per hour" in out
    assert "meters per second" in out


def test_normalize_emoji_only_input_rejected_by_endpoint(client, bridge):
    use_fake_http(bridge, tts=FakeResponse(json_data={}))
    r = client.post("/v1/audio/speech", json={**TTS_BODY, "input": "\U0001f30d\U0001f600"})
    assert r.status_code == 400
    assert "no speakable text" in r.json()["detail"]


# ---------------------------------------------------------
# extract_asr_text
# ---------------------------------------------------------
@pytest.mark.parametrize("output,expected", [
    ({"sentences": [{"text": "a"}, {"text": "b"}]}, "a b"),
    ({"sentence": {"text": " hi "}}, "hi"),
    ({"sentence": [{"text": "x"}, {"text": "y"}]}, "x y"),
    ({"choices": [{"message": {"content": "deep"}}]}, "deep"),
    ({"choices": [{"message": {"content": [{"text": "p"}, {"text": "q"}]}}]}, "p q"),
    ({}, ""),
    ("not-a-dict", ""),
    (None, ""),
    ({"sentences": ["garbage", {"text": None}, {"text": "ok"}]}, "ok"),
    ({"choices": ["not-a-dict"]}, ""),
    ({"choices": [{"message": "not-a-dict"}]}, ""),
])
def test_extract_asr_text(bridge, output, expected):
    assert bridge.extract_asr_text(output) == expected


# ---------------------------------------------------------
# Silence smoothing
# ---------------------------------------------------------
def test_compress_silence_shortens_to_cap(bridge):
    rate = 8000
    samples = tone(rate, 0.5) + silence(rate, 2.0) + tone(rate, 0.5)
    out = bridge.compress_silence(samples, rate, cap_ms=250)
    expected = int(rate * (0.5 + 0.25 + 0.5))
    assert abs(len(out) - expected) <= int(rate * 0.02)
    assert len(out) < len(samples)


def test_smooth_tts_audio_wav(bridge):
    rate = 8000
    samples = tone(rate, 0.3) + silence(rate, 1.5) + tone(rate, 0.3)
    wav_in = make_wav(samples, rate)
    out = bridge.smooth_tts_audio(wav_in, "wav", rate, cap_ms=250)
    assert len(out) < len(wav_in)
    with wave.open(io.BytesIO(out)) as w:
        assert w.getframerate() == rate
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2


def test_smooth_tts_audio_pcm(bridge):
    rate = 8000
    samples = tone(rate, 0.2) + silence(rate, 1.0) + tone(rate, 0.2)
    out = bridge.smooth_tts_audio(samples.tobytes(), "pcm", rate, cap_ms=200)
    assert len(out) < len(samples.tobytes())


def test_smooth_tts_audio_passthrough(bridge):
    rate = 8000
    mp3_bytes = b"\xff\xfb\x90\x00fake-mp3"
    assert bridge.smooth_tts_audio(mp3_bytes, "mp3", rate, 250) is mp3_bytes
    # stereo wav is returned unchanged
    stereo = io.BytesIO()
    with wave.open(stereo, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes((tone(rate, 0.2) + silence(rate, 0.5)).tobytes())
    assert bridge.smooth_tts_audio(stereo.getvalue(), "wav", rate, 250) == stereo.getvalue()


# ---------------------------------------------------------
# Settings validation
# ---------------------------------------------------------
def test_settings_accepts_valid_config(monkeypatch):
    for key, value in TEST_ENV.items():
        monkeypatch.setenv(key, value)
    module = load_fresh_module()
    assert module.settings.tts_format == "wav"
    assert module.settings.bridge_port == 4005


def test_settings_rejects_unknown_tts_format(monkeypatch):
    for key, value in TEST_ENV.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("TTS_FORMAT", "flac")
    with pytest.raises(SystemExit) as excinfo:
        load_fresh_module()
    assert excinfo.value.code == 2


def test_settings_ogg_format_alias(monkeypatch):
    for key, value in TEST_ENV.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("TTS_FORMAT", "ogg")
    module = load_fresh_module()
    assert module.settings.tts_format == "opus"


def test_settings_opus_bitrate_default_and_validation(monkeypatch):
    for key, value in TEST_ENV.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("TTS_OPUS_BITRATE", raising=False)
    assert load_fresh_module().settings.tts_opus_bitrate == 32
    monkeypatch.setenv("TTS_OPUS_BITRATE", "64")
    assert load_fresh_module().settings.tts_opus_bitrate == 64
    monkeypatch.setenv("TTS_OPUS_BITRATE", "1000")  # above the 6-510 range
    with pytest.raises(SystemExit) as excinfo:
        load_fresh_module()
    assert excinfo.value.code == 2


# ---------------------------------------------------------
# /health
# ---------------------------------------------------------
def test_health_exposes_no_key_material(client, bridge):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}
    assert bridge.settings.ali_api_key[:6] not in r.text


def test_health_leaks_nothing(client, bridge):
    # Guards the deployed-old-build regression: /health once returned
    # {"status": "ok", "api_key": "sk-...***"}. No key material, ever.
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert "api_key" not in body
    assert "api_key" not in r.text and bridge.settings.ali_api_key[:6] not in r.text


# ---------------------------------------------------------
# TTS endpoint
# ---------------------------------------------------------
def test_tts_success_base64(client, bridge):
    fake = use_fake_http(bridge, tts=tts_data_response(TTS_WAV))
    r = client.post("/v1/audio/speech", json=TTS_BODY)
    assert r.status_code == 200
    assert r.content == TTS_WAV
    assert r.headers["content-type"] == "audio/wav"
    sent = fake.calls[0]["json"]
    assert sent["model"] == "qwen-audio-3.0-tts-plus"
    assert sent["input"]["voice"] == "v1"
    assert sent["input"]["format"] == "wav"
    assert sent["input"]["sample_rate"] == 24000
    assert "**" not in sent["input"]["text"]  # normalized before synthesis
    assert fake.calls[0]["headers"]["Authorization"] == "Bearer " + bridge.settings.ali_api_key


def test_tts_success_download_url(client, bridge):
    use_fake_http(
        bridge,
        tts=FakeResponse(json_data={"output": {"audio": {"url": "https://cdn.test/a.wav"}}}),
        download=FakeResponse(content=TTS_WAV),
    )
    r = client.post("/v1/audio/speech", json=TTS_BODY)
    assert r.status_code == 200
    assert r.content == TTS_WAV


@pytest.mark.parametrize("body,expected_fragment", [
    ({"model": "m", "voice": "v"}, "'input'"),
    ({"voice": "v", "input": "hi"}, "'model'"),
    ({"model": "m", "input": "hi"}, "'voice'"),
])
def test_tts_required_fields(client, bridge, body, expected_fragment):
    use_fake_http(bridge, tts=FakeResponse(json_data={}))
    r = client.post("/v1/audio/speech", json=body)
    assert r.status_code == 400
    assert expected_fragment in r.json()["detail"]


def test_tts_rejects_unsupported_response_format(client, bridge):
    use_fake_http(bridge, tts=FakeResponse(json_data={}))
    r = client.post("/v1/audio/speech", json={**TTS_BODY, "response_format": "flac"})
    assert r.status_code == 400
    assert "response_format" in r.json()["detail"]


def test_tts_rejects_bad_sample_rate(client, bridge):
    use_fake_http(bridge, tts=FakeResponse(json_data={}))
    r = client.post("/v1/audio/speech", json={**TTS_BODY, "sample_rate": "abc"})
    assert r.status_code == 400
    r = client.post("/v1/audio/speech", json={**TTS_BODY, "sample_rate": 10})
    assert r.status_code == 400
    assert "between" in r.json()["detail"]


def test_tts_speed_validation_and_clamping(client, bridge):
    fake = use_fake_http(bridge, tts=tts_data_response(TTS_WAV))
    assert client.post("/v1/audio/speech", json={**TTS_BODY, "speed": "fast"}).status_code == 400
    assert client.post("/v1/audio/speech", json={**TTS_BODY, "speed": 5}).status_code == 400
    assert fake.calls == []  # rejected before any upstream call

    assert client.post("/v1/audio/speech", json={**TTS_BODY, "speed": 4.0}).status_code == 200
    assert fake.calls[-1]["json"]["input"]["rate"] == 2.0
    assert client.post("/v1/audio/speech", json={**TTS_BODY, "speed": 0.25}).status_code == 200
    assert fake.calls[-1]["json"]["input"]["rate"] == 0.5


@pytest.mark.parametrize("fmt,mime", [
    ("mp3", "audio/mpeg"),
    ("opus", "audio/ogg"),
    ("pcm", "audio/pcm"),
])
def test_tts_media_types(client, bridge, fmt, mime):
    use_fake_http(bridge, tts=tts_data_response(b"raw-audio-bytes"))
    r = client.post("/v1/audio/speech", json={**TTS_BODY, "response_format": fmt})
    assert r.status_code == 200
    assert r.headers["content-type"] == mime
    assert r.content == b"raw-audio-bytes"


# ---------------------------------------------------------
# Compressed-format re-encode pipeline (granule clock fix + silence cap)
# ---------------------------------------------------------
def test_ogg_alias_maps_to_opus(client, bridge):
    # "ogg" is accepted as an alias of "opus"; upstream is asked for wav
    # (the re-encode source) and the client gets audio/ogg.
    fake = use_fake_http(bridge, tts=tts_data_response(TTS_WAV))
    r = client.post("/v1/audio/speech", json={**TTS_BODY, "response_format": "ogg"})
    assert r.status_code == 200
    assert r.headers["content-type"] == "audio/ogg"
    assert fake.calls[0]["json"]["input"]["format"] == "wav"


def test_opus_requested_upstream_as_wav(client, bridge):
    fake = use_fake_http(bridge, tts=tts_data_response(TTS_WAV))
    for fmt in ("opus", "mp3"):
        r = client.post("/v1/audio/speech", json={**TTS_BODY, "response_format": fmt})
        assert r.status_code == 200
        assert fake.calls[-1]["json"]["input"]["format"] == "wav"


@requires_ffmpeg
def test_opus_response_granule_matches_pcm(client, bridge):
    # Bug 1 regression: DashScope's own opus muxer advances granulepos 5:3
    # faster than real PCM (RFC 7845 §4 violation). The locally re-encoded
    # response must carry a 1:1 granule clock (± 3000 samples covers the
    # OpusHead pre-skip / encoder delay).
    rate = 48000
    samples = tone(rate, 3.0) + silence(rate, 0.9)
    use_fake_http(bridge, tts=tts_data_response(make_wav(samples, rate)))
    r = client.post("/v1/audio/speech", json={
        **TTS_BODY, "response_format": "opus", "sample_rate": rate,
    })
    assert r.status_code == 200
    assert r.content[:4] == b"OggS"
    granule = ogg_last_granulepos(r.content)
    decoded = decode_pcm(r.content, rate)
    assert granule > 0 and len(decoded) > 0
    assert abs(granule - len(decoded)) <= 3000
    # and the duration is right: ~3.9 s of input audio (cap disabled in TEST_ENV)
    assert abs(len(decoded) / rate - 3.9) < 0.2


@requires_ffmpeg
def test_silence_cap_applies_to_opus(client, bridge, monkeypatch):
    # Bug 2 regression: TTS_SILENCE_CAP_MS must smooth compressed formats too.
    monkeypatch.setattr(bridge.settings, "tts_silence_cap_ms", 250)
    rate = 48000
    samples = tone(rate, 1.0) + silence(rate, 0.9) + tone(rate, 1.0)
    use_fake_http(bridge, tts=tts_data_response(make_wav(samples, rate)))
    r = client.post("/v1/audio/speech", json={
        **TTS_BODY, "response_format": "opus", "sample_rate": rate,
    })
    assert r.status_code == 200
    decoded = decode_pcm(r.content, rate)
    # ~650 ms of silence removed
    assert len(decoded) < len(samples) - int(rate * 0.5)
    mask = bridge._silence_mask(decoded, rate)
    longest_ms = max(
        (len(list(run)) for silent, run in itertools.groupby(mask) if silent),
        default=0,
    ) * bridge.SILENCE_FRAME_MS
    assert longest_ms <= 250 + 60


def test_reencode_fallback_without_ffmpeg(client, bridge, monkeypatch):
    # Missing ffmpeg binary -> upstream audio passes through unchanged (200).
    monkeypatch.setattr(bridge, "FFMPEG", None)
    use_fake_http(bridge, tts=tts_data_response(b"raw-audio-bytes"))
    r = client.post("/v1/audio/speech", json={**TTS_BODY, "response_format": "opus"})
    assert r.status_code == 200
    assert r.content == b"raw-audio-bytes"
    assert r.headers["content-type"] == "audio/ogg"


@requires_ffmpeg
def test_reencode_fallback_on_ffmpeg_failure(client, bridge):
    # Undecodable upstream body -> warning + unchanged bytes, never a 5xx.
    use_fake_http(bridge, tts=tts_data_response(b"not-audio-at-all"))
    r = client.post("/v1/audio/speech", json={**TTS_BODY, "response_format": "opus"})
    assert r.status_code == 200
    assert r.content == b"not-audio-at-all"


def test_tts_upstream_error_forwarded_truncated(client, bridge):
    use_fake_http(bridge, tts=FakeResponse(status_code=401, body_text="X" * 5000))
    r = client.post("/v1/audio/speech", json=TTS_BODY)
    assert r.status_code == 401
    detail = r.json()["detail"]
    assert "Upstream TTS error" in detail
    assert len(detail) < 300  # raw body is truncated, not forwarded verbatim


def test_tts_upstream_non_json_502(client, bridge):
    use_fake_http(bridge, tts=FakeResponse(status_code=200, body_text="<html>gateway</html>"))
    r = client.post("/v1/audio/speech", json=TTS_BODY)
    assert r.status_code == 502
    assert "malformed" in r.json()["detail"]


def test_tts_upstream_bad_shape_502(client, bridge):
    use_fake_http(bridge, tts=FakeResponse(json_data={"output": {"audio": "garbage"}}))
    r = client.post("/v1/audio/speech", json=TTS_BODY)
    assert r.status_code == 502
    assert "malformed" in r.json()["detail"]


def test_tts_upstream_invalid_base64_502(client, bridge):
    use_fake_http(bridge, tts=FakeResponse(json_data={"output": {"audio": {"data": "a"}}}))
    r = client.post("/v1/audio/speech", json=TTS_BODY)
    assert r.status_code == 502


def test_tts_upstream_no_audio_502(client, bridge):
    use_fake_http(bridge, tts=FakeResponse(json_data={"output": {}}))
    r = client.post("/v1/audio/speech", json=TTS_BODY)
    assert r.status_code == 502
    assert "no audio" in r.json()["detail"]


def test_tts_network_error_502(client, bridge):
    use_fake_http(bridge, tts=httpx.ConnectError("boom"))
    r = client.post("/v1/audio/speech", json=TTS_BODY)
    assert r.status_code == 502
    assert "Upstream TTS request failed" in r.json()["detail"]


def test_tts_invalid_upstream_url_502(client, bridge):
    # httpx.InvalidURL is not a RequestError subclass - must still be a 502
    use_fake_http(bridge, tts=httpx.InvalidURL("Invalid port"))
    r = client.post("/v1/audio/speech", json=TTS_BODY)
    assert r.status_code == 502
    assert "Upstream TTS request failed" in r.json()["detail"]


def test_tts_download_invalid_url_502(client, bridge):
    use_fake_http(
        bridge,
        tts=FakeResponse(json_data={"output": {"audio": {"url": "http://cdn.test:bad/a.wav"}}}),
        download=httpx.InvalidURL("Invalid port"),
    )
    r = client.post("/v1/audio/speech", json=TTS_BODY)
    assert r.status_code == 502
    assert "Failed to download synthesized audio" in r.json()["detail"]


def test_lifespan_client_follows_redirects(client, bridge):
    # requests followed redirects by default; the shared httpx client must too
    # (DashScope download URLs may redirect via OSS/CDN).
    assert bridge.app.state.http.follow_redirects is True


def test_tts_download_failure_502(client, bridge):
    use_fake_http(
        bridge,
        tts=FakeResponse(json_data={"output": {"audio": {"url": "https://cdn.test/a.wav"}}}),
        download=FakeResponse(status_code=500),
    )
    r = client.post("/v1/audio/speech", json=TTS_BODY)
    assert r.status_code == 502


# ---------------------------------------------------------
# STT endpoint
# ---------------------------------------------------------
def stt_post(client, content=TTS_WAV, filename="speech.wav", **data):
    return client.post(
        "/v1/audio/transcriptions",
        files={"file": (filename, content, "audio/wav")},
        data=data,
    )


def test_stt_success(client, bridge):
    fake = use_fake_http(
        bridge, stt=FakeResponse(json_data={"output": {"sentence": {"text": "hello world"}}})
    )
    r = stt_post(client, model="qwen-audio-3.0-asr-flash", language="en")
    assert r.status_code == 200
    assert r.json() == {"text": "hello world"}
    sent = fake.calls[0]["json"]
    assert sent["model"] == "qwen-audio-3.0-asr-flash"
    assert sent["parameters"] == {"format": "wav", "language_hints": ["en"]}
    part = sent["input"]["messages"][0]["content"][0]["input_audio"]["data"]
    assert part.startswith("data:audio/wav;base64,")
    assert base64.b64decode(part.split(",", 1)[1]) == TTS_WAV
    assert fake.calls[0]["headers"]["X-DashScope-SSE"] == "disable"


def test_stt_requires_model(client, bridge):
    use_fake_http(bridge, stt=FakeResponse(json_data={}))
    r = stt_post(client)
    assert r.status_code == 400
    assert "'model'" in r.json()["detail"]


def test_stt_empty_file_400(client, bridge):
    use_fake_http(bridge, stt=FakeResponse(json_data={}))
    r = stt_post(client, content=b"", model="m")
    assert r.status_code == 400
    assert "empty" in r.json()["detail"]


def test_stt_oversize_413(client, bridge, monkeypatch):
    monkeypatch.setattr(bridge.settings, "max_base64_bytes", 10)
    use_fake_http(bridge, stt=FakeResponse(json_data={}))
    r = stt_post(client, model="m")
    assert r.status_code == 413


def test_stt_unknown_extension_sent_as_wav(client, bridge):
    fake = use_fake_http(
        bridge, stt=FakeResponse(json_data={"output": {"sentence": {"text": "x"}}})
    )
    r = stt_post(client, filename="audio.bin", model="m")
    assert r.status_code == 200
    assert fake.calls[0]["json"]["parameters"]["format"] == "wav"


def test_stt_upstream_non_json_502(client, bridge):
    use_fake_http(bridge, stt=FakeResponse(status_code=200, body_text="oops"))
    r = stt_post(client, model="m")
    assert r.status_code == 502
    assert "non-JSON" in r.json()["detail"]


def test_stt_upstream_error_forwarded_truncated(client, bridge):
    use_fake_http(bridge, stt=FakeResponse(status_code=400, body_text="bad request " * 100))
    r = stt_post(client, model="m")
    assert r.status_code == 400
    assert "Upstream STT error" in r.json()["detail"]
    assert len(r.json()["detail"]) < 300


def test_stt_network_error_502(client, bridge):
    use_fake_http(bridge, stt=httpx.ConnectError("down"))
    r = stt_post(client, model="m")
    assert r.status_code == 502
    assert "Upstream STT request failed" in r.json()["detail"]


def test_stt_unrecognized_audio_returns_empty_text(client, bridge):
    use_fake_http(bridge, stt=FakeResponse(json_data={"output": {}}))
    r = stt_post(client, model="m")
    assert r.status_code == 200
    assert r.json() == {"text": ""}
