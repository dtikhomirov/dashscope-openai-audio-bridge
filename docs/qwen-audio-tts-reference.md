# Qwen-Audio-TTS parameter & limits reference

Research notes for this bridge, compiled **2026-10-12** from official Alibaba Cloud
Model Studio documentation plus ~30 live probes against the exact endpoint this
bridge uses. Everything marked *verified* was confirmed empirically on:

- **Endpoint**: `https://token-plan.ap-southeast-1.maas.aliyuncs.com/api/v1/services/audio/tts/SpeechSynthesizer` (HTTP, non-streaming)
- **Model**: `qwen-audio-3.0-tts-plus`
- **Voice**: `qwen-audio-3.0-tts-plus-loongjameszhao`
- **Engine**: CosyVoice (revealed by error message prefixes, e.g. `[cosyvoice]Engine error [421]`)

## TL;DR for agents

- Quality knobs live **inside `input`** of the HTTP payload (same names as the
  WebSocket `parameters` object): `rate`, `volume`, `pitch`, `language_hints`,
  `hot_fix`, `instruction`, `enable_ssml`, `seed`, `bit_rate`, `text_type`,
  `word_timestamp_enabled`.
- The three levers that matter for smoother speech: `rate` ≈ 1.05–1.15,
  `language_hints: ["en"]`, and `hot_fix.replace` for surgical word swaps.
- `seed` is accepted but **not** deterministic on this endpoint — do not rely on it.
- The practical length ceiling is the bridge's `REQUEST_TIMEOUT` (120 s),
  **not** a character limit: ≈31 chars of text produced ≈2.8 s of audio, so
  ~1,300–1,500 spoken characters fit in one non-streaming request. The documented
  per-message cap is 20,000 characters.
- The widely quoted **600-character** limit applies to `qwen3-tts-flash` on the
  *multimodal-generation* endpoint — a different API family. It does **not**
  apply here.

## Verified parameter table (HTTP endpoint, all inside `input`)

| Parameter | Type | Valid values / range | Verified effect | Notes |
|---|---|---|---|---|
| `text` | string | ≤ 20,000 chars per message (docs) | — | 5,000 chars accepted empirically |
| `voice` | string | required | — | See voice list doc; base voices are named `qwen-audio-3.0-tts-plus-<suffix>` |
| `format` | string | `pcm`, `wav`, `mp3` (default), `opus` | `flac` → 400 "Request format is invalid!" | |
| `sample_rate` | int | docs: 8000/16000/22050 (default)/24000/44100/48000 | **arbitrary values honored** — 12345 Hz produced a WAV whose header says 12345 | endpoint is laxer than docs |
| `rate` | float | **0.5–2.0** (0.4 / 2.1 / 5.0 → 400 err 421) | 0.6 → 4.40 s, 1.0 → 2.80 s, 1.8 → 1.38 s for identical text | speech speed |
| `volume` | int | **0–100** (150 → 400) | RMS 515 / 2100 / 4755 at 10 / 50 (default) / 100 | loudness |
| `pitch` | float | **0.5–2.0** (2.5 → 400 err 422) | zero-crossing rate 1778 Hz @0.6 vs 3859 Hz @1.8 (baseline ~2700–3000) | also perturbs duration |
| `language_hints` | array[string] | `zh en fr de ja ko ru pt th id vi es it ms fil ar` | `["zh"]` on English text changed output (2.08 s vs 2.80 s); `["en"]` was byte-identical to baseline | only first element processed; docs: fixes digit/symbol/abbreviation pronunciation |
| `hot_fix` | object | `{replace: [{from: to}], pronunciation: {word: pinyin}}` | **STT-proven**: `replace [{"test":"banana"}]` → ASR transcribed "banana" | text replacement + pinyin correction before synthesis |
| `instruction` | string | free text | accepted (200); effect is voice-dependent | dialect/emotion/character control; err 423 "Voice does not support emotion" exists for emotion tags |
| `enable_ssml` | bool | default `false` | accepted (200) with plain text | when true only one text chunk allowed (WS); voice/model restrictions apply |
| `seed` | int | 0–65535 | accepted but **NOT deterministic**: two identical `seed=42` requests → different audio (2.64 s vs 2.96 s) | identical payloads *without* seed are occasionally byte-identical (replica-dependent) |
| `word_timestamp_enabled` | bool | default `false` | accepted; no timestamps in non-streaming HTTP response | docs: streaming output only |
| `bit_rate` | int | 6–510 kbps (mp3/opus only) | `bit_rate=128` measurably changed mp3 size (41.9 KB vs 57.2 KB default for same text) | docs default 32 |
| `text_type` | string | `PlainText` | byte-identical to omitting | |
| `enable_aigc_tag` (+`aigc_propagator`, `aigc_propagate_id`) | bool/string | default `false` | not probed | embeds AIGC watermark in wav/mp3/opus |

## Limits (definitive)

| Limit | Value | Source |
|---|---|---|
| Text per message | 20,000 chars | docs (WebSocket `continue-task`) |
| Cumulative text per task | 200,000 chars | docs |
| Send interval (WebSocket) | ≤ 23 s between messages or connection times out | docs |
| Practical HTTP ceiling for this bridge | `REQUEST_TIMEOUT` = 120 s ⇒ ~1,300–1,500 spoken chars/request | measured (31 chars ≈ 2.8 s audio) |
| Audio formats | pcm, wav, mp3 (default), opus | verified |
| Sample rates | 8000/16000/22050/24000/44100/48000 documented; arbitrary integers accepted | verified |
| Billing unit | `usage.characters` = input character count | verified |
| Audio URL validity (non-streaming) | 24 h (`output.audio.url`) | docs |
| Base64 audio in response | `output.audio.data` populated in streaming mode; our HTTP calls returned a URL | docs + observed |

## Gotchas

- **Error messages lie about which parameter is wrong.** Engine error 421 always
  says "Invalid volume parameter" even when `rate` is out of range; 422 says
  "Invalid sample rate parameter" for a bad `pitch`. Trust the error *code*.
- TTS is **stochastic** on the same input (pause counts and durations vary run to
  run; occasionally two runs are byte-identical — likely per-replica determinism).
  Pause-measurement experiments must repeat runs.
- Returned WAV headers can have a **bogus frame count** — compute duration from
  the actual data chunk length, not `nframes`.
- Voices are model-family-scoped: passing a `qwen-audio-3.0-tts-flash` voice
  (e.g. `loongmary`) to `qwen-audio-3.0-tts-plus` returns `[cosyvoice] Engine error [411]`.
- STT side (`qwen-audio-3.0-asr-flash`, multimodal-generation endpoint): response
  contains **word-level timestamps** (`output.sentence.words[].begin_time/end_time`),
  useful for A/B verification of TTS changes.

## Reproducible probe methodology

```bash
# 1. POST a tiny text with the parameter under test inside "input"
curl -s "$ALI_TTS_URL" -H "Authorization: Bearer $ALI_API_KEY" -H 'Content-Type: application/json' -d '{
  "model": "qwen-audio-3.0-tts-plus",
  "input": {"text": "Hi there, this is a quick test.",
            "voice": "qwen-audio-3.0-tts-plus-loongjameszhao",
            "format": "wav", "sample_rate": 24000,
            "rate": 1.3}
}'
# 2. Non-streaming responses give output.audio.url (valid 24h) — download the WAV.
# 3. Verify effect physically: duration (chars→seconds), RMS (volume),
#    zero-crossing rate (pitch), or STT round-trip through
#    qwen-audio-3.0-asr-flash (content changes, e.g. hot_fix).
# 4. Invalid values cost nothing: they return 400 with the engine error code,
#    which is how the ranges above were mapped.
```

## Reference links

Official docs (EN / ZH — ZH pages render more content server-side):

- Qwen-Audio-TTS API hub: <https://www.alibabacloud.com/help/en/model-studio/qwen-audio-tts-api> / <https://help.aliyun.com/zh/model-studio/qwen-audio-tts-api>
- **Client events (primary parameter reference)**: <https://www.alibabacloud.com/help/en/model-studio/qwen-audio-tts-client-events> / <https://help.aliyun.com/zh/model-studio/qwen-audio-tts-client-events>
- Server events (response/task lifecycle): <https://www.alibabacloud.com/help/en/model-studio/qwen-audio-tts-server-events> / <https://help.aliyun.com/zh/model-studio/qwen-audio-tts-server-events>
- Model connection options (WebSocket/AOQ/SDKs): <https://www.alibabacloud.com/help/en/model-studio/qwen-audio-tts-model-access> / <https://help.aliyun.com/zh/model-studio/qwen-audio-tts-model-access>
- Voice list: <https://www.alibabacloud.com/help/en/model-studio/qwen-audio-tts-voice-list> / <https://help.aliyun.com/zh/model-studio/qwen-audio-tts-voice-list>
- SSML usage & restrictions: <https://www.alibabacloud.com/help/en/model-studio/ssml-latex-user-guide#sl01_constraint_h3>
- Instruction control (realtime TTS user guide): <https://www.alibabacloud.com/help/en/model-studio/realtime-tts-user-guide#12884a10929p9>
- Voice cloning/design HTTP API: <https://www.alibabacloud.com/help/en/model-studio/voice-clone-design-http-api#h2-create>
- Different family (do not confuse): Qwen-TTS / qwen3-tts-flash on multimodal-generation, 600-char limit: <https://www.alibabacloud.com/help/en/model-studio/qwen-tts-api>

Tip: fetch with a browser User-Agent header; some pages are JS-rendered and
return only navigation chrome (~3 KB) when the slug is wrong or the page is a stub.
