# Qwen-Audio-TTS parameters and limits

This article describes the request parameters, service limits, response fields, and
voice catalog for the Qwen-Audio-TTS speech synthesis models in Alibaba Cloud Model
Studio, with a focus on `qwen-audio-3.0-tts-plus` — the model this bridge calls over
the HTTP `SpeechSynthesizer` endpoint. All values are quoted from the official
documentation listed in [References](#references).

## API families

Two separate speech-synthesis API families exist in Model Studio. They use different
models, endpoints, and references, and their parameters and limits are not
interchangeable:

| API family | Models | Endpoint | Official reference |
|---|---|---|---|
| **Qwen-Audio-TTS** (this article, except where noted) | `qwen-audio-3.0-tts-plus`, `qwen-audio-3.0-tts-flash`, `qwen-audio-3.1-tts-flash` | Real-time WebSocket / AOQ; HTTP `SpeechSynthesizer` | Client events, server events, voice list |
| **Qwen-TTS** | `qwen-tts`, `qwen3-tts-flash` | Multimodal-generation | Qwen-TTS API reference |

> [!NOTE]
> The Qwen-Audio-TTS parameter reference documents the real-time (WebSocket / AOQ)
> API, where synthesis parameters are sent in the `payload.parameters` object of the
> `run-task` event and text is sent through `continue-task` events. The HTTP
> `SpeechSynthesizer` endpoint used by this bridge accepts the same parameter names.
> The bridge currently forwards `text`, `voice`, `format`, and `sample_rate` inside
> the `input` object of each request.

## Request parameters

The following parameters are documented for **Qwen-Audio-TTS** speech synthesis tasks
(client-events reference). They do not apply to the Qwen-TTS family, which has its
own, smaller parameter set.

| Parameter | Type | Required | Default value | Valid values | Description |
|---|---|---|---|---|---|
| `model` | string | Yes | — | `qwen-audio-3.0-tts-plus`, `qwen-audio-3.0-tts-flash`, `qwen-audio-3.1-tts-flash` | The model name. |
| `text` | string | Yes | — | Maximum 20,000 characters per message and 200,000 characters cumulatively | The text to synthesize. Supports multilingual mixed input. |
| `text_type` | string | Yes (WebSocket API) | — | `PlainText` | The type of the input text. |
| `voice` | string | Yes | — | See [Voices](#voices) | The voice used for speech synthesis. Voice IDs are case-sensitive and scoped to a model. System voices, cloned voices, and designed voices are supported. |
| `format` | string | No | `mp3` | `pcm`, `wav`, `mp3`, `opus` | The audio encoding format. |
| `sample_rate` | integer | No | `22050` | `8000`, `16000`, `22050`, `24000`, `44100`, `48000` | The audio sample rate, in hertz (Hz). |
| `volume` | integer | No | `50` | `0`–`100` | The volume level. |
| `rate` | float | No | `1.0` | `0.5`–`2.0` | The speech rate. |
| `pitch` | float | No | `1.0` | `0.5`–`2.0` | The pitch of the synthesized speech. |
| `bit_rate` | integer | No | `32` | `6`–`510` | The audio bit rate, in kilobits per second (kbps). Applies only to the `mp3` and `opus` formats. |
| `language_hints` | array[string] | No | Automatic detection | `zh`, `en`, `fr`, `de`, `ja`, `ko`, `ru`, `pt`, `th`, `id`, `vi`, `es`, `it`, `ms`, `fil`, `ar` | The target language for synthesis. Specifying the language improves synthesis quality and corrects digit pronunciation, abbreviation expansion, and symbol reading. The current version processes only the first element of the array. |
| `instruction` | string | No | — | Free text | An instruction that controls dialect, emotion, or voice character during synthesis. |
| `hot_fix` | object | No | — | See below | Pronunciation corrections and text replacements applied before synthesis. |
| `enable_ssml` | boolean | No | `false` | `true`, `false` | Specifies whether to enable SSML. When set to `true`, only one text message is allowed per task. For supported models and voices, see the SSML limitations reference. |
| `word_timestamp_enabled` | boolean | No | `false` | `true`, `false` | Specifies whether to enable word-level timestamps. Available only in streaming output mode. Cloned voices are supported; for system voices, see the voice list. |
| `seed` | integer | No | `0` | `0`–`65535` | A random seed for controlling variation in the synthesis output. Per the documentation, when the model version, text, voice, and other parameters are unchanged, the same seed produces identical results. |
| `enable_aigc_tag` | boolean | No | `false` | `true`, `false` | Specifies whether to embed an AIGC watermark in generated audio files of supported formats (`wav`, `mp3`, `opus`). |
| `aigc_propagator` | string | No | Alibaba Cloud UID | — | Sets the `ContentPropagator` field in the AIGC watermark. Takes effect only when `enable_aigc_tag` is `true`. |
| `aigc_propagate_id` | string | No | The request ID | — | Sets the `PropagateID` field in the AIGC watermark. Takes effect only when `enable_aigc_tag` is `true`. |

The `hot_fix` object supports two fields:

- `pronunciation`: custom pinyin annotations for words whose default pronunciation is
  inaccurate — for example, `{"weather": "tian1 qi4"}`.
- `replace`: text replacements applied before synthesis — for example,
  `{"today": "gold day"}`. The replaced text is used as the actual synthesis input.

## Limits

| Limit | Value | Applies to |
|---|---|---|
| Text per message | 20,000 characters | Qwen-Audio-TTS synthesis tasks |
| Cumulative text per task | 200,000 characters | Qwen-Audio-TTS synthesis tasks |
| Interval between text messages | 23 seconds; longer intervals time out the connection | WebSocket API |
| Input text length | 600 characters | **Qwen-TTS family only** (`qwen-tts`, `qwen3-tts-flash`, multimodal-generation endpoint) — does not apply to Qwen-Audio-TTS |
| Audio URL validity | 24 hours | `output.audio.url` in HTTP non-streaming responses (Qwen-TTS API reference; the same field is returned by the `SpeechSynthesizer` endpoint this bridge uses) |
| Voice availability by region | Multilingual and dialect voices: Beijing and Singapore. All other voice categories: Beijing only | `qwen-audio-3.1-tts-flash` |
| Bridge request timeout | `REQUEST_TIMEOUT` (120 seconds in `.env.example`) | This bridge; long texts may exceed the timeout before synthesis completes |

> [!IMPORTANT]
> The 600-character input limit documented in the Qwen-TTS API reference applies to
> the `qwen-tts` / `qwen3-tts-flash` model family on the multimodal-generation
> endpoint. It does not apply to Qwen-Audio-TTS models such as
> `qwen-audio-3.0-tts-plus`.

## Response and billing

### Real-time API responses (Qwen-Audio-TTS)

Per the server-events reference, the server returns `task-started`,
`result-generated`, and `task-finished` events. Each `result-generated` event carries
sentence-level metadata with the sub-event types `sentence-begin`,
`sentence-synthesis`, and `sentence-end`. One audio frame is transmitted over the
WebSocket binary channel immediately after each `sentence-synthesis` event. The
`sentence-end` event returns the sentence text, word-level timestamps
(`sentence.words`, with `begin_time` and `end_time` in milliseconds), and billing
usage (`usage.characters` — the number of characters in the input text).

### HTTP response fields (Qwen-TTS API reference)

The following fields are documented in the **Qwen-TTS** API reference, which covers
the `qwen-tts` / `qwen3-tts-flash` models on the multimodal-generation endpoint:

- `output.audio.url` — the URL of the complete audio file, valid for 24 hours.
  Returned directly in non-streaming responses; in streaming mode, only in the last
  SSE chunk.
- `output.audio.expires_at` — the URL expiration timestamp.
- `output.audio.data` — base64-encoded audio data; an empty string in non-streaming
  mode.
- `usage.characters` — the number of characters in the input text (documented as
  returned by the Qwen3-TTS-Flash model).

Alibaba Cloud does not publish a separate HTTP reference for Qwen-Audio-TTS. The
`SpeechSynthesizer` endpoint this bridge calls returns audio in the same
`output.audio.url` field, and the bridge downloads the audio from that URL.

## Voices

Voice IDs are case-sensitive. The selected voice must be supported by the selected
model. Do not use voices from another model — an unsupported voice returns
`[cosyvoice:] Engine error [411]: TTS speak operation failed`. The text must use a
language supported by the voice; otherwise, pronunciation errors or unnatural speech
might occur.

### System voices for qwen-audio-3.0-tts-plus

| Voice parameter | Name | Gender | Age | Character | Languages |
|---|---|---|---|---|---|
| `longanlingxin` | Long An Ling Xin | Female | 25 | Warm and empathetic | Chinese (Mandarin), English |
| `longanlufeng` | Long An Lu Feng | Male | 25 | Bright and cheerful | Chinese (Mandarin), English |

### Base voices for qwen-audio-3.0-tts-plus

In addition to the system voices, `qwen-audio-3.0-tts-plus` and
`qwen-audio-3.0-tts-flash` each provide more than 500 base voices generated through
voice cloning. Base voices are called in the same way as system voices and follow the
naming pattern `qwen-audio-3.0-tts-{plus|flash}-{voice suffix}`. The same suffix in
both models maps to the same set of preview audio files.

The official base-voice list for `qwen-audio-3.0-tts-plus` (Excel) contains 597
voices: 589 Chinese voices and 8 English voices. The English voices are:

| Voice parameter | Name | Gender | Age | Character | Use case |
|---|---|---|---|---|---|
| `qwen-audio-3.0-tts-plus-loongjameszhao` | James Zhao | Male | 28 | News-broadcast voice (used in this repository's examples) | Daily conversation |
| `qwen-audio-3.0-tts-plus-loongolivialin` | Olivia Lin | Female | 28 | Gentle, intellectual | Emotional companionship |
| `qwen-audio-3.0-tts-plus-loonglunawang` | Luna Wang | Female | 21 | Smart-assistant voice | Daily conversation |
| `qwen-audio-3.0-tts-plus-loongnorahu` | Nora Hu | Female | 25 | Lively, spirited | Daily conversation |
| `qwen-audio-3.0-tts-plus-loongivyhu` | Ivy Hu | Female | 54 | Confident, composed | Speech and recitation |
| `qwen-audio-3.0-tts-plus-loongalexanderhu` | Alexander Hu | Male | 27 | Retro-cassette voice | Daily conversation |
| `qwen-audio-3.0-tts-plus-loongadriangao` | Adrian Gao | Male | 22 | Calm, steady | Audiobook reading |
| `qwen-audio-3.0-tts-plus-loongryanma` | Ryan Ma | Male | 38 | Electronic, trendy | Daily conversation |

> [!NOTE]
> Alibaba Cloud recommends prioritizing system voices for more consistent synthesis
> results. Base voices offer more options; listen to the previews and evaluate them
> before use.

### Voices of sibling models

Voices of other models must not be passed to `qwen-audio-3.0-tts-plus`.

- `qwen-audio-3.0-tts-flash` system voices: `longanfengyue`, `longanyuanfei`,
  `longanlingxi`, `longanxiaoxin` (Chinese and English); `longanhuan_v3.6`,
  `longjielidou_v3.6`, `longpaopao_v3.6`, `longhuohuo_v3.6`, `longchuanshu_v3.6`
  (Chinese and English); `loongmary`, `loongjohn` (English), and `loongeva_v3.6`
  (English). Its base voices follow the pattern `qwen-audio-3.0-tts-flash-{suffix}`.
- `qwen-audio-3.1-tts-flash` voices: four multilingual and dialect voices
  (`longanhuan_v3.1`, `longanlingxin_v3.1`, `longanfengyue_v3.1`,
  `xunanchuan_v3.1`) that support Chinese dialects (Shanghainese, Cantonese,
  Northeastern, Chongqing, Shaanxi, Yunnan, Ningbo, Gansu) plus Japanese, Korean,
  French, German, Portuguese, Italian, Vietnamese, and Indonesian; premium Chinese
  voices (such as `yuxiaoyun_v3.1` and `qiaoxiaojiao_v3.1`); and 15 English voices
  with British or American accents (`Emily_v3.1`, `Luna_v3.1`, `Eric_v3.1`,
  `Luca_v3.1`, `Abby_v3.1`, `Annie_v3.1`, `Ava_v3.1`, `Beth_v3.1`, `Betty_v3.1`,
  `Cally_v3.1`, `Cindy_v3.1`, `Donna_v3.1`, `Andy_v3.1`, `Brian_v3.1`,
  `David_v3.1`).

## References

Official Alibaba Cloud Model Studio documentation (English and Chinese editions):

- Qwen-Audio-TTS API hub:
  [EN](https://www.alibabacloud.com/help/en/model-studio/qwen-audio-tts-api) ·
  [ZH](https://help.aliyun.com/zh/model-studio/qwen-audio-tts-api)
- Client events — request parameter reference:
  [EN](https://www.alibabacloud.com/help/en/model-studio/qwen-audio-tts-client-events) ·
  [ZH](https://help.aliyun.com/zh/model-studio/qwen-audio-tts-client-events)
- Server events — response events and usage:
  [EN](https://www.alibabacloud.com/help/en/model-studio/qwen-audio-tts-server-events) ·
  [ZH](https://help.aliyun.com/zh/model-studio/qwen-audio-tts-server-events)
- Model connection options (WebSocket, AOQ, SDKs):
  [EN](https://www.alibabacloud.com/help/en/model-studio/qwen-audio-tts-model-access) ·
  [ZH](https://help.aliyun.com/zh/model-studio/qwen-audio-tts-model-access)
- Voice list:
  [EN](https://www.alibabacloud.com/help/en/model-studio/qwen-audio-tts-voice-list) ·
  [ZH](https://help.aliyun.com/zh/model-studio/qwen-audio-tts-voice-list)
- SSML usage and limitations:
  [EN](https://www.alibabacloud.com/help/en/model-studio/ssml-latex-user-guide#sl01_constraint_h3)
- Instruction control (real-time TTS user guide):
  [EN](https://www.alibabacloud.com/help/en/model-studio/realtime-tts-user-guide#12884a10929p9)
- Voice cloning and voice design HTTP API:
  [EN](https://www.alibabacloud.com/help/en/model-studio/voice-clone-design-http-api#h2-create)
- Qwen-TTS API reference (`qwen-tts` / `qwen3-tts-flash`, multimodal-generation
  endpoint; the source of the 600-character limit and the HTTP audio response
  fields):
  [EN](https://www.alibabacloud.com/help/en/model-studio/qwen-tts-api)

Downloads:

- [qwen-audio-3.0-tts-plus base voices (Excel, 597 rows)](https://help-static-aliyun-doc.aliyuncs.com/file-manage-files/zh-CN/20260723/ydwqqz/qwen-audio-3.0-tts-plus%E5%9F%BA%E7%A1%80%E9%9F%B3%E8%89%B2.xlsx)
- [qwen-audio-3.0-tts-flash base voices (Excel)](https://help-static-aliyun-doc.aliyuncs.com/file-manage-files/zh-CN/20260723/thosjr/qwen-audio-3.0-tts-flash%E5%9F%BA%E7%A1%80%E9%9F%B3%E8%89%B2.xlsx)
- [Base voice preview audio package (ZIP, shared by plus and flash)](https://help-static-aliyun-doc.aliyuncs.com/file-manage-files/zh-CN/20260720/tuuuqo/%E5%9F%BA%E7%A1%80%E9%9F%B3%E8%89%B2%E8%AF%95%E5%90%AC%E9%9F%B3%E9%A2%91%E5%8C%85.zip)
