# cutt

## Local transcription

`tools/transcribe.py` runs speech-to-text locally with
[faster-whisper](https://github.com/SYSTRAN/faster-whisper) — no API key,
works in the Claude Code cloud container (CPU, int8).

```bash
python3 -m venv .venv && .venv/bin/pip install faster-whisper
.venv/bin/python tools/transcribe.py input.mp4 --out words.json --srt draft.srt
```

- Default model `large-v3-turbo`: a 39 s clip transcribes in about 24 s on 4 CPU cores.
  `small` is faster but makes many more mistakes on Mandarin and can skip
  whole stretches of speech over music (it dropped the first 23 s of a 34 s ad).
- Output still needs proofreading (brand names and near-homophones),
  so check it against the script or the burned-in captions before delivery.
- Models download from huggingface.co on first run.

## Vocal / music separation

`tools/separate_vocals.py` splits a track into vocals and instrumental with an
MDX-Net ONNX model (e.g. `Kim_Vocal_2.onnx`, available on Hugging Face), using
only numpy + onnxruntime — no PyTorch.

```bash
.venv/bin/pip install onnxruntime numpy
.venv/bin/python tools/separate_vocals.py input.mp4 Kim_Vocal_2.onnx out/name
# -> out/name_vocals.wav, out/name_inst.wav
```

A 39 s clip separates in about 15 s on 4 CPU cores. Useful for reusing a
finished film's voice-over or music bed, and for transcribing live speech
buried under music (transcribe the `_vocals.wav`).
