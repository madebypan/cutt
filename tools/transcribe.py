#!/usr/bin/env python3
"""Local speech-to-text with faster-whisper (no API key).

Usage:
  python tools/transcribe.py <video-or-audio> [--model large-v3-turbo] [--lang zh] [--out words.json] [--srt out.srt]

Decodes audio with the ffmpeg CLI and passes a numpy array to the model,
which avoids the PyAV incompatibility (`av.open() got an unexpected keyword
argument 'metadata_errors'`) seen with faster-whisper 1.2.1.
"""
import argparse, json, subprocess, sys
import numpy as np
from faster_whisper import WhisperModel

SR = 16000

def load_audio(path):
    pcm = subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-i", path,
                          "-vn", "-ac", "1", "-ar", str(SR), "-f", "s16le", "-"],
                         check=True, stdout=subprocess.PIPE).stdout
    return np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768

def ts(t):
    ms = int(round(t * 1000)); h, ms = divmod(ms, 3600000); m, ms = divmod(ms, 60000); s, ms = divmod(ms, 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--model", default="large-v3-turbo")
    ap.add_argument("--lang", default="zh")
    ap.add_argument("--prompt", default="以下是繁體中文的句子。")
    ap.add_argument("--out", help="write segments + word timestamps as JSON")
    ap.add_argument("--srt", help="write segment-level SRT")
    a = ap.parse_args()

    model = WhisperModel(a.model, device="cpu", compute_type="int8")
    segs, _ = model.transcribe(load_audio(a.input), language=a.lang, word_timestamps=True,
                               vad_filter=False, initial_prompt=a.prompt)
    out = []
    for s in segs:
        out.append({"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip(),
                    "words": [{"w": w.word, "s": round(w.start, 2), "e": round(w.end, 2),
                               "p": round(w.probability, 2)} for w in s.words]})
        print(f"{s.start:7.2f}-{s.end:7.2f}  {s.text.strip()}", flush=True)
    if a.out:
        json.dump(out, open(a.out, "w"), ensure_ascii=False, indent=1)
    if a.srt:
        with open(a.srt, "w") as f:
            for i, s in enumerate(out, 1):
                f.write(f"{i}\n{ts(s['start'])} --> {ts(s['end'])}\n{s['text']}\n\n")

if __name__ == "__main__":
    sys.exit(main())
