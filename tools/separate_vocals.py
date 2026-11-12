#!/usr/bin/env python3
"""Split a track into vocals / instrumental with an MDX-Net ONNX model (e.g. Kim_Vocal_2), numpy + onnxruntime only.

Usage: python tools/separate_vocals.py <input> <model.onnx> <out_prefix>
Writes <out_prefix>_vocals.wav and <out_prefix>_inst.wav (44.1 kHz stereo).
Defaults match Kim_Vocal_2: n_fft 7680, hop 1024, dim_f 3072, dim_t 256, compensate 1.009.
"""
import sys, subprocess, wave
import numpy as np
import onnxruntime as ort

SR, N_FFT, HOP, DIM_F, DIM_T, COMP = 44100, 7680, 1024, 3072, 256, 1.009
CHUNK = HOP * (DIM_T - 1)
TRIM = N_FFT // 2
GEN = CHUNK - 2 * TRIM
WIN = np.hanning(N_FFT + 1)[:-1].astype(np.float32)  # periodic hann, like torch.hann_window


def load(path):
    pcm = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", path, "-vn", "-ac", "2", "-ar", str(SR),
                          "-f", "f32le", "-"], check=True, stdout=subprocess.PIPE).stdout
    return np.frombuffer(pcm, np.float32).reshape(-1, 2).T.copy()


def stft(x):  # x: [C, CHUNK] -> [C, F, T] complex, center=True reflect pad
    x = np.pad(x, ((0, 0), (N_FFT // 2, N_FFT // 2)), mode="reflect")
    idx = np.arange(DIM_T)[:, None] * HOP + np.arange(N_FFT)[None, :]
    return np.fft.rfft(x[:, idx] * WIN, axis=-1).transpose(0, 2, 1)


def istft(X):  # [C, F, T] -> [C, CHUNK]
    fr = np.fft.irfft(X.transpose(0, 2, 1), n=N_FFT, axis=-1) * WIN
    L = N_FFT + HOP * (DIM_T - 1)
    out = np.zeros((X.shape[0], L), np.float64); wsum = np.zeros(L)
    for t in range(DIM_T):
        out[:, t * HOP:t * HOP + N_FFT] += fr[:, t]; wsum[t * HOP:t * HOP + N_FFT] += WIN ** 2
    out /= np.maximum(wsum, 1e-8)
    return out[:, N_FFT // 2:N_FFT // 2 + CHUNK]


def save(path, x):
    x = np.clip(x.T, -1, 1); w = wave.open(path, "wb"); w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
    w.writeframes((x * 32767).astype(np.int16).tobytes()); w.close()


def main():
    src, model, prefix = sys.argv[1:4]
    sess = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
    mix = load(src); n = mix.shape[1]
    pad = GEN - n % GEN
    x = np.pad(mix, ((0, 0), (TRIM, TRIM + pad)))
    voc = np.zeros_like(x[:, :n + pad])
    for i in range(0, n + pad, GEN):
        spec = stft(x[:, i:i + CHUNK])[:, :DIM_F]                      # [2, 3072, 256]
        inp = np.stack([spec.real, spec.imag], 1).reshape(1, 4, DIM_F, DIM_T).astype(np.float32)
        out = sess.run(None, {sess.get_inputs()[0].name: inp})[0].reshape(2, 2, DIM_F, DIM_T)
        Y = np.zeros((2, N_FFT // 2 + 1, DIM_T), np.complex64); Y[:, :DIM_F] = out[:, 0] + 1j * out[:, 1]
        voc[:, i:i + GEN] = istft(Y)[:, TRIM:TRIM + GEN]
    voc = voc[:, :n] * COMP
    save(prefix + "_vocals.wav", voc); save(prefix + "_inst.wav", mix - voc)
    print("ok", n / SR, "s")


if __name__ == "__main__":
    main()
