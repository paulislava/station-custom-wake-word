#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Прогон wav через активационный tflite-споттер Станции (Kaldi-подобный log-mel fbank + CMN).

Признаки прикидочные: 16 кГц, окно 25 мс, шаг 10 мс, preemph 0.97, окно povey, FFT 512,
40 mel-полос 20 Гц..8 кГц, log, скользящий CMN. Второй канал (bnr) недоступен снаружи,
поэтому пробуем три варианта раскладки 80 признаков: [x,x], [x,0], [0,x].
"""
import sys, wave, numpy as np
from ai_edge_litert.interpreter import Interpreter

SR, FRAME, SHIFT, NFFT, NMEL = 16000, 400, 160, 512, 40


def read_wav(p):
    w = wave.open(p, "rb")
    assert w.getframerate() == SR and w.getnchannels() == 1 and w.getsampwidth() == 2, (w.getparams())
    x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32)
    return x  # масштаб int16, как в Kaldi


def mel_matrix():
    def hz2mel(f): return 1127.0 * np.log(1.0 + f / 700.0)
    def mel2hz(m): return 700.0 * (np.exp(m / 1127.0) - 1.0)
    lo, hi = hz2mel(20.0), hz2mel(SR / 2.0)
    pts = mel2hz(np.linspace(lo, hi, NMEL + 2))
    bins = np.floor((NFFT + 1) * pts / SR).astype(int)
    M = np.zeros((NMEL, NFFT // 2 + 1), dtype=np.float32)
    for i in range(NMEL):
        a, b, c = bins[i], bins[i + 1], bins[i + 2]
        for k in range(a, b):
            M[i, k] = (k - a) / max(1, b - a)
        for k in range(b, c):
            M[i, k] = (c - k) / max(1, c - b)
    return M


def fbank(x):
    n = 1 + max(0, (len(x) - FRAME) // SHIFT)
    idx = np.arange(FRAME)[None, :] + SHIFT * np.arange(n)[:, None]
    fr = x[idx].copy()
    fr -= fr.mean(axis=1, keepdims=True)                      # remove DC
    fr[:, 1:] -= 0.97 * fr[:, :-1]; fr[:, 0] *= (1 - 0.97)    # preemphasis
    win = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(FRAME) / (FRAME - 1))) ** 0.85  # povey
    fr *= win
    spec = np.abs(np.fft.rfft(fr, NFFT)) ** 2
    mel = spec @ mel_matrix().T
    return np.log(np.maximum(mel, 1e-10)).astype(np.float32)


def cmn(f, window=600, latency=50):
    out = np.empty_like(f)
    for i in range(len(f)):
        a, b = max(0, i + latency - window), min(len(f), i + latency + 1)
        out[i] = f[i] - f[a:b].mean(axis=0)
    return out


def run(model, feats80):
    it = Interpreter(model_path=model); it.allocate_tensors()
    sig = it.get_signature_runner("serving_default")
    ins = sig.get_input_details()
    state = {}
    for k, d in ins.items():
        if not k.startswith("state"):
            continue
        zp = d.get("quantization_parameters", {}).get("zero_points", [])
        zp = int(zp[0]) if len(zp) else 0
        state[k] = np.full(d["shape"], zp, dtype=d["dtype"])   # «ноль» квантованного тензора = zero point
    probs = []
    n = len(feats80) // 6 * 6
    for i in range(0, n, 6):
        chunk = feats80[i:i + 6][None, None, :, :].astype(np.float32)
        out = sig(frames=chunk, **state)
        state = {k: out[k + "_new"] for k in state}
        lg = out["logits"][0, 0]                       # [6,2]
        e = np.exp(lg - lg.max(axis=1, keepdims=True)); p = e / e.sum(axis=1, keepdims=True)
        probs.extend(p[:, 1].tolist())
    probs = np.array(probs)
    sm = np.convolve(probs, np.ones(11) / 11, mode="same") if len(probs) >= 11 else probs
    return probs.max() if len(probs) else 0.0, sm.max() if len(sm) else 0.0, (sm.argmax() * 0.01 if len(sm) else 0)


if __name__ == "__main__":
    model = sys.argv[1]
    for wavp in sys.argv[2:]:
        x = read_wav(wavp)
        raw = fbank(x); f = cmn(raw)
        variants = {"cmn[x,x]": np.concatenate([f, f], 1), "raw[x,x]": np.concatenate([raw, raw], 1)}
        res = {k: run(model, v) for k, v in variants.items()}
        print(f"{wavp}: {len(x)/SR:.2f}s, frames={len(f)} | " + " | ".join(f"{k} max={r[0]:.3f} smooth={r[1]:.3f}@{r[2]:.2f}s" for k, r in res.items()))
