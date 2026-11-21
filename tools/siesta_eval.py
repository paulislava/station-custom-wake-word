#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Оценка споттера (любого tflite с сигнатурой frames/state_i -> logits/state_i_new) на wav-файлах.

  python siesta_eval.py MODEL.tflite --cls 2 wav1.wav dir/ ...   # cls: индекс класса (2 = сиеста, 1 = алиса)
  python siesta_eval.py MODEL.tflite --cls 2 --thr 0.69 --labels pos.txt neg.txt  # метрики

Для каждого файла: 2 с тихого шума в начале (разогрев), признаки как в обучении (log-mel 40, два
канала = копия), сглаживание 11 кадров, пик и время пика. Для штатной модели Яндекса (2 класса)
--cls 1.
"""
import argparse, glob, os, sys, wave
import numpy as np
from ai_edge_litert.interpreter import Interpreter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from siesta_train import fbank, SR, CHUNK  # noqa: E402


def read_wav(p):
    w = wave.open(p, "rb")
    x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32)
    if w.getnchannels() == 2:
        x = x.reshape(-1, 2).mean(axis=1)
    if w.getframerate() != SR:
        n = int(len(x) * SR / w.getframerate()); x = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x)
    return x.astype(np.float32)


class Spotter:
    def __init__(self, path):
        self.it = Interpreter(model_path=path); self.it.allocate_tensors()
        self.sig = self.it.get_signature_runner("serving_default")
        self.ins = self.sig.get_input_details()

    def init_state(self):
        st = {}
        for k, d in self.ins.items():
            if k.startswith("state"):
                zp = d.get("quantization_parameters", {}).get("zero_points", [])
                st[k] = np.full(d["shape"], int(zp[0]) if len(zp) else 0, dtype=d["dtype"])
        return st

    def run(self, feats80, cls):
        st = self.init_state(); probs = []
        for i in range(0, len(feats80) // CHUNK * CHUNK, CHUNK):
            out = self.sig(frames=feats80[i:i + CHUNK][None, None].astype(np.float32), **st)
            st = {k: out[k + "_new"] for k in st}
            lg = out["logits"][0, 0]
            e = np.exp(lg - lg.max(axis=1, keepdims=True)); p = e / e.sum(axis=1, keepdims=True)
            probs.extend(p[:, cls].tolist())
        probs = np.array(probs)
        sm = np.convolve(probs, np.ones(11) / 11, mode="same") if len(probs) >= 11 else probs
        return sm


def score_file(sp, path, cls, warm=2.0):
    x = read_wav(path)
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.normal(0, 30, int(warm * SR)).astype(np.float32), x, rng.normal(0, 30, SR // 2).astype(np.float32)])
    f = fbank(x); f80 = np.concatenate([f, f], axis=1)
    sm = sp.run(f80, cls)
    sm[: int(warm * 100) - 20] = 0  # разогрев не считаем
    return float(sm.max()), float(sm.argmax() * 0.01 - warm)


def expand(paths):
    out = []
    for p in paths:
        if os.path.isdir(p):
            out += sorted(glob.glob(os.path.join(p, "**", "*.wav"), recursive=True))
        else:
            out += sorted(glob.glob(p))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("paths", nargs="*")
    ap.add_argument("--cls", type=int, default=2); ap.add_argument("--thr", type=float, default=0.69)
    ap.add_argument("--pos", nargs="*", default=[], help="файлы/папки, где слово есть")
    ap.add_argument("--neg", nargs="*", default=[], help="файлы/папки, где слова нет")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--events", nargs="*", default=[], help="длинные файлы: считать срабатывания (пики >= thr с разрывом >= 1 с)")
    a = ap.parse_args()
    sp = Spotter(a.model)
    for p in expand(a.events):
        x = read_wav(p); f = fbank(x); sm = sp.run(np.concatenate([f, f], axis=1), a.cls)
        ev = []; last = -10**9
        for i in np.where(sm >= a.thr)[0]:
            if i - last >= 100: ev.append(i)
            last = i
        hrs = len(x) / SR / 3600
        print(f"{p}: {len(x)/SR/60:.1f} мин, срабатываний >= {a.thr}: {len(ev)} = {len(ev)/max(hrs,1e-9):.1f}/час; max {sm.max():.3f}; времена: " + " ".join(f"{i/100:.1f}" for i in ev[:20]))
    for p in expand(a.paths):
        m, t = score_file(sp, p, a.cls)
        print(f"{m:.3f} @{t:5.2f}s  {p}")
    if a.pos or a.neg:
        pos = [score_file(sp, p, a.cls)[0] for p in expand(a.pos)]
        neg = [score_file(sp, p, a.cls)[0] for p in expand(a.neg)]
        negdur = sum(len(read_wav(p)) / SR for p in expand(a.neg)) / 3600 if neg else 0
        tp = sum(m >= a.thr for m in pos); fp = sum(m >= a.thr for m in neg)
        print(f"thr {a.thr}: recall {tp}/{len(pos)} = {tp/max(1,len(pos)):.3f}; false alarms {fp}/{len(neg)} файлов"
              + (f" = {fp/negdur:.1f}/час речи" if negdur else ""))
        if pos: print("pos quantiles 5/50/95%:", np.percentile(pos, [5, 50, 95]).round(3))
        if neg: print("neg quantiles 50/95/99/max:", np.percentile(neg, [50, 95, 99]).round(3), round(max(neg), 3))
