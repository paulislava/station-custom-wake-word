#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Отбор лучшей эпохи по тестовым наборам (быстро, через Keras-модель на предвычисленных признаках).

  python sonet_select.py --run "$WAKEWORD_ROOT"/sonet/run3 [--thr 0.69]
Тестовые наборы берутся из $WAKEWORD_ROOT/sonet (если не задан - ./work/sonet).
Печатает таблицу по эпохам: recall «сонет» и «алиса» на sonet/test, ложные на файлах, срабатывания
на длинном потоке; выбирает лучшую по score = recall_sonet - 0.02*FA_files - 0.01*events + 0.3*recall_alisa
и экспортирует её как model.tflite (+ model_sonet2 / model_alisa2) в каталог прогона.
"""
import argparse, glob, os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sonet_train as T
from sonet_eval import read_wav

ROOT = os.path.join(os.path.abspath(os.path.expanduser(os.environ.get("WAKEWORD_ROOT", "./work"))), "sonet")


def feats(path, warm=2.0, rng=np.random.default_rng(0)):
    x = read_wav(path)
    x = np.concatenate([rng.normal(0, 30, int(warm * T.SR)).astype(np.float32), x, rng.normal(0, 30, T.SR // 2).astype(np.float32)])
    f = T.fbank(x)
    return np.concatenate([f, f], axis=1)


def smooth_max(p, warm_frames=180):
    sm = np.convolve(p, np.ones(11) / 11, mode="same") if len(p) >= 11 else p
    sm[:warm_frames] = 0
    return sm


def events(sm, thr):
    ev = 0; last = -10**9
    for i in np.where(sm >= thr)[0]:
        if i - last >= 100: ev += 1
        last = i
    return ev


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--thr", type=float, default=0.69)
    ap.add_argument("--kernels", default="9,13,17,21,25,29,33"); ap.add_argument("--width", type=int, default=64)
    a = ap.parse_args()
    import tensorflow as tf
    kernels = [int(k) for k in a.kernels.split(",")]
    model = T.build(tf, kernels, a.width, len(T.CLASSES))
    sets = {
        "pos_sonet": sorted(glob.glob(f"{ROOT}/test/pos_sonet/*.wav")),
        "pos_alisa": sorted(glob.glob(f"{ROOT}/test/pos_alisa/*.wav")),
        "neg": sorted(glob.glob(f"{ROOT}/test/neg_sent/*.wav")),
    }
    F = {k: [feats(p) for p in v] for k, v in sets.items()}
    long_x = read_wav(f"{ROOT}/test_long_neg2.wav"); fl = T.fbank(long_x); FL = np.concatenate([fl, fl], axis=1)
    print("features ready", {k: len(v) for k, v in F.items()}, flush=True)

    def predict_max(fs, cls):
        out = []
        for f in fs:
            lg = model.predict(f[None], verbose=0)[0]
            e = np.exp(lg - lg.max(-1, keepdims=True)); p = e / e.sum(-1, keepdims=True)
            out.append(smooth_max(p[:, cls]).max())
        return np.array(out)

    rows = []
    weights = sorted(glob.glob(os.path.join(a.run, "ep*.weights.h5")))
    for w in weights:
        model.load_weights(w)
        rs = predict_max(F["pos_sonet"], 2); ra = predict_max(F["pos_alisa"], 1)
        ns = predict_max(F["neg"] + F["pos_alisa"], 2); na = predict_max(F["neg"] + F["pos_sonet"], 1)
        lg = model.predict(FL[None], verbose=0)[0]; e = np.exp(lg - lg.max(-1, keepdims=True)); p = e / e.sum(-1, keepdims=True)
        ev = events(smooth_max(p[:, 2], 0), a.thr)
        rec_s = float((rs >= a.thr).mean()); rec_a = float((ra >= a.thr).mean()); fa_s = int((ns >= a.thr).sum()); fa_a = int((na >= a.thr).sum())
        score = rec_s - 0.02 * fa_s - 0.01 * ev + 0.3 * (rec_a - 0.02 * fa_a)
        rows.append((score, w, rec_s, fa_s, ev, rec_a, fa_a, float(np.median(rs))))
        print(f"{os.path.basename(w)}: сонет recall {rec_s:.2f} (медиана {np.median(rs):.2f}) FA {fa_s} поток {ev}/20мин | алиса recall {rec_a:.2f} FA {fa_a} | score {score:.3f}", flush=True)
    best = max(rows, key=lambda r: r[0])
    print("BEST:", os.path.basename(best[1]), f"score {best[0]:.3f}")
    model.load_weights(best[1])
    T.export_streaming(tf, model, kernels, a.width, len(T.CLASSES), os.path.join(a.run, "model.tflite"))
    T.export_streaming(tf, model, kernels, a.width, len(T.CLASSES), os.path.join(a.run, "model_sonet2.tflite"), fold=[[0, 1], [2]])
    T.export_streaming(tf, model, kernels, a.width, len(T.CLASSES), os.path.join(a.run, "model_alisa2.tflite"), fold=[[0, 2], [1]])
    json.dump({"best": os.path.basename(best[1]), "rows": [[r[0], os.path.basename(r[1])] + list(r[2:]) for r in rows]}, open(os.path.join(a.run, "select.json"), "w"), indent=1)
    print("exported model.tflite from", os.path.basename(best[1]))


if __name__ == "__main__":
    main()
