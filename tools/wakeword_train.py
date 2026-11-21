#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Обобщённое обучение стримингового споттера для произвольного wake word.

Импортирует ML-инфраструктуру (признаки, модель, экспорт, верификацию) из sonet_train.py
и заменяет классы/виды данных на конфигурацию из wakeword_config.py.

Запуск (venv312):
  python wakeword_train.py --word votchi --epochs 40
  python wakeword_train.py --word siesta --epochs 30

Данные по умолчанию: $WAKEWORD_ROOT/<word>/raw → $WAKEWORD_ROOT/<word>/run1
"""
import argparse, json, os, random, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
# Импортируем всё из sonet_train, потом патчим глобалы
import sonet_train as T
from wakeword_config import get_config


def patch_globals(word):
    """Заменить глобальные константы sonet_train на конфигурацию слова."""
    cfg = get_config(word)
    T.CLASSES = cfg["classes"]
    T.KIND2CLS = cfg["kind2cls"]
    # Имя доли: было SONET_SHARE, используем общий WORD_SHARE
    word_share = cfg["word_share"]
    # Патчим SONET_SHARE (sonet_train uses this name internally)
    T.SONET_SHARE = word_share
    return cfg


def main():
    ap = argparse.ArgumentParser(description="Обучение wake word модели")
    ap.add_argument("--word", required=True, choices=["siesta", "votchi", "miku", "smith"],
                    help="Wake word key")
    ap.add_argument("--raw", default=None, help="Каталог с manifest.jsonl (по умолчанию: $WAKEWORD_ROOT/<word>/raw)")
    ap.add_argument("--out", default=None, help="Выходной каталог (по умолчанию: $WAKEWORD_ROOT/<word>/run1)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--holdout-voice", default="piper-irina")
    ap.add_argument("--kernels", default="9,13,17,21,25,29,33")
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--workers", type=int, default=5, help="процессов сборки батчей (0 = в основном процессе)")
    ap.add_argument("--word-share", type=float, default=None, help="Доля кастомного слова (по умолчанию из конфига)")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--init-weights", default="", help="Начать с весов из другого прогона")
    ap.add_argument("--extra-raw", default="", help="Доп. manifest.jsonl, через запятую")
    a = ap.parse_args()

    cfg = patch_globals(a.word)
    word_share = a.word_share if a.word_share is not None else cfg["word_share"]
    T.SONET_SHARE = word_share

    raw = a.raw or os.path.join(T.ROOT, a.word, "raw")
    out = a.out or os.path.join(T.ROOT, a.word, "run1")

    os.makedirs(out, exist_ok=True)
    random.seed(a.seed); np.random.seed(a.seed)
    import tensorflow as tf
    tf.random.set_seed(a.seed)

    kernels = [int(k) for k in a.kernels.split(",")]
    items = T.load_manifest(raw)
    for extra in [e for e in a.extra_raw.split(",") if e]:
        items += T.load_manifest(extra)

    print(f"[wakeword] word={a.word} classes={T.CLASSES} share={word_share}", flush=True)
    print(f"[wakeword] raw={raw} items={len(items)}", flush=True)

    train_items = [it for it in items if it["voice"] != a.holdout_voice]
    val_items = [it for it in items if it["voice"] == a.holdout_voice] or items[:200]
    vcomp = T.Composer(val_items, a.seed + 100)
    Xv, Yv = vcomp.batch(256)

    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    pool = ctx.Pool(a.workers, initializer=T._init_worker, initargs=(train_items, a.seed, word_share)) if a.workers > 0 else None
    comp = None if pool else T.Composer(train_items, a.seed)

    model = T.build(tf, kernels, a.width, len(T.CLASSES))
    model.summary(print_fn=lambda s: print(s, flush=True))
    if a.init_weights:
        model.load_weights(a.init_weights); print("init from", a.init_weights, flush=True)

    opt = tf.keras.optimizers.Adam(a.lr)
    cw = tf.constant([1.0, 3.0, 3.0])
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none")

    @tf.function
    def step(X, Y):
        with tf.GradientTape() as tape:
            lg = model(X, training=True)
            l = loss_fn(Y, lg) * tf.gather(cw, Y)
            l = tf.reduce_mean(l)
        g = tape.gradient(l, model.trainable_variables)
        opt.apply_gradients(zip(g, model.trainable_variables))
        return l

    def evaluate(X, Y):
        p = tf.nn.softmax(model.predict(X, verbose=0), -1).numpy()
        res = {}
        for c in (1, 2):
            has = (Y == c).any(axis=1)
            det = (p[:, :, c].max(axis=1) > 0.5)
            res[T.CLASSES[c]] = dict(
                recall=float((det & has).sum() / max(1, has.sum())),
                fa=float((det & ~has).sum() / max(1, (~has).sum())))
        return res

    best = -1
    log = open(os.path.join(out, "train.log"), "a")
    for ep in range(a.epochs):
        t0 = time.time(); ls = []
        if pool:
            for X, Y in pool.imap_unordered(T._make_batch, [(ep * 100003 + i, a.batch) for i in range(a.steps)]):
                ls.append(float(step(tf.constant(X), tf.constant(Y))))
        else:
            for _ in range(a.steps):
                X, Y = comp.batch(a.batch)
                ls.append(float(step(tf.constant(X), tf.constant(Y))))
        ev = evaluate(Xv, Yv)
        # Score by custom word (class 2) primarily
        word_name = T.CLASSES[2]
        score = ev[word_name]["recall"] - 2 * ev[word_name]["fa"] + 0.5 * (ev["алиса"]["recall"] - 2 * ev["алиса"]["fa"])
        msg = f"ep {ep+1}/{a.epochs} loss {np.mean(ls):.4f} val {json.dumps(ev, ensure_ascii=False)} {time.time()-t0:.0f}s"
        print(msg, flush=True); log.write(msg + "\n"); log.flush()
        model.save_weights(os.path.join(out, f"ep{ep+1:02d}.weights.h5"))
        if score > best:
            best = score; model.save_weights(os.path.join(out, "best.weights.h5"))
        if (ep + 1) % 5 == 0:
            opt.learning_rate.assign(float(opt.learning_rate) * 0.7)

    model.load_weights(os.path.join(out, "best.weights.h5"))
    tfl_path = os.path.join(out, "model.tflite")
    m = T.export_streaming(tf, model, kernels, a.width, len(T.CLASSES), tfl_path)
    err = T.verify(tf, model, m, kernels, a.width, Xv[0])
    print(f"exported {tfl_path} ({os.path.getsize(tfl_path)} bytes), max diff {err:.2e}", flush=True)

    # words.txt
    with open(os.path.join(out, "words.txt"), "w") as f:
        for i, c in enumerate(T.CLASSES): f.write(f"{c} {i}\n")

    # Запасной вариант: только кастомное слово (2 класса)
    export_path_2cls = os.path.join(out, f"model_{a.word}2.tflite")
    T.export_streaming(tf, model, kernels, a.width, len(T.CLASSES), export_path_2cls, fold=[[0, 1], [2]])
    with open(os.path.join(out, f"words_{a.word}2.txt"), "w") as f:
        f.write(f"filler 0\n{cfg['word_ru']} 1\n")

    # И только алиса (для сверки)
    T.export_streaming(tf, model, kernels, a.width, len(T.CLASSES), os.path.join(out, "model_alisa2.tflite"), fold=[[0, 2], [1]])

    # meta.json
    json.dump({"kernels": kernels, "width": a.width, "classes": T.CLASSES,
                "word": a.word, "val": evaluate(Xv, Yv)},
               open(os.path.join(out, "meta.json"), "w"), ensure_ascii=False, indent=1)

    print(f"[wakeword] done: {out}", flush=True)


if __name__ == "__main__":
    main()
