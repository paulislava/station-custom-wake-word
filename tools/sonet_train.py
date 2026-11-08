#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Обучение стримингового споттера «Сонет»/«Алиса» и экспорт в TFLite с сигнатурой Станции.

Сигнатура экспорта (как у штатной модели Мини 2):
  вход  frames  float32 [1,1,6,80]   (6 кадров по 10 мс, 80 = 40 log-mel x 2 канала)
  входы state_i float32 [1, k_i-1, C]  (хвост входа каждой временной свёртки)
  выход logits  float32 [1,1,6,NCLS]   (0 filler, 1 алиса, 2 сонет)
  выходы state_i_new

Запуск (venv312):
  python sonet_train.py --raw "$WAKEWORD_ROOT"/sonet/raw --out "$WAKEWORD_ROOT"/sonet/run1 --epochs 30
Каталоги по умолчанию - внутри $WAKEWORD_ROOT (если не задан - ./work).
"""
import argparse, json, os, random, sys, time, wave
import numpy as np

ROOT = os.path.abspath(os.path.expanduser(os.environ.get("WAKEWORD_ROOT", "./work")))  # данные, голоса, прогоны
SR, FRAME, SHIFT, NFFT, NMEL = 16000, 400, 160, 512, 40
CLASSES = ["filler", "алиса", "сонет"]
KIND2CLS = {"pos_alisa": 1, "pos_sonet": 2}
CHUNK = 6            # кадров на вызов модели
SEQ = 240            # кадров в обучающей последовательности (2.4 с)
POS_BEFORE, POS_AFTER = 12, 4   # окно положительных меток вокруг конца слова (кадры)
SONET_SHARE = 0.65              # доля «сонет» среди позитивов (остальное «алиса»); --sonet-share


# ---------------- признаки (как в kws_probe.py) ----------------
def mel_matrix(alpha=1.0):
    """alpha != 1 - растяжение частотной шкалы (VTLP, имитация другого диктора), только для обучения."""
    def hz2mel(f): return 1127.0 * np.log(1.0 + f / 700.0)
    def mel2hz(m): return 700.0 * (np.exp(m / 1127.0) - 1.0)
    pts = mel2hz(np.linspace(hz2mel(20.0), hz2mel(SR / 2.0), NMEL + 2))
    if alpha != 1.0:
        fmax = SR / 2.0; brk = 0.8 * fmax
        pts = np.where(pts <= brk, pts * alpha, brk * alpha + (pts - brk) * (fmax - brk * alpha) / (fmax - brk))
        pts = np.clip(pts, 0, fmax)
    bins = np.floor((NFFT + 1) * pts / SR).astype(int)
    M = np.zeros((NMEL, NFFT // 2 + 1), dtype=np.float32)
    for i in range(NMEL):
        a, b, c = bins[i], bins[i + 1], bins[i + 2]
        for k in range(a, b): M[i, k] = (k - a) / max(1, b - a)
        for k in range(b, c): M[i, k] = (c - k) / max(1, c - b)
    return M


MEL = mel_matrix()
_ALPHAS = np.round(np.linspace(0.88, 1.12, 13), 3)
_MELS = {float(a): mel_matrix(float(a)) for a in _ALPHAS}
WIN = ((0.5 - 0.5 * np.cos(2 * np.pi * np.arange(FRAME) / (FRAME - 1))) ** 0.85).astype(np.float32)


def fbank(x16, alpha=1.0):
    """x16: float32 в масштабе int16. -> [T,40] log-mel. alpha - VTLP только при обучении."""
    n = 1 + max(0, (len(x16) - FRAME) // SHIFT)
    if n <= 0:
        return np.zeros((0, NMEL), np.float32)
    idx = np.arange(FRAME)[None, :] + SHIFT * np.arange(n)[:, None]
    fr = x16[idx].astype(np.float32)
    fr -= fr.mean(axis=1, keepdims=True)
    fr[:, 1:] -= 0.97 * fr[:, :-1]; fr[:, 0] *= 0.03
    fr *= WIN
    spec = np.abs(np.fft.rfft(fr, NFFT)) ** 2
    M = MEL if alpha == 1.0 else _MELS.get(float(alpha), MEL)
    return np.log(np.maximum(spec @ M.T, 1e-10)).astype(np.float32)


# ---------------- данные ----------------
def read_wav(p):
    w = wave.open(p, "rb"); x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32)
    return x


def load_manifest(raw):
    mp = raw if raw.endswith(".jsonl") else os.path.join(raw, "manifest.jsonl")
    items = [json.loads(l) for l in open(mp, encoding="utf-8")]
    return [it for it in items if os.path.exists(it["path"])]


class Noise:
    """Синтетические шумы: цветной шум, гул, бабл из негативной речи."""
    def __init__(self, speech):
        self.speech = speech
        self.rng = np.random.default_rng(0)

    def colored(self, n):
        w = self.rng.normal(0, 1, n).astype(np.float32)
        kind = self.rng.integers(0, 3)
        if kind == 1:   # розовый (грубо): интегрирование с утечкой
            from scipy.signal import lfilter
            w = lfilter([1.0], [1.0, -0.98], w).astype(np.float32)
        elif kind == 2:  # коричневый
            w = np.cumsum(w); w -= np.linspace(w[0], w[-1], n)
        return w / (np.abs(w).max() + 1e-6)

    def hum(self, n):
        t = np.arange(n) / SR
        return (np.sin(2 * np.pi * 50 * t) + 0.3 * np.sin(2 * np.pi * 100 * t)).astype(np.float32)

    def babble(self, n):
        y = np.zeros(n, np.float32)
        for _ in range(self.rng.integers(2, 5)):
            s = self.speech[self.rng.integers(len(self.speech))]
            if len(s) >= n:
                o = self.rng.integers(0, len(s) - n + 1); seg = s[o:o + n]
            else:
                seg = np.pad(s, (0, n - len(s)))
            y += seg / (np.abs(seg).max() + 1e-6)
        return y / (np.abs(y).max() + 1e-6)

    def sample(self, n):
        k = self.rng.random()
        if k < 0.4: return self.colored(n)
        if k < 0.5: return self.hum(n)
        if k < 0.9: return self.babble(n)
        return np.zeros(n, np.float32)


def rir(rng):
    rt60 = rng.uniform(0.15, 0.6); n = int(rt60 * SR)
    h = rng.normal(0, 1, n).astype(np.float32) * np.exp(-6.9 * np.arange(n) / n)
    h[0] = 1.0
    return h / np.abs(h).sum() * 1.0


def augment_wave(x, rng, noise):
    """x: float32 int16-масштаб. Возвращает wav и коэффициент растяжения времени."""
    # темп/высота: ресемплинг
    f = rng.uniform(0.85, 1.18) if rng.random() < 0.7 else 1.0
    if f != 1.0:
        n = int(len(x) / f); x = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)
    # громкость
    x = x * (10 ** (rng.uniform(-18, 6) / 20))
    # реверберация
    if rng.random() < 0.35:
        from scipy.signal import fftconvolve
        h = rir(rng); x = fftconvolve(x, h)[: len(x)].astype(np.float32)
    # наклон спектра
    if rng.random() < 0.3:
        a = rng.uniform(-0.6, 0.6); y = x.copy(); y[1:] = x[1:] + a * x[:-1]; x = y
    # грубый НЧ-срез (дальний микрофон / телефон): простой однополюсный фильтр
    if rng.random() < 0.2:
        from scipy.signal import lfilter
        a = rng.uniform(0.3, 0.7); x = lfilter([1 - a], [1, -a], x).astype(np.float32)
    return x, f


def mix_noise(x, rng, noise):
    if rng.random() < 0.8:
        nz = noise.sample(len(x))
        snr = rng.uniform(-2, 25)
        px = np.mean(x ** 2) + 1e-6; pn = np.mean(nz ** 2) + 1e-9
        x = x + nz * np.sqrt(px / (pn * 10 ** (snr / 10)))
    if rng.random() < 0.05:   # клиппинг
        x = np.clip(x, -8000, 8000)
    return np.clip(x, -32767, 32767)


class Composer:
    """Собирает обучающие последовательности [SEQ,80] + метки [SEQ]."""
    def __init__(self, items, rng_seed=0):
        self.rng = np.random.default_rng(rng_seed)
        self.pos = {c: [read_wav(it["path"]) for it in items if KIND2CLS.get(it["kind"]) == c] for c in (1, 2)}
        self.neg = [read_wav(it["path"]) for it in items if it["kind"] in ("neg_word", "neg_sent")]
        self.pfx = [read_wav(it["path"]) for it in items if it["kind"] == "neg_prefix"]
        # позитивы «в предложении» с известным концом слова (sonet_align.py)
        self.sent = {1: [], 2: []}
        for it in items:
            c = {"sent_alisa": 1, "sent_sonet": 2}.get(it["kind"])
            if c and it.get("kw_end"):
                self.sent[c].append((read_wav(it["path"]), float(it["kw_end"])))
        self.noise = Noise(self.neg)
        print(f"clips: алиса={len(self.pos[1])}+{len(self.sent[1])}sent сонет={len(self.pos[2])}+{len(self.sent[2])}sent neg={len(self.neg)} prefix={len(self.pfx)}", flush=True)

    def one(self):
        rng = self.rng
        total = SEQ * SHIFT + FRAME
        r = rng.random()
        if r < 0.5:
            cls = 2 if rng.random() < SONET_SHARE else 1
        else:
            cls = 0
        # фон: негативная речь / тишина / шум
        parts = []
        n_bg = rng.integers(0, 3)
        for _ in range(n_bg):
            parts.append(self.neg[rng.integers(len(self.neg))])
        labels_end = []
        if cls and self.sent[cls] and rng.random() < 0.5:
            # целая фраза с ключевым словом внутри
            clip, kw_end = self.sent[cls][rng.integers(len(self.sent[cls]))]
            clip, f = augment_wave(clip, rng, self.noise)
            x = np.concatenate(parts + [clip]) if parts else clip
            end = len(x) - len(clip) + int(kw_end * SR / f)
            labels_end.append((cls, end))
        elif cls:
            kw = self.pos[cls][rng.integers(len(self.pos[cls]))]
            kw, _ = augment_wave(kw, rng, self.noise)
            pre = []
            if rng.random() < 0.4 and self.pfx:
                pre.append(self.pfx[rng.integers(len(self.pfx))])
            gap = np.zeros(int(rng.uniform(0.0, 0.6) * SR), np.float32)
            seq = pre + [gap, kw]
            x = np.concatenate(parts + seq) if parts else np.concatenate(seq)
            end = len(x)
            if rng.random() < 0.5:
                x = np.concatenate([x, np.zeros(int(rng.uniform(0.1, 0.5) * SR), np.float32), self.neg[rng.integers(len(self.neg))]])
            labels_end.append((cls, end))
        if not cls:
            if not parts:
                parts.append(self.neg[rng.integers(len(self.neg))])
            x = np.concatenate(parts)
            x, _ = augment_wave(x, rng, self.noise)
        # уложить в окно total: случайный сдвиг так, чтобы слово (если есть) было внутри
        if len(x) < total:
            padl = rng.integers(0, total - len(x) + 1)
            x = np.pad(x, (padl, total - len(x) - padl)); labels_end = [(c, e + padl) for c, e in labels_end]
        else:
            if labels_end:
                c, e = labels_end[0]
                lo, hi = max(0, e - total + int(0.2 * SR)), max(0, min(e - int(0.3 * SR), len(x) - total))
                start = rng.integers(lo, hi + 1) if hi >= lo else max(0, e - total)
            else:
                start = rng.integers(0, len(x) - total + 1)
            x = x[start:start + total]; labels_end = [(c, e - start) for c, e in labels_end]
        x = mix_noise(x, rng, self.noise)
        alpha = float(_ALPHAS[rng.integers(len(_ALPHAS))]) if rng.random() < 0.6 else 1.0
        f = fbank(x, alpha)[:SEQ]
        if len(f) < SEQ:
            f = np.pad(f, ((0, SEQ - len(f)), (0, 0)))
        y = np.zeros(SEQ, np.int64)
        for c, e in labels_end:
            fe = int(e / SHIFT)
            y[max(0, fe - POS_BEFORE): min(SEQ, fe + POS_AFTER)] = c
        # два канала: копия с небольшим независимым смещением громкости
        f2 = f + rng.normal(0, 0.15, f.shape).astype(np.float32) + rng.uniform(-0.5, 0.5)
        return np.concatenate([f, f2], axis=1).astype(np.float32), y

    def batch(self, b):
        X, Y = zip(*[self.one() for _ in range(b)])
        return np.stack(X), np.stack(Y)


# ---------------- модель ----------------
def build(tf, kernels, width, ncls):
    from tensorflow import keras
    L = keras.layers
    inp = L.Input((None, 80))
    x = L.Conv1D(width, 1, name="proj")(inp)
    x = L.ReLU()(x)
    for i, k in enumerate(kernels):
        y = L.ZeroPadding1D((k - 1, 0))(x)
        y = L.DepthwiseConv1D(k, padding="valid", name=f"dw{i}")(y)
        y = L.Conv1D(width, 1, name=f"pw{i}")(y)
        y = L.BatchNormalization(name=f"bn{i}")(y)
        y = L.ReLU()(y)
        x = L.Add()([x, y]) if i > 0 else y
    x = L.Conv1D(width, 1, name="head1")(x); x = L.ReLU()(x)
    out = L.Conv1D(ncls, 1, name="logits")(x)
    return keras.Model(inp, out)


def export_streaming(tf, model, kernels, width, ncls, path, fold=None):
    """Стриминговая функция на «сырых» tf-операциях с явными состояниями; BatchNorm свёрнут в веса.

    fold: список групп индексов классов, например [[0,1],[2]] -> 2 выхода (logsumexp по группе):
    так из 3-классовой сети получается модель «filler / сонет» без переобучения.
    """
    L = {l.name: l for l in model.layers}
    nout = len(fold) if fold else ncls

    def conv1x1(name):
        w, b = L[name].get_weights()                      # (1, Cin, Cout), (Cout,)
        return tf.constant(w[0]), tf.constant(b)

    blocks = []
    for i, k in enumerate(kernels):
        dw_w, dw_b = L[f"dw{i}"].get_weights()            # (k, C, 1), (C,)
        pw_w, pw_b = L[f"pw{i}"].get_weights()            # (1, C, C), (C,)
        gamma, beta, mean, var = L[f"bn{i}"].get_weights()
        eps = L[f"bn{i}"].epsilon
        sc = gamma / np.sqrt(var + eps)
        pw_wf = pw_w[0] * sc[None, :]                     # складываем BN в pointwise
        pw_bf = (pw_b - mean) * sc + beta
        blocks.append((k, tf.constant(dw_w[:, :, 0].T.astype(np.float32)),   # [C, k]
                       tf.constant(dw_b.astype(np.float32)), tf.constant(pw_wf.astype(np.float32)), tf.constant(pw_bf.astype(np.float32))))
    proj_w, proj_b = conv1x1("proj"); h1_w, h1_b = conv1x1("head1"); lg_w, lg_b = conv1x1("logits")
    spec = [tf.TensorSpec([1, 1, CHUNK, 80], tf.float32, name="frames")] + [
        tf.TensorSpec([1, k - 1, width], tf.float32, name=f"state_{i}") for i, k in enumerate(kernels)]

    def dwconv(cat, w_ck, k):
        # cat: [1, T+k-1, C]; свёртка по времени для каждого канала = сумма сдвигов
        T = CHUNK
        acc = tf.zeros([1, T, width], tf.float32)
        for j in range(k):
            acc = acc + cat[:, j:j + T, :] * w_ck[:, j][None, None, :]
        return acc

    class M(tf.Module):
        @tf.function(input_signature=spec)
        def __call__(self, frames, *states):
            x = tf.reshape(frames, [1, CHUNK, 80])
            x = tf.nn.relu(tf.matmul(x, proj_w) + proj_b)
            outs = {}
            for i, (k, dw_w, dw_b, pw_w, pw_b) in enumerate(blocks):
                cat = tf.concat([states[i], x], axis=1)
                outs[f"state_{i}_new"] = cat[:, -(k - 1):, :]
                y = dwconv(cat, dw_w, k) + dw_b
                y = tf.nn.relu(tf.matmul(y, pw_w) + pw_b)
                x = x + y if i > 0 else y
            x = tf.nn.relu(tf.matmul(x, h1_w) + h1_b)
            lg = tf.matmul(x, lg_w) + lg_b
            if fold:
                lg = tf.stack([tf.reduce_logsumexp(tf.gather(lg, g, axis=-1), axis=-1) for g in fold], axis=-1)
            outs["logits"] = tf.reshape(lg, [1, 1, CHUNK, nout])
            return outs

    m = M()
    cf = m.__call__.get_concrete_function()
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        tf.saved_model.save(m, d, signatures={"serving_default": cf})
        conv = tf.lite.TFLiteConverter.from_saved_model(d)
        conv.optimizations = []
        tfl = conv.convert()
    open(path, "wb").write(tfl)
    return m


def export_engine(tf, model, kernels, width, ncls, path, rep_feats=None, fold=None, quantize=True):
    """Экспорт под TFLite прошивки Станции: только CONV_2D / DEPTHWISE_CONV_2D / CONCATENATION /
    STRIDED_SLICE (+ QUANTIZE/DEQUANTIZE при int8). Граф 4-D NHWC [1,1,T,C], без RESHAPE/ADD/FC:
    остаточная связь = CONCATENATION по каналам + CONV_2D с постоянным ядром [I;I].
    rep_feats: список массивов [T,80] для representative dataset (int8); fold - как в export_streaming.
    """
    L = {l.name: l for l in model.layers}
    nout = len(fold) if fold else ncls

    def pw(name, extra_scale=None, extra_bias=None):
        w, b = L[name].get_weights()                          # (1,Cin,Cout),(Cout,)
        w = w[0]
        if extra_scale is not None:
            w = w * extra_scale[None, :]; b = (b - extra_bias[0]) * extra_scale + extra_bias[1]
        return tf.constant(w[None, None].astype(np.float32)), tf.constant(b.astype(np.float32))   # [1,1,Cin,Cout]

    blocks = []
    for i, k in enumerate(kernels):
        dw_w, dw_b = L[f"dw{i}"].get_weights()                # (k,C,1),(C,)
        gamma, beta, mean, var = L[f"bn{i}"].get_weights(); eps = L[f"bn{i}"].epsilon
        sc = gamma / np.sqrt(var + eps)
        pw_w, pw_b = L[f"pw{i}"].get_weights()
        pw_wf = (pw_w[0] * sc[None, :])[None, None].astype(np.float32)
        pw_bf = ((pw_b - mean) * sc + beta).astype(np.float32)
        blocks.append((k, tf.constant(dw_w[None].astype(np.float32)), tf.constant(dw_b.astype(np.float32)),
                       tf.constant(pw_wf), tf.constant(pw_bf)))
    proj_w, proj_b = pw("proj"); h1_w, h1_b = pw("head1"); lg_w, lg_b = pw("logits")
    eye2 = tf.constant(np.concatenate([np.eye(width), np.eye(width)], axis=0)[None, None].astype(np.float32))  # [1,1,2C,C]
    if fold:
        # свёртка классов logsumexp недоступна без EXP/LOG; приближение: складываем логиты групп через постоянное ядро нельзя
        # (это не logsumexp). Поэтому для fold берём max-подобную замену: отдельный выход на группу = сумма логитов группы.
        G = np.zeros((ncls, len(fold)), np.float32)
        for j, g in enumerate(fold):
            for c in g: G[c, j] = 1.0
        fold_w = tf.constant(G[None, None])
    spec = [tf.TensorSpec([1, 1, CHUNK, 80], tf.float32, name="frames")] + [
        tf.TensorSpec([1, 1, k - 1, width], tf.float32, name=f"state_{i}") for i, k in enumerate(kernels)]

    class M(tf.Module):
        @tf.function(input_signature=spec)
        def __call__(self, frames, *states):
            x = tf.nn.relu(tf.nn.conv2d(frames, proj_w, 1, "VALID") + proj_b)
            outs = {}
            for i, (k, dw_w, dw_b, pw_w_, pw_b_) in enumerate(blocks):
                # состояние хранится в ОБРАТНОМ порядке кадров: срезы с шагом -1 конвертер оставляет
                # как STRIDED_SLICE (срез с шагом 1 он превращает в SLICE, которого нет в прошивке)
                st_fwd = states[i][:, :, ::-1, :]
                cat = tf.concat([st_fwd, x], axis=2)
                outs[f"state_{i}_new"] = cat[:, :, :CHUNK - 1 if CHUNK - 1 > 0 else None:-1, :] if False else cat[:, :, ::-1, :][:, :, :k - 1, :]
                y = tf.nn.depthwise_conv2d(cat, dw_w, [1, 1, 1, 1], "VALID") + dw_b
                y = tf.nn.relu(tf.nn.conv2d(y, pw_w_, 1, "VALID") + pw_b_)
                if i > 0:
                    x = tf.nn.conv2d(tf.concat([x, y], axis=3), eye2, 1, "VALID")   # x + y без ADD
                else:
                    x = y
            x = tf.nn.relu(tf.nn.conv2d(x, h1_w, 1, "VALID") + h1_b)
            lg = tf.nn.conv2d(x, lg_w, 1, "VALID") + lg_b
            if fold:
                lg = tf.nn.conv2d(lg, fold_w, 1, "VALID")
            outs["logits"] = lg
            return outs

    m = M()
    cf = m.__call__.get_concrete_function()
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        tf.saved_model.save(m, d, signatures={"serving_default": cf})
        conv = tf.lite.TFLiteConverter.from_saved_model(d)
        if quantize and rep_feats:
            def rep():
                for f80 in rep_feats:
                    states = [np.zeros((1, 1, k - 1, width), np.float32) for k in kernels]
                    for i in range(0, len(f80) // CHUNK * CHUNK, CHUNK):
                        fr = f80[i:i + CHUNK][None, None].astype(np.float32)
                        yield dict({"frames": fr}, **{f"state_{j}": states[j] for j in range(len(kernels))})
                        r = m(tf.constant(fr), *[tf.constant(s) for s in states])
                        states = [r[f"state_{j}_new"].numpy() for j in range(len(kernels))]
            conv.optimizations = [tf.lite.Optimize.DEFAULT]
            conv.representative_dataset = rep
            conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            conv.inference_input_type = tf.float32
            conv.inference_output_type = tf.float32
        tfl = conv.convert()
    open(path, "wb").write(tfl)
    return m


def verify(tf, model, m, kernels, width, X):
    """Стриминг == нестриминг на одной последовательности."""
    ref = model.predict(X[None], verbose=0)[0]
    states = [np.zeros((1, k - 1, width), np.float32) for k in kernels]
    out = []
    for i in range(0, len(X) // CHUNK * CHUNK, CHUNK):
        r = m(tf.constant(X[i:i + CHUNK][None, None]), *[tf.constant(s) for s in states])
        states = [r[f"state_{j}_new"].numpy() for j in range(len(kernels))]
        out.append(r["logits"].numpy()[0, 0])
    out = np.concatenate(out)
    return float(np.abs(out - ref[: len(out)]).max())


_W = {}


def _init_worker(items, seed, share=0.65):
    import os as _os
    global SONET_SHARE; SONET_SHARE = share
    _W["comp"] = Composer(items, seed + _os.getpid())


def _make_batch(arg):
    seed, b = arg
    _W["comp"].rng = np.random.default_rng(seed)
    return _W["comp"].batch(b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=os.path.join(ROOT, "sonet", "raw"))
    ap.add_argument("--out", default=os.path.join(ROOT, "sonet", "run1"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--steps", type=int, default=150, help="батчей на эпоху")
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--kernels", default="9,13,17,21,25,29,33")
    ap.add_argument("--holdout-voice", default="piper-denis", help="голос только для валидации")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--init-weights", default="", help="дообучение: веса предыдущего прогона (best.weights.h5)")
    ap.add_argument("--workers", type=int, default=5, help="процессов сборки батчей (0 = в основном процессе)")
    ap.add_argument("--sonet-share", type=float, default=0.65)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--extra-raw", default="", help="доп. manifest.jsonl (например, реальные записи), через запятую")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    global SONET_SHARE; SONET_SHARE = a.sonet_share
    random.seed(a.seed); np.random.seed(a.seed)
    import tensorflow as tf
    tf.random.set_seed(a.seed)
    kernels = [int(k) for k in a.kernels.split(",")]
    items = load_manifest(a.raw)
    for extra in [e for e in a.extra_raw.split(",") if e]:
        items += load_manifest(extra)
    train_items = [it for it in items if it["voice"] != a.holdout_voice]
    val_items = [it for it in items if it["voice"] == a.holdout_voice] or items[:200]
    vcomp = Composer(val_items, a.seed + 100)
    Xv, Yv = vcomp.batch(256)
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    pool = ctx.Pool(a.workers, initializer=_init_worker, initargs=(train_items, a.seed, a.sonet_share)) if a.workers > 0 else None
    comp = None if pool else Composer(train_items, a.seed)

    model = build(tf, kernels, a.width, len(CLASSES))
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
            res[CLASSES[c]] = dict(recall=float((det & has).sum() / max(1, has.sum())), fa=float((det & ~has).sum() / max(1, (~has).sum())))
        return res

    best = -1
    log = open(os.path.join(a.out, "train.log"), "a")
    for ep in range(a.epochs):
        t0 = time.time(); ls = []
        if pool:
            for X, Y in pool.imap_unordered(_make_batch, [(ep * 100003 + i, a.batch) for i in range(a.steps)]):
                ls.append(float(step(tf.constant(X), tf.constant(Y))))
        else:
            for _ in range(a.steps):
                X, Y = comp.batch(a.batch)
                ls.append(float(step(tf.constant(X), tf.constant(Y))))
        ev = evaluate(Xv, Yv)
        score = ev["сонет"]["recall"] - 2 * ev["сонет"]["fa"] + 0.5 * (ev["алиса"]["recall"] - 2 * ev["алиса"]["fa"])
        msg = f"ep {ep+1}/{a.epochs} loss {np.mean(ls):.4f} val {json.dumps(ev, ensure_ascii=False)} {time.time()-t0:.0f}s"
        print(msg, flush=True); log.write(msg + "\n"); log.flush()
        model.save_weights(os.path.join(a.out, f"ep{ep+1:02d}.weights.h5"))   # для отбора по тестам (sonet_select.py)
        if score > best:
            best = score; model.save_weights(os.path.join(a.out, "best.weights.h5"))
        if (ep + 1) % 5 == 0:
            opt.learning_rate.assign(float(opt.learning_rate) * 0.7)
    model.load_weights(os.path.join(a.out, "best.weights.h5"))
    tfl_path = os.path.join(a.out, "model.tflite")
    m = export_streaming(tf, model, kernels, a.width, len(CLASSES), tfl_path)
    err = verify(tf, model, m, kernels, a.width, Xv[0])
    print(f"exported {tfl_path} ({os.path.getsize(tfl_path)} байт), streaming-vs-offline max diff {err:.2e}", flush=True)
    with open(os.path.join(a.out, "words.txt"), "w") as f:
        for i, c in enumerate(CLASSES): f.write(f"{c} {i}\n")
    # запасной вариант: только «сонет» (filler+алиса слиты), на случай если движок не примет 3 класса
    export_streaming(tf, model, kernels, a.width, len(CLASSES), os.path.join(a.out, "model_sonet2.tflite"), fold=[[0, 1], [2]])
    with open(os.path.join(a.out, "words_sonet2.txt"), "w") as f:
        f.write("filler 0\nсонет 1\n")
    # и только «алиса» (для сверки с оригинальной моделью колонки)
    export_streaming(tf, model, kernels, a.width, len(CLASSES), os.path.join(a.out, "model_alisa2.tflite"), fold=[[0, 2], [1]])
    json.dump({"kernels": kernels, "width": a.width, "classes": CLASSES, "val": evaluate(Xv, Yv)}, open(os.path.join(a.out, "meta.json"), "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
