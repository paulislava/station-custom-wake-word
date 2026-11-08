#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Нарезка голосовых заметок (.oga/.ogg/.m4a/.wav) на отдельные произнесения по паузам.

  python segment_voice.py --label sonet  voice_note.oga  --out "$WAKEWORD_ROOT"/sonet/real
  python segment_voice.py --label alisa  … ;  --label neg …

Выход: OUT/<label>/<имя>-NNN.wav (16 кГц моно int16) + печать длительностей. Порог тишины
адаптивный (медиана энергии кадров), минимальная пауза 0,35 с, минимальная длина сегмента 0,25 с,
запас 0,12 с по краям. Конвертация через ffmpeg.
"""
import argparse, os, subprocess, tempfile, wave
import numpy as np

ROOT = os.path.abspath(os.path.expanduser(os.environ.get("WAKEWORD_ROOT", "./work")))  # данные, голоса, прогоны
SR = 16000


def to_wav(src):
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", src, "-ac", "1", "-ar", str(SR), "-f", "wav", tmp], check=True)
    w = wave.open(tmp, "rb"); x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32); w.close()
    os.unlink(tmp)
    return x


def segments(x, min_gap=0.35, min_len=0.25, pad=0.12):
    hop = 160
    n = len(x) // hop
    e = np.array([np.sqrt(np.mean(x[i * hop:(i + 1) * hop] ** 2) + 1e-6) for i in range(n)])
    thr = max(np.percentile(e, 30) * 3.0, 60.0)
    voiced = e > thr
    # сглаживание: закрыть дыры короче min_gap
    segs = []; i = 0
    while i < n:
        if voiced[i]:
            j = i
            while j < n:
                if voiced[j]:
                    j += 1; continue
                k = j
                while k < n and not voiced[k]: k += 1
                if (k - j) * hop / SR >= min_gap or k >= n: break
                j = k
            segs.append((i, j)); i = j
        else:
            i += 1
    out = []
    for a, b in segs:
        if (b - a) * hop / SR >= min_len:
            s = max(0, int(a * hop - pad * SR)); t = min(len(x), int(b * hop + pad * SR))
            out.append(x[s:t])
    return out


def write(path, x):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    w = wave.open(path, "wb"); w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
    w.writeframes(np.clip(x, -32767, 32767).astype(np.int16).tobytes()); w.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+"); ap.add_argument("--label", required=True); ap.add_argument("--out", default=os.path.join(ROOT, "sonet", "real"))
    ap.add_argument("--whole", action="store_true", help="не резать, сохранить целиком")
    a = ap.parse_args()
    for f in a.files:
        x = to_wav(f)
        base = os.path.splitext(os.path.basename(f))[0]
        parts = [x] if a.whole else segments(x)
        for i, p in enumerate(parts):
            write(os.path.join(a.out, a.label, f"{base}-{i:03d}.wav"), p)
        print(f"{f}: {len(x)/SR:.1f}s -> {len(parts)} сегментов: " + " ".join(f"{len(p)/SR:.2f}" for p in parts))
