#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Быстрое выравнивание ключевого слова в синтезированных фразах по паузам (без ASR).

Шаблоны фраз разделены знаками препинания, и синтез ставит на них паузы. Делим текст на куски по
[,.!?], находим номер куска с ключевым словом, делим звук на сегменты по паузам (энергия) и, если
число сегментов совпало с числом кусков, берём конец нужного сегмента как kw_end. Иначе kw_end=null.

  python siesta_align_pause.py manifest.jsonl [manifest2.jsonl ...]
Пишет kw_start/kw_end в тот же manifest. Проверка качества: siesta_eval штатной моделью на sent_alisa
должна давать пик рядом с kw_end.
"""
import json, re, sys, wave, os
import numpy as np

SR = 16000


def read(p):
    w = wave.open(p, "rb"); return np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32)


def segments(x, min_gap=0.10, min_len=0.12):
    hop = 80
    n = len(x) // hop
    e = np.sqrt(np.mean(x[: n * hop].reshape(n, hop) ** 2, axis=1) + 1e-6)
    thr = max(np.percentile(e, 20) * 4.0, np.max(e) * 0.02, 40.0)
    v = e > thr
    segs = []; i = 0
    while i < n:
        if v[i]:
            j = i
            while j < n:
                if v[j]: j += 1; continue
                k = j
                while k < n and not v[k]: k += 1
                if (k - j) * hop / SR >= min_gap or k >= n: break
                j = k
            segs.append((i * hop / SR, j * hop / SR)); i = j
        else:
            i += 1
    return [s for s in segs if s[1] - s[0] >= min_len]


def chunks(text):
    return [c.strip() for c in re.split(r"[,.!?;:]+", text) if c.strip()]


def main(paths):
    for mp in paths:
        items = [json.loads(l) for l in open(mp, encoding="utf-8")]
        ok = miss = 0
        for it in items:
            if not it["kind"].startswith("sent_"):
                continue
            ch = chunks(it["text"]); kw = it["kw"]
            idx = next((i for i, c in enumerate(ch) if kw in c.lower()), None)
            if idx is None:
                it["kw_end"] = None; miss += 1; continue
            segs = segments(read(it["path"]))
            if len(segs) == len(ch):
                a, b = segs[idx]
                # если в куске есть другие слова (например «Эй Сиеста»), слово в конце куска, когда kw стоит последним
                words = ch[idx].lower().split()
                if len(words) > 1 and words.index(next(w for w in words if kw in w)) != len(words) - 1:
                    # слово не последнее в куске: границу по паузе не определить
                    it["kw_end"] = None; miss += 1; continue
                it["kw_start"] = round(a, 3); it["kw_end"] = round(b, 3); ok += 1
            else:
                it["kw_end"] = None; miss += 1
        with open(mp, "w", encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
        print(f"{mp}: найдено {ok}, не определено {miss}")


if __name__ == "__main__":
    main(sys.argv[1:])
