#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Обобщённый генератор обучающих wav для споттера произвольного wake word.

Использует TTS-инфраструктуру из sonet_data.py и конфигурацию слова из wakeword_config.py.

Запуск в окружении venv_tts (torch + piper):
  python wakeword_data.py --word votchi --per-voice 40
  python wakeword_data.py --word siesta --per-voice 40

Пишет 16 кГц моно wav в OUT/<раздел>/,
и manifest.jsonl с полями: path, kind (pos_<word> | pos_alisa | neg_word | neg_sent), voice, text.
"""
import argparse, json, os, random, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sonet_data as D
from wakeword_config import get_config, ALISA_VARIANTS, COMMON_NEG_WORDS

NEG_SENTENCES = D.NEG_SENTENCES
PREFIXES = D.PREFIXES


def main():
    ap = argparse.ArgumentParser(description="Генерация обучающих данных для wake word")
    ap.add_argument("--word", required=True, choices=["siesta", "votchi", "miku", "smith"],
                    help="Wake word key")
    ap.add_argument("--out", default=None, help="Выходной каталог (по умолчанию: $WAKEWORD_ROOT/<word>/raw)")
    ap.add_argument("--per-voice", type=int, default=40, help="Клипов ключевого слова на голос")
    ap.add_argument("--neg-sent-per-voice", type=int, default=40)
    ap.add_argument("--backends", default="say,silero,piper")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    cfg = get_config(a.word)
    out = a.out or os.path.join(D.ROOT, a.word, "raw")

    # Build keyword dict: {word_key: variants}
    keywords = {a.word: cfg["variants"], "alisa": ALISA_VARIANTS}
    # Negative words: word-specific + common
    neg_words = cfg["all_neg_words"]

    random.seed(a.seed)
    os.makedirs(out, exist_ok=True)
    man = open(os.path.join(out, "manifest.jsonl"), "a", encoding="utf-8")
    backends = []
    for b in a.backends.split(","):
        try:
            backends.append({"say": D.Say, "silero": D.Silero, "silero31": D.Silero31, "piper": D.Piper}[b]())
            print("backend ok:", b, flush=True)
        except Exception as e:
            print("backend FAIL:", b, e, flush=True)

    def emit(kind, be, voice, text, x, tag):
        x = D.trim(x)
        if len(x) < D.SR * 0.15:
            return
        p = os.path.join(out, kind, f"{be.name}-{voice}-{tag}.wav")
        D.write_wav(p, x)
        man.write(json.dumps({"path": p, "kind": kind, "voice": f"{be.name}-{voice}",
                              "text": text, "dur": len(x) / D.SR}, ensure_ascii=False) + "\n")

    n = 0
    for be in backends:
        for voice in be.voices:
            # Ключевые слова: отдельным клипом
            for kw, variants in keywords.items():
                for i in range(a.per_voice):
                    text = random.choice(variants)
                    try:
                        emit(f"pos_{kw}", be, voice, text, be.synth(text, voice), f"{i:03d}")
                    except Exception as e:
                        print("err", be.name, voice, text, e, flush=True)
                    n += 1
            # Трудные негативы: каждое слово 2 раза
            for i, w in enumerate(neg_words):
                for r in range(2):
                    try:
                        emit("neg_word", be, voice, w, be.synth(w, voice), f"{i:03d}-{r}")
                    except Exception as e:
                        print("err", be.name, voice, w, e, flush=True)
            # Предложения и префиксы
            sents = random.sample(NEG_SENTENCES, min(a.neg_sent_per_voice, len(NEG_SENTENCES)))
            for i, s in enumerate(sents):
                try:
                    emit("neg_sent", be, voice, s, be.synth(s, voice), f"{i:03d}")
                except Exception as e:
                    print("err", be.name, voice, s, e, flush=True)
            for i, pfx in enumerate([p for p in PREFIXES if p]):
                try:
                    emit("neg_prefix", be, voice, pfx, be.synth(pfx, voice), f"{i:03d}")
                except Exception as e:
                    print("err", be.name, voice, pfx, e, flush=True)
            man.flush()
            print("done voice", be.name, voice, flush=True)
    man.close()
    print(f"total keyword clips requested: {n} (word={a.word})")


if __name__ == "__main__":
    main()
