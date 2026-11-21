#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Тестовый набор: целые фразы с wake word и «Алиса» внутри, трудные негативы.

  venv_tts: python wakeword_testset.py --word votchi
"""
import argparse, json, os, random, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sonet_data as D
from wakeword_config import get_config

random.seed(5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--word", required=True, choices=["siesta", "votchi", "miku", "smith"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    cfg = get_config(a.word)
    kw = cfg["word_capitalized"]
    word_ru = cfg["word_ru"]
    out = a.out or os.path.join(D.ROOT, a.word, "test")

    POS_W = [
        f"{kw}, включи свет на кухне.", f"Эй, {kw}, какая сегодня погода?", f"Скажи-ка, {kw}, сколько времени?",
        f"{kw}!", f"{kw}, напомни позвонить другу.", f"Слушай, {kw}, расскажи анекдот.", f"{kw}, поставь будильник на семь.",
        f"А ну-ка, {kw}, сколько будет дважды два?", f"{kw}, тише.", f"Привет, {kw}.",
    ]
    POS_A = ["Алиса, включи музыку.", "Алиса, какая погода завтра?", "Эй, Алиса, сколько времени?", "Алиса!", "Привет, Алиса."]
    NEG = cfg["neg_words"][:10] + [
        "Включи свет на кухне.", "Какая завтра погода?", "Напомни позвонить другу.", "Сколько времени?",
    ]

    bes = [D.Silero(), D.Piper()]
    os.makedirs(out, exist_ok=True)
    man = open(os.path.join(out, "manifest.jsonl"), "w", encoding="utf-8")
    for be in bes:
        for voice in be.voices:
            for kind, texts in ((f"pos_{a.word}", POS_W), ("pos_alisa", POS_A), ("neg_sent", NEG)):
                for i, t in enumerate(texts):
                    try:
                        x = D.trim(be.synth(t, voice))
                    except Exception as e:
                        print("err", be.name, voice, t, e); continue
                    p = os.path.join(out, kind, f"{be.name}-{voice}-{i:02d}.wav"); D.write_wav(p, x)
                    man.write(json.dumps({"path": p, "kind": kind, "voice": f"{be.name}-{voice}", "text": t}, ensure_ascii=False) + "\n")
            print("done", be.name, voice, flush=True)
    man.close()
    print(f"test set written to {out}")


if __name__ == "__main__":
    main()
