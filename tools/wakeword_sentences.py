#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Позитивы «в предложении» для произвольного wake word.

Ключевое слово в начале / середине / конце целой фразы (естественная коартикуляция),
плюс те же фразы без ключевого слова как негативы. Границы слова потом находит
sonet_align.py (whisper, word timestamps).

  venv_tts: python wakeword_sentences.py --word votchi --per-voice 50
"""
import argparse, json, os, random, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sonet_data as D
from wakeword_config import get_config

TEMPLATES = [
    "{KW}, включи свет на кухне.", "{KW}, какая сегодня погода?", "{KW}, сколько времени?", "{KW}, поставь будильник на семь.",
    "{KW}, напомни позвонить другу.", "{KW}, расскажи анекдот.", "{KW}, тише.", "{KW}, громче.", "{KW}, стоп.", "{KW}, продолжай.",
    "{KW}, что нового?", "{KW}, какой курс доллара?", "{KW}, сколько будет семью восемь?", "{KW}, переведи слово кошка.",
    "{KW}, включи музыку.", "{KW}, выключи телевизор.", "{KW}, где мои ключи?", "{KW}, закажи такси.", "{KW}, спокойной ночи.",
    "{KW}, доброе утро.", "{KW}!", "{KW}?", "{KW}.", "{KW}, ты здесь?", "{KW}, слушай.",
    "Эй, {KW}, какая погода?", "Слушай, {KW}, сколько времени?", "Скажи, {KW}, что делать?", "Привет, {KW}.", "Так, {KW}, записывай.",
    "Ну-ка, {KW}, напомни про котёл.", "Послушай, {KW}, тише.", "Итак, {KW}, поехали.", "Ладно, {KW}, выключи всё.", "Хорошо, {KW}, спасибо.",
    "Включи свет, {KW}.", "Какая погода, {KW}?", "Сколько времени, {KW}?", "Спасибо, {KW}.", "Пока, {KW}.", "Я спрашиваю тебя, {KW}.",
    "Где ты, {KW}?", "Поставь таймер, {KW}.", "Напомни мне, {KW}, про врача.", "Давай, {KW}, расскажи сказку.",
]
NEG_TEMPLATES = [t.replace("{KW}, ", "").replace(", {KW}", "").replace("{KW}", "скажи") for t in TEMPLATES]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--word", required=True, choices=["siesta", "votchi", "miku", "smith"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--per-voice", type=int, default=50)
    ap.add_argument("--backends", default="say,silero,piper")
    ap.add_argument("--seed", type=int, default=21)
    a = ap.parse_args()

    cfg = get_config(a.word)
    out = a.out or os.path.join(D.ROOT, a.word, "raw_sent")
    kw_word = cfg["word_capitalized"]

    random.seed(a.seed); os.makedirs(out, exist_ok=True)
    man = open(os.path.join(out, "manifest.jsonl"), "a", encoding="utf-8")
    bes = []
    for b in a.backends.split(","):
        try:
            bes.append({"say": D.Say, "silero": D.Silero, "silero31": D.Silero31, "piper": D.Piper}[b]()); print("backend ok", b, flush=True)
        except Exception as e:
            print("backend FAIL", b, e, flush=True)
    for be in bes:
        for voice in be.voices:
            for kw, kind in ((kw_word, f"sent_{a.word}"), ("Алиса", "sent_alisa")):
                for i in range(a.per_voice):
                    t = random.choice(TEMPLATES).replace("{KW}", kw)
                    try:
                        x = D.trim(be.synth(t, voice))
                    except Exception as e:
                        print("err", be.name, voice, t, e, flush=True); continue
                    p = os.path.join(out, kind, f"{be.name}-{voice}-{i:03d}.wav"); D.write_wav(p, x)
                    man.write(json.dumps({"path": p, "kind": kind, "voice": f"{be.name}-{voice}", "text": t, "kw": kw.lower(), "dur": len(x) / D.SR}, ensure_ascii=False) + "\n")
            for i in range(a.per_voice // 2):
                t = random.choice(NEG_TEMPLATES)
                try:
                    x = D.trim(be.synth(t, voice))
                except Exception as e:
                    print("err", be.name, voice, t, e, flush=True); continue
                p = os.path.join(out, "neg_sent", f"{be.name}-{voice}-s{i:03d}.wav"); D.write_wav(p, x)
                man.write(json.dumps({"path": p, "kind": "neg_sent", "voice": f"{be.name}-{voice}", "text": t, "dur": len(x) / D.SR}, ensure_ascii=False) + "\n")
            man.flush(); print("done voice", be.name, voice, flush=True)
    man.close()


if __name__ == "__main__":
    main()
