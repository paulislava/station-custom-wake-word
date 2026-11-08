#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Тестовый набор: целые фразы (не склейка) с «Сонет»/«Алиса» внутри и трудные негативы-предложения."""
import os, random, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sonet_data as D
random.seed(5)
OUT = os.path.join(D.ROOT, "sonet", "test")   # $WAKEWORD_ROOT/sonet/test
POS_S = ["Сонет, включи свет на кухне.", "Эй, Сонет, какая сегодня погода?", "Скажи-ка, Сонет, сколько времени?",
         "Сонет!", "Сонет, напомни позвонить другу.", "Слушай, Сонет, расскажи анекдот.", "Сонет, поставь будильник на семь.",
         "А ну-ка, Сонет, сколько будет дважды два?", "Сонет, тише.", "Привет, Сонет."]
POS_A = ["Алиса, включи музыку.", "Алиса, какая погода завтра?", "Эй, Алиса, сколько времени?", "Алиса!", "Привет, Алиса."]
NEG = ["Он читал сонеты Шекспира весь вечер.", "Соня спит, не шумите.", "Монета упала под диван.", "В кабинете холодно.",
       "Сенатор выступил с речью.", "Соната Бетховена звучала долго.", "Мне приснился странный сон.", "Планета Марс красная.",
       "Балет начался в семь вечера.", "Самолёт сел вовремя.", "Это сонный город у моря.", "Соня, иди сюда.",
       "Брюнет в синем пальто.", "Кларнет и корнет в оркестре.", "Сто нет, двести да.", "Лариса, Мелисса и Василиса.",
       "Включи свет на кухне.", "Какая завтра погода?", "Напомни позвонить другу.", "Сколько времени?"]
bes = [D.Silero(), D.Piper()]
os.makedirs(OUT, exist_ok=True)
man = open(os.path.join(OUT, "manifest.jsonl"), "w", encoding="utf-8")
import json
for be in bes:
    for voice in be.voices:
        for kind, texts in (("pos_sonet", POS_S), ("pos_alisa", POS_A), ("neg_sent", NEG)):
            for i, t in enumerate(texts):
                try:
                    x = D.trim(be.synth(t, voice))
                except Exception as e:
                    print("err", be.name, voice, t, e); continue
                p = os.path.join(OUT, kind, f"{be.name}-{voice}-{i:02d}.wav"); D.write_wav(p, x)
                man.write(json.dumps({"path": p, "kind": kind, "voice": f"{be.name}-{voice}", "text": t}, ensure_ascii=False) + "\n")
        print("done", be.name, voice, flush=True)
