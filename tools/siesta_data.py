#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Генерация обучающих wav для споттера «Сиеста» (и «Алиса») из синтеза речи.

Запуск в окружении venv_tts (torch + piper). Пишет 16 кГц моно wav в OUT/<раздел>/,
и manifest.jsonl с полями: path, kind (pos_siesta | pos_alisa | neg_word | neg_sent), voice, text.
Ключевое слово синтезируется ОТДЕЛЬНЫМ клипом, чтобы его границы были известны точно
(контекст до/после приклеивается на этапе обучения).

  python siesta_data.py --out "$WAKEWORD_ROOT"/siesta/raw --per-voice 40
Каталог по умолчанию: $WAKEWORD_ROOT (если не задан - ./work); голоса TTS - в $WAKEWORD_ROOT/tts.
"""
import argparse, json, os, random, subprocess, sys, tempfile, wave
import numpy as np

SR = 16000
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.expanduser(os.environ.get("WAKEWORD_ROOT", "./work")))  # данные, голоса, прогоны
TTS_DIR = os.path.join(ROOT, "tts")

KEYWORDS = {
    "siesta": ["Сиеста", "Сиеста.", "Сиеста!", "Сиес+та", "Сиеста,", "Сиеста?"],
    "alisa": ["Алиса", "Алиса.", "Алиса!", "Али+са", "Алиса,", "Алиса?"],
}
# трудные негативы: похожие слова и сочетания
NEG_WORDS = [
    "сеста", "сесть", "сесто", "сиест", "симптом", "система", "сиест", "феест", "фест",
    "места", "мести", "месть", "часто", "чисто", "шесть", "жест", "шест", "лес", "тест",
    "шесть часов", "двести", "триста", "шестьсот", "сорок шесть", "пять шесть",
    "Сиена", "Сьеста", "Фиеста", "сиеста", "висла", "исписать", "испечь",
    "сиесты", "сиесту", "сиестой", "сиесте", "сестра", "сестрица", "сестёр", "сестрин",
    "шесть", "шестой", "шестерня", "шест", "шестнадцать", "шестидесятый",
    "Алиса", "Яндекс", "Алиска", "Лариса", "лиса", "Мелисса", "Василиса", "Анфиса", "Раиса", "Алис",
    "Клод", "Сири", "Маруся", "Салют", "Джой", "окей гугл", "алиса стоп", "хватит", "пауза",
]
# общие фразы: быт, команды Алисе, новости, разговор
NEG_SENTENCES = [
    "Алиса, включи музыку погромче.", "Какая завтра погода за городом?", "Поставь будильник на семь утра.",
    "Включи свет на кухне.", "Выключи телевизор, пожалуйста.", "Сколько сейчас времени?", "Расскажи новости.",
    "Напомни позвонить другу вечером.", "Где мои ключи, ты не видела?", "Давай поужинаем пораньше сегодня.",
    "Передай соль, пожалуйста.", "Я сегодня очень устал на работе.", "Завтра поедем на дачу, если не будет дождя.",
    "Курс доллара вырос на два процента.", "Правительство обсуждает новый бюджет на следующий год.",
    "В столице открылась выставка современного искусства.", "Футболисты сборной провели тренировку перед матчем.",
    "Приготовь мне чашку кофе без сахара.", "Сделай потише, ребёнок спит.", "Когда приедет курьер с заказом?",
    "Надо купить хлеб, молоко и яйца.", "Ты записал показания счётчиков?", "Котёл опять мигает ошибкой.",
    "Позвони в управляющую компанию по поводу отопления.", "Включи сериал на телевизоре.", "Следующий трек.",
    "Поставь на паузу.", "Продолжи воспроизведение.", "Громче на два деления.", "Тише, пожалуйста.",
    "Сколько будет двадцать семь умножить на три?", "Переведи на английский слово «дом».",
    "Какой сегодня день недели?", "Расскажи анекдот.", "Спокойной ночи.", "Доброе утро, как спалось?",
    "Во сколько закрывается магазин на углу?", "Запиши в список покупок стиральный порошок.",
    "Какие пробки по дороге на работу?", "Включи радио Маяк.", "Найди рецепт борща.",
    "Сколько километров до вокзала?", "Напомни про поверку газового счётчика в марте.",
    "Открой шторы в спальне.", "Поставь таймер на десять минут для яиц.", "Выключи всё и пойдём спать.",
    "Соседка звонила, просила перезвонить.", "Опять до ночи сидишь в ноутбуке.", "Рейс вылетает в субботу утром.",
    "Сегодня по телевизору показывали старый фильм.", "Соседи сверху снова сверлят стену.",
    "На улице минус пять и гололёд.", "Собака просится гулять.", "Кот опять уронил цветок с подоконника.",
    "Машина не заводится, аккумулятор сел.", "Надо записаться к стоматологу на следующей неделе.",
    "Переключи на первый канал.", "Что идёт сегодня вечером по телевизору?", "Включи мультики для детей.",
    "Поставь детскую песню про жирафа.", "Расскажи сказку про колобка.", "Какой счёт в матче Спартак Зенит?",
    "Сколько стоит билет на поезд?", "Закажи такси к подъезду через пятнадцать минут.",
    "У нас закончился сахар и соль.", "Я положил документы на стол в кабинете.", "Он играл сонату Бетховена.",
    "Это новая монета из коллекции.", "Соня проснулась и плачет.", "Сон был странный и длинный.",
    "В кабинете директора идёт совещание.", "Балет начинается в семь вечера.", "Планета Марс видна на востоке.",
    "Сенатор выступил с речью.", "Брюнет в синем пальто спросил дорогу.", "Кларнет звучал очень нежно.",
]
PREFIXES = ["", "", "", "Эй,", "Слушай,", "Так,", "Ну,", "Скажи,", "Привет,", "Послушай,", "Э-э,", "Итак,"]


def write_wav(path, x):
    x = np.clip(x, -1, 1)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    w = wave.open(path, "wb"); w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
    w.writeframes((x * 32767).astype(np.int16).tobytes()); w.close()


def resample(x, sr):
    if sr == SR:
        return x
    n = int(round(len(x) * SR / sr))
    return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)


def trim(x, thr_db=-40, pad=0.08):
    """Обрезать тишину по краям, оставив pad секунд."""
    if len(x) == 0:
        return x
    thr = 10 ** (thr_db / 20) * max(1e-6, np.abs(x).max())
    idx = np.where(np.abs(x) > thr)[0]
    if len(idx) == 0:
        return x
    a, b = max(0, idx[0] - int(pad * SR)), min(len(x), idx[-1] + int(pad * SR))
    return x[a:b]


# ---------- бэкенды TTS ----------
class Say:
    name = "say"
    voices = ["Milena"]

    def synth(self, text, voice, rate=None):
        rate = rate or random.choice([130, 150, 170, 190, 210, 230])
        with tempfile.TemporaryDirectory() as d:
            aiff, wavp = os.path.join(d, "t.aiff"), os.path.join(d, "t.wav")
            subprocess.run(["say", "-v", voice, "-r", str(rate), "-o", aiff, text.replace("+", "")], check=True)
            subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", aiff, wavp], check=True)
            w = wave.open(wavp, "rb"); x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
        return x


class Silero:
    name = "silero"
    voices = ["aidar", "baya", "kseniya", "xenia", "eugene"]

    def __init__(self):
        import torch
        torch.set_num_threads(4)
        from silero import silero_tts
        self.model, _ = silero_tts(language='ru', speaker='v4_ru')
        self.model.to(torch.device("cpu"))

    def synth(self, text, voice, rate=None):
        rate = rate or random.choice(["x-slow", "slow", "medium", "medium", "fast", "x-fast"])
        pitch = random.choice(["x-low", "low", "medium", "medium", "high", "x-high"])
        ssml = f"<speak><prosody rate='{rate}' pitch='{pitch}'>{text}</prosody></speak>"
        try:
            a = self.model.apply_tts(ssml_text=ssml, speaker=voice, sample_rate=24000, put_accent=True, put_yo=True)
        except Exception:
            a = self.model.apply_tts(text=text.replace("+", ""), speaker=voice, sample_rate=24000)
        return resample(a.numpy().astype(np.float32), 24000)


class Silero31:
    """Silero v3.1: speaker=random даёт новый случайный голос на каждый клип (разнообразие дикторов)."""
    name = "silero31"
    voices = ["rnd1", "rnd2", "rnd3", "rnd4", "rnd5", "rnd6"]

    def __init__(self):
        import torch
        torch.set_num_threads(4)
        from silero import silero_tts
        self.model, _ = silero_tts(language='ru', speaker='v3_1_ru')
        self.model.to(torch.device("cpu"))

    def synth(self, text, voice, rate=None):
        rate = rate or random.choice(["x-slow", "slow", "medium", "medium", "fast", "x-fast"])
        pitch = random.choice(["x-low", "low", "medium", "medium", "high", "x-high"])
        ssml = f"<speak><prosody rate='{rate}' pitch='{pitch}'>{text}</prosody></speak>"
        try:
            a = self.model.apply_tts(ssml_text=ssml, speaker="random", sample_rate=24000)
        except Exception:
            a = self.model.apply_tts(text=text.replace("+", ""), speaker="random", sample_rate=24000)
        return resample(a.numpy().astype(np.float32), 24000)


class Piper:
    name = "piper"
    voices = ["irina", "denis", "dmitri", "ruslan"]

    def __init__(self):
        from piper import PiperVoice
        self.v = {n: PiperVoice.load(os.path.join(TTS_DIR, f"ru_RU-{n}-medium.onnx")) for n in self.voices}

    def synth(self, text, voice, rate=None):
        ls = rate or random.choice([0.8, 0.9, 1.0, 1.0, 1.1, 1.25])
        v = self.v[voice]
        text = text.replace("+", "")
        try:
            from piper import SynthesisConfig
            cfg = SynthesisConfig(length_scale=ls, noise_scale=random.uniform(0.5, 0.8), noise_w_scale=random.uniform(0.6, 1.0))
            chunks = list(v.synthesize(text, syn_config=cfg))
            pcm = b"".join(c.audio_int16_bytes for c in chunks)
            sr = chunks[0].sample_rate if chunks else v.config.sample_rate
        except ImportError:
            pcm = b"".join(v.synthesize_stream_raw(text, length_scale=ls))
            sr = v.config.sample_rate
        x = np.frombuffer(pcm, np.int16).astype(np.float32) / 32768
        return resample(x, sr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "siesta", "raw"))
    ap.add_argument("--per-voice", type=int, default=40, help="клипов ключевого слова на голос и слово")
    ap.add_argument("--neg-sent-per-voice", type=int, default=40)
    ap.add_argument("--backends", default="say,silero,piper")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    random.seed(a.seed)
    os.makedirs(a.out, exist_ok=True)
    man = open(os.path.join(a.out, "manifest.jsonl"), "a", encoding="utf-8")
    backends = []
    for b in a.backends.split(","):
        try:
            backends.append({"say": Say, "silero": Silero, "silero31": Silero31, "piper": Piper}[b]())
            print("backend ok:", b, flush=True)
        except Exception as e:
            print("backend FAIL:", b, e, flush=True)

    def emit(kind, be, voice, text, x, tag):
        x = trim(x)
        if len(x) < SR * 0.15:
            return
        p = os.path.join(a.out, kind, f"{be.name}-{voice}-{tag}.wav")
        write_wav(p, x)
        man.write(json.dumps({"path": p, "kind": kind, "voice": f"{be.name}-{voice}", "text": text, "dur": len(x) / SR}, ensure_ascii=False) + "\n")

    n = 0
    for be in backends:
        for voice in be.voices:
            # ключевые слова: отдельным клипом, с вариациями темпа/интонации
            for kw, variants in KEYWORDS.items():
                for i in range(a.per_voice):
                    text = random.choice(variants)
                    try:
                        emit(f"pos_{kw}", be, voice, text, be.synth(text, voice), f"{i:03d}")
                    except Exception as e:
                        print("err", be.name, voice, text, e, flush=True)
                    n += 1
            # трудные негативы: каждое слово 2 раза
            for i, w in enumerate(NEG_WORDS):
                for r in range(2):
                    try:
                        emit("neg_word", be, voice, w, be.synth(w, voice), f"{i:03d}-{r}")
                    except Exception as e:
                        print("err", be.name, voice, w, e, flush=True)
            # предложения и префиксы
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
    print("total keyword clips requested:", n)


if __name__ == "__main__":
    main()
