#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Находит конец ключевого слова в клипах «в предложении» через whisper (word timestamps) и
дописывает в manifest поле kw_end (секунды). Клипы, где слово не найдено, помечаются kw_end=null
и в обучение как позитивы не идут.

Запуск интерпретатором, в котором установлен пакет openai-whisper:
  python siesta_align.py "$WAKEWORD_ROOT"/siesta/raw_sent/manifest.jsonl
"""
import json, re, sys, os
import whisper

VARIANTS = {"сиеста": ["сиеста", "соне", "сонэт", "siesta", "сиестат", "санет"], "алиса": ["алиса", "алис", "alisa", "олиса"]}


def norm(w):
    return re.sub(r"[^а-яёa-z]", "", w.lower())


def main(manifest, model_name="small"):
    model = whisper.load_model(model_name)
    items = [json.loads(l) for l in open(manifest, encoding="utf-8")]
    out = open(manifest + ".aligned", "w", encoding="utf-8")
    found = 0; total = 0
    for it in items:
        if it["kind"].startswith("sent_") and it.get("kw_end") is None:
            total += 1
            kw = it["kw"]
            try:
                r = model.transcribe(it["path"], language="ru", word_timestamps=True, fp16=False, temperature=0.0,
                                     initial_prompt="Сиеста. Алиса. Сиеста, включи свет.")
                words = [w for s in r["segments"] for w in s.get("words", [])]
                hit = None
                for w in words:
                    if any(norm(w["word"]).startswith(v) for v in VARIANTS[kw]):
                        hit = w; break
                if hit:
                    it["kw_end"] = round(float(hit["end"]), 3); it["kw_start"] = round(float(hit["start"]), 3); found += 1
                else:
                    it["kw_end"] = None; it["asr"] = " ".join(norm(w["word"]) for w in words)
            except Exception as e:
                it["kw_end"] = None; it["asr_err"] = str(e)[:100]
        out.write(json.dumps(it, ensure_ascii=False) + "\n")
        if total and total % 50 == 0:
            print(f"{total} обработано, найдено {found}", flush=True)
    out.close()
    os.replace(manifest + ".aligned", manifest)
    print(f"готово: найдено слово в {found} из {total}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "small")
