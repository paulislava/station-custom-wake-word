#!/bin/zsh
# SPDX-License-Identifier: Apache-2.0
# Дообучение на реальных записях. Использование:
#   zsh tools/sonet_finetune_real.sh "<oga-файлы Сонет через пробел>" "<oga Алиса>" "<oga без слов>" run2
# Пути к записям - абсолютные (скрипт переходит в $WAKEWORD_ROOT).
set -e
REPO=${0:A:h:h}                       # корень репозитория (tools/, deploy/)
PY=${PYTHON:-python3}                 # интерпретатор окружения с TensorFlow/LiteRT
export WAKEWORD_ROOT=${WAKEWORD_ROOT:-./work}   # данные, прогоны (sonet/...)
mkdir -p $WAKEWORD_ROOT; export WAKEWORD_ROOT=${WAKEWORD_ROOT:A}
cd $WAKEWORD_ROOT
SON=$1; ALI=$2; NEG=$3; RUN=${4:-run2}
rm -rf sonet/real; mkdir -p sonet/real
[ -n "$SON" ] && $PY -I $REPO/tools/segment_voice.py --label pos_sonet --out sonet/real ${=SON}
[ -n "$ALI" ] && $PY -I $REPO/tools/segment_voice.py --label pos_alisa --out sonet/real ${=ALI}
[ -n "$NEG" ] && $PY -I $REPO/tools/segment_voice.py --label neg_sent --out sonet/real --whole ${=NEG}
$PY -I - <<'PY'
import json, glob, os, wave
out = open("sonet/real/manifest.jsonl", "w", encoding="utf-8")
n = 0
for kind in ("pos_sonet", "pos_alisa", "neg_sent"):
    for p in sorted(glob.glob(f"sonet/real/{kind}/*.wav")):
        w = wave.open(p); dur = w.getnframes() / w.getframerate()
        # реальные клипы дублируем в манифесте x6, чтобы весили больше синтеза
        for r in range(6):
            out.write(json.dumps({"path": os.path.abspath(p), "kind": kind, "voice": "real-user", "text": kind, "dur": dur}, ensure_ascii=False) + "\n"); n += 1
print("real manifest:", n)
PY
# сначала оценка run1 на реальных записях
echo "== run1 на реальных:"; $PY -I $REPO/tools/sonet_eval.py sonet/run1/model.tflite --cls 2 --pos sonet/real/pos_sonet --neg sonet/real/neg_sent sonet/real/pos_alisa 2>&1 | grep -v XNNPACK | tail -3
$PY -I $REPO/tools/sonet_train.py --raw sonet/raw_all --extra-raw sonet/real/manifest.jsonl --init-weights sonet/run1/best.weights.h5 --lr 3e-4 --epochs 8 --steps 150 --out sonet/$RUN 2>&1 | grep -v -E "^WARNING|oneDNN|cpu_feature|rebuild TensorFlow|I0000|W0000" | tee sonet/$RUN.log
echo "== $RUN на реальных:"; $PY -I $REPO/tools/sonet_eval.py sonet/$RUN/model.tflite --cls 2 --pos sonet/real/pos_sonet --neg sonet/real/neg_sent sonet/real/pos_alisa 2>&1 | grep -v XNNPACK | tail -3
