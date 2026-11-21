#!/bin/zsh
# SPDX-License-Identifier: Apache-2.0
# run2: дообучение run1 на всех данных + фразы с ключевым словом внутри (kw_end по паузам)
set -e
REPO=${0:A:h:h}                       # корень репозитория (tools/, deploy/)
PY=${PYTHON:-python3}                 # интерпретатор окружения с TensorFlow/LiteRT
export WAKEWORD_ROOT=${WAKEWORD_ROOT:-./work}   # данные, прогоны (siesta/...)
mkdir -p $WAKEWORD_ROOT; export WAKEWORD_ROOT=${WAKEWORD_ROOT:A}
cd $WAKEWORD_ROOT
RUN=${1:-run2}; EPOCHS=${2:-15}; INIT=${3:-siesta/run1/best.weights.h5}
$PY -I $REPO/tools/siesta_train.py --raw siesta/raw_all --extra-raw siesta/raw_sent/manifest.jsonl,siesta/raw_sent_say/manifest.jsonl \
  --init-weights $INIT --lr 5e-4 --epochs $EPOCHS --steps 150 --batch 48 --holdout-voice piper-denis --out siesta/$RUN \
  2>&1 | grep --line-buffered -v -E "^WARNING|oneDNN|cpu_feature|rebuild TensorFlow|I0000|W0000" | tee siesta/$RUN.log
for thr in 0.5 0.69; do echo "== $RUN test thr $thr (сиеста)"; $PY -I $REPO/tools/siesta_eval.py siesta/$RUN/model.tflite --cls 2 --thr $thr --pos siesta/test/pos_siesta --neg siesta/test/neg_sent siesta/test/pos_alisa 2>&1 | grep -v XNNPACK | tail -3; done
echo "== $RUN test алиса 0.69"; $PY -I $REPO/tools/siesta_eval.py siesta/$RUN/model.tflite --cls 1 --thr 0.69 --pos siesta/test/pos_alisa --neg siesta/test/neg_sent siesta/test/pos_siesta 2>&1 | grep -v XNNPACK | tail -3
echo "== $RUN длинные негативы"; $PY -I $REPO/tools/siesta_eval.py siesta/$RUN/model.tflite --cls 2 --thr 0.69 --events siesta/test_long_neg2.wav siesta/test_long_neg.wav 2>&1 | grep -v XNNPACK
