#!/bin/zsh
# SPDX-License-Identifier: Apache-2.0
# run2: дообучение run1 на всех данных + фразы с ключевым словом внутри (kw_end по паузам)
set -e
REPO=${0:A:h:h}                       # корень репозитория (tools/, deploy/)
PY=${PYTHON:-python3}                 # интерпретатор окружения с TensorFlow/LiteRT
export WAKEWORD_ROOT=${WAKEWORD_ROOT:-./work}   # данные, прогоны (sonet/...)
mkdir -p $WAKEWORD_ROOT; export WAKEWORD_ROOT=${WAKEWORD_ROOT:A}
cd $WAKEWORD_ROOT
RUN=${1:-run2}; EPOCHS=${2:-15}; INIT=${3:-sonet/run1/best.weights.h5}
$PY -I $REPO/tools/sonet_train.py --raw sonet/raw_all --extra-raw sonet/raw_sent/manifest.jsonl,sonet/raw_sent_say/manifest.jsonl \
  --init-weights $INIT --lr 5e-4 --epochs $EPOCHS --steps 150 --batch 48 --holdout-voice piper-denis --out sonet/$RUN \
  2>&1 | grep --line-buffered -v -E "^WARNING|oneDNN|cpu_feature|rebuild TensorFlow|I0000|W0000" | tee sonet/$RUN.log
for thr in 0.5 0.69; do echo "== $RUN test thr $thr (сонет)"; $PY -I $REPO/tools/sonet_eval.py sonet/$RUN/model.tflite --cls 2 --thr $thr --pos sonet/test/pos_sonet --neg sonet/test/neg_sent sonet/test/pos_alisa 2>&1 | grep -v XNNPACK | tail -3; done
echo "== $RUN test алиса 0.69"; $PY -I $REPO/tools/sonet_eval.py sonet/$RUN/model.tflite --cls 1 --thr 0.69 --pos sonet/test/pos_alisa --neg sonet/test/neg_sent sonet/test/pos_sonet 2>&1 | grep -v XNNPACK | tail -3
echo "== $RUN длинные негативы"; $PY -I $REPO/tools/sonet_eval.py sonet/$RUN/model.tflite --cls 2 --thr 0.69 --events sonet/test_long_neg2.wav sonet/test_long_neg.wav 2>&1 | grep -v XNNPACK
