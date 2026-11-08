#!/bin/zsh
# SPDX-License-Identifier: Apache-2.0
# Собрать каталог модели для колонки из выбранного прогона (по умолчанию deploy/model-dir;
# MODEL_DIR - другой каталог). COPY_HOST=<user@host> [COPY_DIR=<путь>] - дополнительно
# скопировать каталог на машину в сети колонки.
#   zsh tools/sonet_pack.sh run2            # 3 класса (алиса+сонет)
#   zsh tools/sonet_pack.sh run2 sonet2     # запасной: только сонет (2 класса)
set -e
REPO=${0:A:h:h}                       # корень репозитория (tools/, deploy/)
PY=${PYTHON:-python3}                 # интерпретатор окружения с TensorFlow/LiteRT
export WAKEWORD_ROOT=${WAKEWORD_ROOT:-./work}   # данные, прогоны (sonet/...)
mkdir -p $WAKEWORD_ROOT; export WAKEWORD_ROOT=${WAKEWORD_ROOT:A}
cd $WAKEWORD_ROOT
RUN=${1:-run2}; VAR=${2:-3cls}
D=${MODEL_DIR:-$REPO/deploy/model-dir}; mkdir -p $D
# flags.txt - заводской файл TFLite-модели «Алиса» с вашей колонки (файл прошивки, в репозитории его нет), см. README шаг 4
[ -f $D/flags.txt ] || { echo "нет $D/flags.txt: снимите заводской flags.txt с колонки (README, шаг 4)" >&2; exit 1; }
if [ "$VAR" = "sonet2" ]; then
  cp sonet/$RUN/model_sonet2.tflite $D/model.tflite; cp sonet/$RUN/words_sonet2.txt $D/words.txt
  sed -i '' -e 's/^commands .*/commands 1/' -e 's/^string-representations .*/string-representations сонет/' -e 's/^thresholds .*/thresholds 0.691/' -e 's/^subthresholds .*/subthresholds 0.435/' -e "s/^version .*/version ru-RU-custom-sonet-only-$RUN/" $D/flags.txt
else
  SRC=sonet/$RUN/model_engine_int8.tflite; [ -f $SRC ] || SRC=sonet/$RUN/model.tflite
  # движок колонки требует metadata right_context и заводской порядок/имена входов-выходов (см. tfl_add_metadata.py)
  $PY -I $REPO/tools/tfl_add_metadata.py $SRC sonet/$RUN/model_engine_int8_meta.tflite --right-context 16 --min-runtime 1.15.0 --fix-io 2>&1 | grep -E "right_context|io order|байт"
  cp sonet/$RUN/model_engine_int8_meta.tflite $D/model.tflite; cp sonet/$RUN/words.txt $D/words.txt
  sed -i '' -e 's/^commands .*/commands 1:2/' -e 's/^string-representations .*/string-representations алиса:сонет/' -e 's/^thresholds .*/thresholds 0.4:0.5/' -e 's/^subthresholds .*/subthresholds 0.25:0.35/' -e 's/^use-tts-blocker .*/use-tts-blocker false/' -e "s/^version .*/version ru-RU-custom-sonet-alisa-$RUN-int8-io/" $D/flags.txt
fi
$PY -I -c "
from ai_edge_litert.interpreter import Interpreter; it=Interpreter(model_path='$D/model.tflite'); s=it.get_signature_list()['serving_default']; print('signature ok:', s['inputs'][:2], '...', s['outputs'][:1])"
ls -la $D; cat $D/words.txt; grep -E "^(commands|string-representations|thresholds|version)" $D/flags.txt
if [ -n "$COPY_HOST" ]; then
  CDIR=${COPY_DIR:-deploy/model-dir}
  ssh -o BatchMode=yes $COPY_HOST "mkdir -p $CDIR"
  scp -q $D/model.tflite $D/flags.txt $D/words.txt $COPY_HOST:$CDIR/
  ssh -o BatchMode=yes $COPY_HOST "ls -la $CDIR"
fi
