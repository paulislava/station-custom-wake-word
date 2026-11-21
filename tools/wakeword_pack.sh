#!/bin/zsh
# SPDX-License-Identifier: Apache-2.0
# Собрать каталог модели для колонки из выбранного прогона wake word.
# Usage: zsh tools/wakeword_pack.sh <word> [run_dir] [variant]
#   zsh tools/wakeword_pack.sh votchi run1           # 3 класса (алиса+votchi)
#   zsh tools/wakeword_pack.sh votchi run1 votchi2   # запасной: только votchi (2 класса)
set -e
REPO=${0:A:h:h}
PY=${PYTHON:-python3}
export WAKEWORD_ROOT=${WAKEWORD_ROOT:-./work}
mkdir -p $WAKEWORD_ROOT; export WAKEWORD_ROOT=${WAKEWORD_ROOT:A}
cd $WAKEWORD_ROOT

WORD=$1; RUN=${2:-run1}; VAR=${3:-3cls}
if [ -z "$WORD" ]; then echo "Usage: $0 <word> [run] [variant]"; exit 1; fi

# Поддерживаемые слова и их русские написания
case "$WORD" in
  siesta) WORD_RU="сиеста" ;;
  votchi) WORD_RU="вотчи" ;;
  miku)   WORD_RU="мику" ;;
  smith)  WORD_RU="смит" ;;
  *) echo "Unknown word: $WORD"; exit 1 ;;
esac

D=${MODEL_DIR:-$REPO/deploy/model-dir}; mkdir -p $D
[ -f $D/flags.txt ] || { echo "нет $D/flags.txt: снимите заводской flags.txt с колонки (README, шаг 4)" >&2; exit 1; }

if [ "$VAR" = "${WORD}2" ]; then
  cp $WORD/$RUN/model_${WORD}2.tflite $D/model.tflite
  cp $WORD/$RUN/words_${WORD}2.txt $D/words.txt
  sed -i '' -e 's/^commands .*/commands 1/' \
            -e "s/^string-representations .*/string-representations ${WORD_RU}/" \
            -e 's/^thresholds .*/thresholds 0.691/' \
            -e 's/^subthresholds .*/subthresholds 0.435/' \
            -e "s/^version .*/version ru-RU-custom-${WORD}-only-$RUN/" $D/flags.txt
else
  SRC=$WORD/$RUN/model_engine_int8.tflite; [ -f $SRC ] || SRC=$WORD/$RUN/model.tflite
  $PY -I $REPO/tools/tfl_add_metadata.py $SRC $WORD/$RUN/model_engine_int8_meta.tflite --right-context 16 --min-runtime 1.15.0 --fix-io 2>&1 | grep -E "right_context|io order|байт"
  cp $WORD/$RUN/model_engine_int8_meta.tflite $D/model.tflite
  cp $WORD/$RUN/words.txt $D/words.txt
  sed -i '' -e 's/^commands .*/commands 1:2/' \
            -e "s/^string-representations .*/string-representations алиса:${WORD_RU}/" \
            -e 's/^thresholds .*/thresholds 0.4:0.5/' \
            -e 's/^subthresholds .*/subthresholds 0.25:0.35/' \
            -e 's/^use-tts-blocker .*/use-tts-blocker false/' \
            -e "s/^version .*/version ru-RU-custom-${WORD}-alisa-$RUN-int8-io/" $D/flags.txt
fi
$PY -I -c "
from ai_edge_litert.interpreter import Interpreter; it=Interpreter(model_path='$D/model.tflite'); s=it.get_signature_list()['serving_default']; print('signature ok:', s['inputs'][:2], '...', s['outputs'][:1])"
ls -la $D; cat $D/words.txt; grep -E "^(commands|string-representations|thresholds|version)" $D/flags.txt
