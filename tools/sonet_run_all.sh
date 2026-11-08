#!/bin/zsh
# SPDX-License-Identifier: Apache-2.0
# Слить манифесты трёх генераторов и обучить run1. Запуск: zsh tools/sonet_run_all.sh [имя_прогона] [эпохи]
set -e
REPO=${0:A:h:h}                       # корень репозитория (tools/, deploy/)
PY=${PYTHON:-python3}                 # интерпретатор окружения с TensorFlow/LiteRT
export WAKEWORD_ROOT=${WAKEWORD_ROOT:-./work}   # данные, прогоны (sonet/...)
mkdir -p $WAKEWORD_ROOT; export WAKEWORD_ROOT=${WAKEWORD_ROOT:A}
cd $WAKEWORD_ROOT
RUN=${1:-run1}; EPOCHS=${2:-30}
mkdir -p sonet/raw_all
cat sonet/raw/manifest.jsonl sonet/raw_silero/manifest.jsonl sonet/raw_piper/manifest.jsonl 2>/dev/null > sonet/raw_all/manifest.jsonl
echo "manifest: $(wc -l < sonet/raw_all/manifest.jsonl) строк"
$PY -I - <<'EOF'
import json, collections
c = collections.Counter(); v = collections.Counter()
for l in open("sonet/raw_all/manifest.jsonl", encoding="utf-8"):
    it = json.loads(l); c[it["kind"]] += 1; v[it["voice"]] += 1
print(dict(c)); print(dict(v))
EOF
$PY -I $REPO/tools/sonet_train.py --raw $WAKEWORD_ROOT/sonet/raw_all --out $WAKEWORD_ROOT/sonet/$RUN --epochs $EPOCHS --steps 150 --batch 48 --holdout-voice piper-denis 2>&1 | grep -v -E "^WARNING|oneDNN|cpu_feature|rebuild TensorFlow|I0000|W0000" | tee sonet/$RUN.log
