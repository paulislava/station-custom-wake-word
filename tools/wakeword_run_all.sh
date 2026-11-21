#!/bin/bash
# wakeword_run_all.sh — полный пайплайн обучения wake word модели.
# Запуск: bash tools/wakeword_run_all.sh <word> [epochs]
#   bash tools/wakeword_run_all.sh votchi 40
set -e
WORD=${1:?Usage: $0 <word> [epochs]}
EPOCHS=${2:-30}
REPO=$(cd "$(dirname "$0")/.." && pwd)
export WAKEWORD_ROOT=${WAKEWORD_ROOT:-./work}
mkdir -p "$WAKEWORD_ROOT"; export WAKEWORD_ROOT=$(cd "$WAKEWORD_ROOT" && pwd)

echo "=== Wake word pipeline: $WORD ($EPOCHS epochs) ==="
echo "WAKEWORD_ROOT=$WAKEWORD_ROOT"

# Шаг 1: Данные (venv_tts)
echo ""
echo "[1/5] Генерация данных (venv_tts)..."
source venv_tts/bin/activate 2>/dev/null || true
python "$REPO/tools/wakeword_data.py" --word "$WORD" --per-voice 40
python "$REPO/tools/wakeword_sentences.py" --word "$WORD" --per-voice 50
python "$REPO/tools/wakeword_testset.py" --word "$WORD"
deactivate 2>/dev/null || true

# Шаг 2: Обучение (venv312)
echo ""
echo "[2/5] Обучение (venv312)..."
source venv312/bin/activate 2>/dev/null || true
python "$REPO/tools/wakeword_train.py" --word "$WORD" --epochs "$EPOCHS"
deactivate 2>/dev/null || true

# Шаг 3: Отбор лучшей эпохи (venv312)
echo ""
echo "[3/5] Отбор лучшей эпохи..."
source venv312/bin/activate 2>/dev/null || true
python "$REPO/tools/wakeword_select.py" --word "$WORD" --run "$WAKEWORD_ROOT/$WORD/run1"
deactivate 2>/dev/null || true

# Шаг 4: Экспорт engine int8
echo ""
echo "[4/5] Экспорт int8 engine..."
source venv312/bin/activate 2>/dev/null || true
python "$REPO/tools/wakeword_export_engine.py" --word "$WORD" --run "$WAKEWORD_ROOT/$WORD/run1"
deactivate 2>/dev/null || true

# Шаг 5: Упаковка
echo ""
echo "[5/5] Упаковка модели..."
bash "$REPO/tools/wakeword_pack.sh" "$WORD" run1

echo ""
echo "=== Готово! Модель $WORD в deploy/model-dir/ ==="
echo "Установка на колонку:"
echo "  scp -r deploy/model-dir root@mini:/data/opt/models/$WORD/"
echo "  ssh root@mini 'sh /data/opt/models/switch_model.sh $WORD'"
