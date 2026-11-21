#!/bin/sh
# switch_model.sh — переключение wake word моделей на Яндекс Станции Мини 2.
#
# Хранит модели в /data/opt/models/<word>/ и активирует выбранную,
# копируя её в заводской каталог споттера.
# Заводская модель сохраняется в /data/opt/models/.backup-factory/ при первом запуске.
#
# Usage: switch_model.sh <word>
#   switch_model.sh siesta     # активировать модель «Сиеста»
#   switch_model.sh votchi     # активировать модель «Вотчи»
#   switch_model.sh alisa      # восстановить заводскую модель «Алиса»
#   switch_model.sh list       # показать установленные модели
#
# Требует: root, mount -o remount,rw /
# После переключения нужен reboot для применения.

MODELS_DIR="/data/opt/models"
FACTORY_BACKUP="$MODELS_DIR/.backup-factory"
SPOTTER_DIR="/system/vendor/quasar/spotter_models/activation/alisa"

WORD=$1

if [ -z "$WORD" ]; then
    echo "Usage: $0 <word|alisa|list>"
    echo "  $0 siesta    # активировать «Сиеста»"
    echo "  $0 votchi    # активировать «Вотчи»"
    echo "  $0 miku      # активировать «Мику»"
    echo "  $0 smith     # активировать «Смит»"
    echo "  $0 alisa     # восстановить заводскую «Алиса»"
    echo "  $0 list      # показать установленные модели"
    exit 1
fi

if [ "$WORD" = "list" ]; then
    echo "=== Установленные модели ==="
    if [ -d "$MODELS_DIR" ]; then
        for d in "$MODELS_DIR"/*/; do
            name=$(basename "$d")
            if [ "$name" = ".backup-factory" ]; then continue; fi
            if [ -f "$d/model.tflite" ]; then
                echo "  $name: $(ls -lh "$d/model.tflite" | awk '{print $5}')"
                if [ -f "$d/words.txt" ]; then
                    echo "    words: $(cat "$d/words.txt" | tr '\n' ' ')"
                fi
            fi
        done
    else
        echo "  (нет моделей)"
    fi
    if [ -d "$FACTORY_BACKUP" ]; then
        echo "=== Заводская модель (бэкап) ==="
        echo "  $(ls -lh "$FACTORY_BACKUP/model.tflite" 2>/dev/null | awk '{print $5}')"
    else
        echo "=== Заводская модель: бэкап НЕ создан ==="
    fi
    # Показать активную
    if [ -f "$SPOTTER_DIR/words.txt" ]; then
        echo "=== Активная модель ==="
        echo "  words: $(cat "$SPOTTER_DIR/words.txt" | tr '\n' ' ')"
    fi
    exit 0
fi

# Проверка root
if [ "$(id -u)" -ne 0 ]; then
    echo "ERROR: требуется root"
    exit 1
fi

# Создать бэкап заводской модели при первом запуске
if [ ! -d "$FACTORY_BACKUP" ]; then
    echo "Создаю бэкап заводской модели..."
    mkdir -p "$FACTORY_BACKUP"
    if [ -f "$SPOTTER_DIR/model.tflite" ]; then
        cp "$SPOTTER_DIR/model.tflite" "$FACTORY_BACKUP/model.tflite"
        cp "$SPOTTER_DIR/flags.txt" "$FACTORY_BACKUP/flags.txt"
        cp "$SPOTTER_DIR/words.txt" "$FACTORY_BACKUP/words.txt"
        echo "  заводская модель сохранена в $FACTORY_BACKUP"
    else
        echo "  WARNING: заводская модель не найдена в $SPOTTER_DIR"
    fi
fi

if [ "$WORD" = "alisa" ]; then
    echo "Восстанавливаю заводскую модель «Алиса»..."
    SRC="$FACTORY_BACKUP"
else
    SRC="$MODELS_DIR/$WORD"
fi

if [ ! -f "$SRC/model.tflite" ]; then
    echo "ERROR: модель не найдена: $SRC/model.tflite"
    echo "Установите модель через install_model_fallback.sh или wakeword_pack.sh + scp"
    exit 1
fi

# Перемонтировать / в read-write
echo "Перемонтирую / в read-write..."
mount -o remount,rw / || { echo "ERROR: не удалось remount"; exit 1; }

# Копировать модель
echo "Устанавливаю модель «$WORD»..."
cp "$SRC/model.tflite" "$SPOTTER_DIR/model.tflite"
cp "$SRC/flags.txt" "$SPOTTER_DIR/flags.txt"
cp "$SRC/words.txt" "$SPOTTER_DIR/words.txt"

# Вернуть read-only
mount -o remount,ro /

echo ""
echo "Модель «$WORD» установлена."
echo "words: $(cat "$SPOTTER_DIR/words.txt" | tr '\n' ' ')"
echo ""
echo "Для применения перезагрузите колонку:"
echo "  reboot"
