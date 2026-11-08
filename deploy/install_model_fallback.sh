#!/bin/zsh
# SPDX-License-Identifier: Apache-2.0
# Вариант В: наша модель в заводской каталог rootfs + блокировка скачанной кастомной модели.
#   zsh install_model_fallback.sh <IP колонки>     (или STATION_IP=<IP>)
# Откат: zsh rollback_model_fallback.sh <IP колонки>
# MODEL_DIR - каталог с model.tflite/flags.txt/words.txt (по умолчанию deploy/model-dir рядом),
# WAKEWORD_ROOT - куда класть бэкапы (по умолчанию ./work).
set -e
IP=${1:-$STATION_IP}
[ -n "$IP" ] || { echo "укажите IP колонки аргументом или в STATION_IP"; exit 1; }
MODEL_DIR=${MODEL_DIR:-${0:A:h}/model-dir}
WR=${${WAKEWORD_ROOT:-./work}:A}       # рабочий каталог: бэкапы с колонки
export PATH=/opt/homebrew/bin:$PATH
adb connect $IP:5555 >/dev/null
A() { adb -s $IP:5555 "$@"; }
A shell 'id' | grep -q uid=0 || { echo "нет root по ADB"; exit 1; }
STAMP=$(date +%Y%m%d-%H%M%S); B=$WR/deploy/backup-fallback-$STAMP; mkdir -p $B
echo "== бэкап в $B"
A pull /data/quasar/data/spotter.json $B/ >/dev/null
A pull /system/vendor/quasar/spotter_models/activation/alisa $B/ >/dev/null
echo "== свежие файлы модели -> /data/opt/sonet-model (всегда из $MODEL_DIR)"
A shell 'mkdir -p /data/opt/sonet-model'
A push $MODEL_DIR/model.tflite $MODEL_DIR/flags.txt $MODEL_DIR/words.txt /data/opt/sonet-model/ >/dev/null
A shell 'md5sum /data/opt/sonet-model/model.tflite; grep ^version /data/opt/sonet-model/flags.txt'
A shell 'ls /data/opt/sonet-model' | grep -q model.tflite || { echo "нет /data/opt/sonet-model"; exit 1; }
echo "== rootfs rw: наша модель в заводской каталог"
A shell 'mount -o remount,rw / && D=/system/vendor/quasar/spotter_models/activation/alisa && mkdir -p /data/opt/sonet-backup-sys && cp -a $D/flags.txt /data/opt/sonet-backup-sys/flags.txt && cp -a $D/words.txt /data/opt/sonet-backup-sys/words.txt && cp /data/opt/sonet-model/model.tflite /data/opt/sonet-model/flags.txt /data/opt/sonet-model/words.txt $D/ && sync && mount -o remount,ro / && ls -la $D'
echo "== убрать кастомную активационную модель и запретить её скачивание"
A shell "cat /data/quasar/data/spotter.json" | python3 -c '
import json,sys
d=json.load(sys.stdin)
d["models"]=[m for m in d["models"] if m["type"]!="activation/alisa"]
print(json.dumps(d, ensure_ascii=False))' > /tmp/spotter.json.noact
# при повторном запуске файлы уже закреплены (chattr +i) - снять, иначе push упадёт и до reboot не дойдём
A shell 'chattr -i /data/quasar/data/spotter.json /data/quasar/data/spotter_model 2>/dev/null; true'
A push /tmp/spotter.json.noact /data/quasar/data/spotter.json >/dev/null
A shell 'cd /data/quasar/data/spotter_model && for d in */; do if [ -f "$d/model.tflite" ] && [ -d "$d/tts-timings-info" ]; then rm -rf "$d"; fi; done; chattr +i /data/quasar/data/spotter.json; chattr +i /data/quasar/data/spotter_model; lsattr -d /data/quasar/data/spotter.json /data/quasar/data/spotter_model; ls /data/quasar/data/spotter_model'
echo "== перезагрузка (прозвучит приветствие)"
A shell reboot || true
