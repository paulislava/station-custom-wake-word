#!/bin/zsh
# SPDX-License-Identifier: Apache-2.0
# Откат варианта В: снять immutable, вернуть заводские flags/words, удалить наш tflite из rootfs, вернуть spotter.json.
#   zsh rollback_model_fallback.sh <IP колонки> [каталог бэкапа]   (по умолчанию - последний в $WAKEWORD_ROOT/deploy)
set -e
IP=${1:-$STATION_IP}
[ -n "$IP" ] || { echo "укажите IP колонки аргументом или в STATION_IP"; exit 1; }
WR=${${WAKEWORD_ROOT:-./work}:A}       # рабочий каталог: бэкапы с колонки
export PATH=/opt/homebrew/bin:$PATH
adb connect $IP:5555 >/dev/null
A() { adb -s $IP:5555 "$@"; }
B=${2:-$(ls -d $WR/deploy/backup-fallback-* | tail -1)}
echo "== откат из $B"
A shell 'chattr -i /data/quasar/data/spotter.json /data/quasar/data/spotter_model 2>/dev/null; true'
A push $B/spotter.json /data/quasar/data/spotter.json >/dev/null
A shell 'mount -o remount,rw / && D=/system/vendor/quasar/spotter_models/activation/alisa && cp /data/opt/sonet-backup-sys/flags.txt $D/flags.txt && cp /data/opt/sonet-backup-sys/words.txt $D/words.txt && rm -f $D/model.tflite && sync && mount -o remount,ro / && ls -la $D'
A shell reboot || true
echo "откат отправлен"
