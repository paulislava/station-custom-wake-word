#!/bin/zsh
# SPDX-License-Identifier: Apache-2.0
# Облачная перепроверка слова активации: выключить для типа activation (иначе облако, зная только «Алису»,
# отменяет запрос после «Сиеста»). Правится локальный /system/vendor/quasar/quasar.cfg:
#   voiceDialogSettings.onlineSpotterValidationDisabledTypes: ["noise-detection","vad"] -> + "activation"
# Облачный system_config эту графу не несёт (проверено 06.11.2026), поэтому локальное значение остаётся в силе.
#   zsh disable_online_validation.sh <IP> apply     # правка + перезагрузка (колонка прозвучит)
#   zsh disable_online_validation.sh <IP> restore   # вернуть оригинал из бэкапа + перезагрузка
#   zsh disable_online_validation.sh <IP> show      # только показать текущее значение
set -e
IP=${1:-$STATION_IP}; MODE=${2:-show}
[ -n "$IP" ] || { echo "укажите IP колонки аргументом или в STATION_IP"; exit 1; }
WR=${${WAKEWORD_ROOT:-./work}:A}       # рабочий каталог: бэкапы с колонки
export PATH=/opt/homebrew/bin:$PATH
adb connect $IP:5555 >/dev/null
A() { adb -s $IP:5555 "$@"; }
CFG=/system/vendor/quasar/quasar.cfg
mkdir -p $WR/deploy; BK=$WR/deploy/quasar.cfg.orig   # бэкап оригинала (в репозиторий не класть)
W=/tmp/qcfg-work; mkdir -p $W
A pull $CFG $W/quasar.cfg >/dev/null
case $MODE in
  show)
    grep -o -E '"onlineSpotterValidationDisabledTypes" : \[[^]]*\]' $W/quasar.cfg ;;
  apply)
    [ -f $BK ] || cp $W/quasar.cfg $BK
    python3 - $W/quasar.cfg $W/quasar.cfg.new <<'EOF'
import re,sys
src,dst=sys.argv[1],sys.argv[2]
t=open(src,encoding="utf-8").read()
m=re.search(r'("onlineSpotterValidationDisabledTypes"\s*:\s*\[)([^\]]*)(\])',t)
assert m, "ключ не найден"
items=[x.strip().strip('"') for x in m.group(2).split(",") if x.strip()]
if "activation" not in items: items.append("activation")
new=m.group(1)+" "+", ".join(f'"{x}"' for x in items)+" "+m.group(3)
open(dst,"w",encoding="utf-8").write(t[:m.start()]+new+t[m.end():])
print("new:",new)
EOF
    python3 -c "import json,sys; json.load(open('$W/quasar.cfg.new',encoding='utf-8')); print('json ok')" 2>/dev/null || echo "предупреждение: cfg не чистый JSON (есть \${VARS}), проверка пропущена"
    A push $W/quasar.cfg.new /data/opt/quasar.cfg.new >/dev/null
    A shell "mount -o remount,rw / && cp /data/opt/quasar.cfg.new $CFG && sync && mount -o remount,ro /; grep -o -E '\"onlineSpotterValidationDisabledTypes\" : \[[^]]*\]' $CFG"
    echo "== перезагрузка (прозвучит приветствие)"
    A shell reboot || true ;;
  restore)
    [ -f $BK ] || { echo "нет бэкапа $BK"; exit 1; }
    A push $BK /data/opt/quasar.cfg.orig >/dev/null
    A shell "mount -o remount,rw / && cp /data/opt/quasar.cfg.orig $CFG && sync && mount -o remount,ro /; grep -o -E '\"onlineSpotterValidationDisabledTypes\" : \[[^]]*\]' $CFG"
    echo "== перезагрузка"
    A shell reboot || true ;;
esac
