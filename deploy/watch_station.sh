#!/bin/zsh
# SPDX-License-Identifier: Apache-2.0
# Живой лог споттера колонки во время теста (запуск с компьютера в сети колонки):
#   zsh watch_station.sh <IP колонки>     (или STATION_IP=<IP>)
IP=${1:-$STATION_IP}
[ -n "$IP" ] || { echo "укажите IP колонки аргументом или в STATION_IP"; exit 1; }
export PATH=/opt/homebrew/bin:$PATH
adb connect $IP:5555 >/dev/null
adb -s $IP:5555 shell "tail -n 50 -F /data/quasar/daemons_logs/maind.log" | grep --line-buffered -i -E "onPhraseSpotted|recognizedPhrase|outputSpeech|spotter|Spotter|model|Model|cancel|Cancel|validation" | cut -c1-220
