# Своё слово активации для Яндекс Станции Мини 2

Обучение и установка своего слова вызова на Станцию Мини 2 с root. В примере слово «Сиеста»:
колонка слушает его вместе со штатным «Алиса». Модель - потоковый TFLite-споттер того же вида,
что заводской, собранный под встроенный движок прошивки.

Проверено на: Яндекс Станция Мини 2 с часами (платформа `yandexmini_2`), прошивка
`0.369.1.46.4778262943.20261013.299` (quasar 3.21), 06.11.2026; в повседневной работе с
06.11.2026. Один экземпляр колонки, замеры срабатываний - один вечер, один говорящий.

Неофициальный проект, с Яндексом не связан. Root, сертификаты, прошивка и модели Яндекса
в репозиторий не входят.

> Установка правит rootfs колонки. Модель, которую движок не принял, оставляет колонку без
> слова вызова до отката. Держать под рукой `deploy/rollback_model_fallback.sh` и ADB.

## Что получится

- Колонка срабатывает на «Сиеста» и на «Алиса»: одна модель на три класса
  (filler / алиса / сиеста) стоит на месте заводской активационной модели.
- После слова вызова запрос идёт обычным путём в облако, отвечает Алиса. Что делать с
  запросами после «Сиеста» (свой обработчик, другой ассистент) - за рамками репозитория.
- Колонка не перекачивает облачную модель поверх своей (заводской каталог + `chattr +i`).
- Облако не отменяет запросы после чужого слова (перепроверка активации выключена).

## Что понадобится

- Мини 2 с root и ADB по сети: `adb connect <IP колонки>:5555`, `adb shell id` -> `uid=0`.
  Root и отключение обновлений прошивки - по https://hackalice.me (разделы start/root и
  start/ssh). Здесь root не описывается.
- Компьютер в одной сети с колонкой: `adb`, `zsh`, `ffmpeg` (нужен `segment_voice.py`),
  Python 3.12. Скрипты написаны под macOS: zsh, `sed -i ''` в `tools/siesta_pack.sh`
  (на Linux - `sed -i`), бэкенд синтеза `say` с голосом Milena и `afconvert`.
- Два окружения Python. Версии, на которых собиралась проверенная модель:
  - обучение и проверка (`venv312`): tensorflow 2.21.0, keras 3.15.1, ai-edge-litert 2.3.0,
    numpy 2.5.4, scipy 1.18.1, flatbuffers 25.12.19, tflite 2.18.0;
  - синтез речи (`venv_tts`): torch 2.15.0, piper-tts 1.8.0, numpy 2.5.4.
- Голоса синтеза в `$WAKEWORD_ROOT/tts/`: Silero `v4_ru.pt` (голоса aidar, baya, kseniya,
  xenia, eugene) и Piper `ru_RU-{irina,denis,dmitri,ruslan}-medium.onnx` с файлами
  `.onnx.json`. Лицензии голосов проверить до того, как распространять данные или модели.
- Две записи чужой речи по 20 минут без слов «Сиеста» и «Алиса» (16 кГц, моно):
  `$WAKEWORD_ROOT/siesta/test_long_neg.wav` и `test_long_neg2.wav`. Скрипта для них в
  репозитории нет, источник записей проверенного прогона не описан.
- Время: 40 эпох обучения, около 30 с на эпоху при 5 процессах сборки батчей.

## Шаги

### 0. Переменные

Команды запускаются из корня репозитория.

```sh
export WAKEWORD_ROOT="$PWD/work"           # данные, голоса, прогоны, бэкапы с колонки
export PYTHON=/путь/к/venv312/bin/python   # окружение обучения для скриптов tools/*.sh
export STATION_IP=<IP колонки>             # или первым аргументом скриптов deploy/
mkdir -p "$WAKEWORD_ROOT"/tts              # сюда - файлы голосов Silero и Piper
```

`python` ниже - из окружения синтеза (шаг 1) или обучения (шаги 2-5).

### 1. Данные

```sh
W="$WAKEWORD_ROOT"/sonet
# ключевые слова отдельными клипами, трудные негативы, бытовые фразы, префиксы
python tools/siesta_data.py --backends say    --per-voice 60 --out $W/raw
python tools/siesta_data.py --backends silero --per-voice 60 --out $W/raw_silero
python tools/siesta_data.py --backends piper  --per-voice 60 --out $W/raw_piper
# слово в начале, середине и конце целой фразы + граница слова по паузам
python tools/siesta_sentences.py --backends silero,piper --out $W/raw_sent
python tools/siesta_sentences.py --backends say --out $W/raw_sent_say
python tools/siesta_align_pause.py $W/raw_sent/manifest.jsonl $W/raw_sent_say/manifest.jsonl
# тестовый набор: целые фразы, не склейка
python tools/siesta_testset.py
```

Манифесты `manifest.jsonl` дописываются, а не перезаписываются: перед повторной генерацией
каталог удалить.

Разметка перед обучением. В списке трудных негативов `siesta_data.py` есть «Алиса»,
«Алиска», «Алис», «алиса стоп»: как фон они спорят с классом «алиса» и ослабляют его.
Для проверенной модели три манифеста шага 1 объединены в
`$WAKEWORD_ROOT/siesta/raw_clean/manifest.jsonl`, записи `neg_word` с текстом на «алис…»
убраны, клипы с текстом «Алиса» переведены в `pos_alisa`, трудные негативы (клипы, на
которых предыдущая модель давала >= 0,35) добавлены повторно. Скрипта этой чистки в
репозитории нет.

Проверка данных штатной моделью (необязательно). Файл модели снять со своей колонки:
`/data/quasar/data/spotter_model/<uuid>/model.tflite`. Модель Яндекса не публиковать.

```sh
python tools/siesta_eval.py stock.tflite --cls 1 --thr 0.691 \
    --pos $W/raw/pos_alisa --neg $W/raw/pos_siesta
```

Синтез годится, если штатная модель узнаёт почти все «Алиса» и не узнаёт «Сиеста».

### 2. Обучение

```sh
W="$WAKEWORD_ROOT"/sonet
$PYTHON -I tools/siesta_train.py --raw $W/raw_clean \
    --extra-raw $W/raw_sent/manifest.jsonl,$W/raw_sent_say/manifest.jsonl \
    --epochs 40 --steps 150 --batch 48 --holdout-voice piper-denis --out $W/run5
$PYTHON -I tools/siesta_select.py --run $W/run5
```

Точная командная строка проверенного прогона не сохранилась; выше - эквивалент по описанию
(с нуля, 40 эпох по 150 шагов, отложенный голос piper-denis). Трудные негативы можно
добавить в `--extra-raw` отдельным манифестом.

`siesta_select.py` прогоняет веса каждой эпохи по тестовому набору и 20-минутному потоку и
печатает таблицу: recall «сиеста» и «алиса», ложные на файлах, срабатывания на потоке.
Лучшая эпоха по формуле score сохраняется как `model.tflite` - это float-модель для проверок
на компьютере, движок колонки её не загрузит (шаг 3). Для домашней колонки разумно выбирать
эпоху с меньшим числом ложных, даже ценой recall.

Дообучение на своих записях (не проверялось):
`zsh tools/siesta_finetune_real.sh "<записи Сиеста>" "<записи Алиса>" "<речь без слов>" run6`
(пути к записям абсолютные; скрипт берёт `siesta/raw_all` и веса `siesta/run1`).

### 3. Экспорт под движок прошивки

```sh
$PYTHON -I tools/siesta_export_engine.py --run "$WAKEWORD_ROOT"/siesta/run5 --weights ep28.weights.h5
```

Результат - `model_engine_int8.tflite`. Встроенный TFLite прошивки знает только операции
штатных моделей. Обёртка печатает опкоды модели и сверяет их со списком:

| Опкод | Версия |
|---|---|
| CONCATENATION | 2 |
| CONV_2D | 3 |
| DEPTHWISE_CONV_2D | 3 |
| DEQUANTIZE | 2 |
| MUL | 2 |
| QUANTIZE | 1 |
| SOFTMAX | 2 |
| STRIDED_SLICE | 2 |
| SUM | 2 |
| TRANSPOSE | 2 |
| HARD_SWISH | 1 |

RESHAPE, ADD, FULLY_CONNECTED, SLICE нет. Проверенная модель использует CONCATENATION v2,
CONV_2D v3, DEPTHWISE_CONV_2D v3, DEQUANTIZE v2, QUANTIZE v1, STRIDED_SLICE v2. Есть ли в
движке другие операции, не проверялось.

Обёртка `siesta_export_engine.py` проверена повторным экспортом тех же весов (run5, эпоха 28):
опкоды те же. Побайтно модель не совпадает с установленной (int8 зависит от набора
калибровочных записей), но на тестовом наборе ведёт себя так же: «Сиеста» 84% против 83%
(90 записей, порог 0.5), «Алиса» 71% и 71% (45 записей), ложных на 180 посторонних фразах
11 и 11 для «Сиеста», 2 и 2 для «Алиса»; расхождение пиков не больше 0,045.

### 4. Метаданные и каталог модели

Заводской `flags.txt` TFLite-модели «Алиса» снимается с вашей колонки до установки (в
репозитории его нет: это файл прошивки). Каталог модели скачан облаком в
`/data/quasar/data/spotter_model/`, нужный узнаётся по строке `version`:

```sh
adb -s <IP>:5555 shell 'grep -l minispeakers-alisa /data/quasar/data/spotter_model/*/flags.txt'
adb -s <IP>:5555 pull /data/quasar/data/spotter_model/<каталог>/flags.txt deploy/model-dir/flags.txt
zsh tools/siesta_pack.sh run5
```

`siesta_pack.sh` вызывает
`tfl_add_metadata.py <in> <out> --right-context 16 --min-runtime 1.15.0 --fix-io`, кладёт
модель в `deploy/model-dir/model.tflite` вместе с `words.txt`, правит `flags.txt` и проверяет
сигнатуру. Что делает `tfl_add_metadata.py`:

- дописывает в metadata модели запись `right_context` (1 байт, 16 как у штатной модели) и
  `min_runtime_version` 1.15.0. Без `right_context` движок пишет
  `spotter_conf_read(): No right_context in metadata`;
- `--fix-io`: переименовывает тензоры в `frames, state_0..state_6` и
  `logits, state_0_new..state_6_new` и ставит их в этом порядке в `subgraph.inputs/outputs`.
  Движок берёт тензоры по позиции и по именам, а конвертер TF даёт имена
  `serving_default_x:0` / `PartitionedCall:N` и свой порядок. Без этого:
  `spotter_conf_read(): Input and output shapes of state 0 differ`.

Пороги. `siesta_pack.sh` пишет значения живого теста: `thresholds 0.4:0.5`,
`subthresholds 0.25:0.35`, `use-tts-blocker false`. С `0.5:0.691` было примерно одно
срабатывание на 10 попыток. Порядок значений - как в `commands 1:2` и
`string-representations алиса:сиеста`: первое для «Алиса», второе для «Сиеста». У штатной
модели порог 0.691.

### 5. Проверка на компьютере

```sh
M=deploy/model-dir/model.tflite; S="$WAKEWORD_ROOT"/sonet
$PYTHON -I tools/tfl_inspect.py $M
$PYTHON -I tools/siesta_eval.py $M --cls 2 --thr 0.69 --pos $S/test/pos_siesta \
    --neg $S/test/neg_sent $S/test/pos_alisa
$PYTHON -I tools/siesta_eval.py $M --cls 1 --thr 0.5 --pos $S/test/pos_alisa \
    --neg $S/test/neg_sent $S/test/pos_siesta
$PYTHON -I tools/siesta_eval.py $M --cls 2 --thr 0.69 \
    --events $S/test_long_neg.wav $S/test_long_neg2.wav
```

`tfl_inspect.py` должен показать входы `frames` [1,1,6,80] и семь `state_i`, выходы `logits`
[1,1,6,3] и `state_i_new`, metadata `right_context` и `min_runtime_version`. Ориентир для
проверенной модели - раздел «Результаты». Правило решения в `siesta_eval.py` (среднее по 11
кадрам, пик выше порога) проще, чем у движка (`metric geometric`, `w-max 20`, `w-smooth 11`,
подпороги), поэтому пороги компьютера и `flags.txt` напрямую не равны (вывод).

### 6. Установка на колонку

```sh
zsh deploy/install_model_fallback.sh "$STATION_IP"
zsh deploy/watch_station.sh "$STATION_IP"       # в соседнем окне, после перезагрузки
```

Что делает `install_model_fallback.sh`:

1. Проверяет root по ADB, снимает бэкап `spotter.json` и заводского каталога
   `/system/vendor/quasar/spotter_models/activation/alisa` в
   `$WAKEWORD_ROOT/deploy/backup-fallback-<время>/`.
2. Кладёт `model.tflite`, `flags.txt`, `words.txt` из `deploy/model-dir` в
   `/data/opt/sonet-model`.
3. Перемонтирует `/` на запись, сохраняет заводские `flags.txt` и `words.txt` в
   `/data/opt/sonet-backup-sys`, копирует свои три файла в заводской каталог, возвращает `ro`.
4. Убирает из `/data/quasar/data/spotter.json` запись `activation/alisa`, удаляет скачанные
   каталоги моделей (с `model.tflite` и `tts-timings-info`), ставит `chattr +i` на
   `spotter.json` и `spotter_model`.
5. Перезагружает колонку (прозвучит приветствие).

Что смотреть в журнале `/data/quasar/daemons_logs/maind.log` (его показывает
`watch_station.sh`):

| Строка в журнале | Значение |
|---|---|
| `createComponent created ACTIVATION` без `failed` | модель загружена |
| `onPhraseSpotted: сиеста`, `hits [0,1]` | сработал «Сиеста» |
| `onPhraseSpotted: алиса`, `hits [1,0]` | сработала «Алиса» |
| `mini_unzip: Can't create .../flags.txt` | защита от перекачки работает |
| `Zip.CreateOutFileFailed` | то же |
| `Didn't find op for builtin opcode 'RESHAPE'` | модель не из шага 3 |
| `No right_context in metadata` | пропущен шаг 4 |
| `Input and output shapes of state 0 differ` | нет `--fix-io` |
| `onRecognitionError: ... SpottingRejectedError` | облако отменило запрос: шаг 7 |

Повторный запуск сам снимает `chattr -i` перед записью. При ошибке загрузки - откат (ниже).

### 7. Отключение облачной перепроверки активации

```sh
zsh deploy/disable_online_validation.sh "$STATION_IP" show
zsh deploy/disable_online_validation.sh "$STATION_IP" apply     # правка + перезагрузка
```

В `/system/vendor/quasar/quasar.cfg`, секция `voiceDialogSettings`:

```
было:  "onlineSpotterValidationDisabledTypes" : [ "noise-detection", "vad" ]
стало: "onlineSpotterValidationDisabledTypes" : [ "noise-detection", "vad", "activation" ]
```

Зачем. После срабатывания колонка отправляет в облако звук вокруг слова, сервер
перепроверяет, было ли там слово активации, и при отказе отменяет запрос. Серверная модель
знает только «Алиса». В журнале это `onRecognitionError: SpottingRejectedError` /
`ErrorSpottingRejected` примерно через 0,9 с после `LISTENING`. В первом живом тесте отказ
пришёл даже на «Алиса», пойманную своей моделью. Облачный `system_config` ключ
`onlineSpotterValidationDisabledTypes` не передаёт (проверено 06.11.2026), поэтому локальное
значение остаётся в силе. Будет ли так после изменений на стороне облака - не проверялось.

Оригинал `quasar.cfg` сохраняется в `$WAKEWORD_ROOT/deploy/quasar.cfg.orig` (файл прошивки,
не публиковать).

## Результаты

Живой тест на колонке, 06.11.2026, один говорящий, пороги `0.4:0.5`, перепроверка выключена:

| Замер | Результат |
|---|---|
| «Сиеста» | 5 из 5 срабатываний, отказов облака 0 |
| «Алиса» через свою модель | около 1 срабатывания на 2-3 попытки |
| С порогами `0.5:0.691` (до снижения) | около 1 срабатывания на 10 попыток |
| Ложные срабатывания на колонке | не проверялось |
| Разные люди, дети, шумная комната | не проверялось |
| Работа днями, после обновления облачного конфига | не проверялось |

Своя модель, проверка на компьютере (`siesta_eval.py`, `siesta_select.py`): тестовые фразы
(90 с «Сиеста», 45 с «Алиса», 180 без ключевых слов) и 20-минутные потоки чужой речи. Порог
0,69 (по умолчанию в инструментах; для строки int8 в журнале явно не записан), кроме
отмеченного.

| Модель | «Сиеста» | «Алиса» | Потоки, за 20 мин |
|---|---|---|---|
| run5 эп. 28, int8 (установлена) | 0,756; 6 ложных файлов | 0,71 при 0,5; 2 ложных | 2 и 4 |
| run5 эп. 28, Keras (отбор) | 0,76 (медиана 0,83); 7 ложных | 0,49; 0 ложных | 2 |
| run3 эп. 16, Keras (отбор) | 0,73 (медиана 0,83); 6 ложных | 0,40; 1 ложный | 1 |

У run3 и у int8-модели run5 ложные файлы «Сиеста» - фраза со словом «сиестаы». Прочие замеры:

| Замер | Результат |
|---|---|
| int8 против Keras, вероятности (run3) | расхождение не больше 0,06 |
| до и после `tfl_add_metadata.py --fix-io` | расхождение 0,0 |
| float-экспорт `export_streaming` против Keras | до 3e-6 |
| run3 в имитации комнаты: чисто | 0,73 |
| то же, RT60 0,3 с, SNR 15 дБ | 0,63 |
| то же, RT60 0,6 с, SNR 5 дБ | 0,51; ложные 5-6 из 180, не растут |
| штатная модель на синтезе, «Алиса» | 299 из 300 >= 0,691 (медиана 0,938) |
| штатная модель на синтезе, «Сиеста» | 0 из 362 (максимум 0,112) |
| штатная модель на 20-минутном потоке | 13 срабатываний (39 в час) |

Числа на компьютере получены на синтезированной речи. Тестовые фразы озвучены теми же
движками синтеза, что и обучающие данные (голос piper-denis из обучения исключён), поэтому
на живых голосах числа будут другими. Дообучение на реальных записях не проверялось.

## Ограничения

- Своё слово только подменой модели: режим своего слова (CKWS) в этой сборке движка выключен
  при компиляции (вывод по строке `Calling CreateCustomActivation() without correct
  compilation flag`).
- «Алиса» через свою модель срабатывает примерно раз на 2-3 попытки; штатная модель в тех же
  условиях отдельно не замерялась. В `flags.txt` своей модели
  `counteractivation-features-enabled false` и `use-tts-blocker false` (в каталоге нет
  `tts-timings-info`; у заводской модели `use-tts-blocker true`). Как это влияет на ложные
  срабатывания, не проверялось. По строкам движка `use-tts-blocker` с `tts-timings-info` -
  штатная защита от реакции на собственный голос, когда Алиса сама говорит «Алиса».
  Возможно, самоперебивание ниже - следствие её выключения; включить её для своего слова не
  пробовали.
- Самоперебивание: если ответ колонки содержит слово вызова (например, начинается с
  «Сиеста»), споттер перебивания ловит его из собственного динамика и обрывает ответ.
  В проверенной связке это решено тем, что ответы слово вызова не произносят.
- «Сиеста» попадает в текст распознавания только хвостом: облако слышит «Нет …»; «Алиса»
  попадает целиком. Хвост «Нет» есть не всегда (в одном из двух проверенных запросов его не
  было). Точку начала распознавания задаёт облако (`spotterSubscribePointShiftMs` -80),
  локально её не переопределить. Какое слово сработало, видно по строке `onPhraseSpotted`
  в журнале колонки.
- Перепроверка облаком выключена для всего типа `activation`, то есть и для «Алиса».
- Модель лежит в rootfs: обновление прошивки её перезапишет (не проверялось).
- Подходы через `spotter.json` не работают: правка `path` и подмена файлов в скачанном
  каталоге отменяются, движок сверяет кастомную модель и перекачивает штатную с CDN в новый
  каталог.
- Данные - только синтез речи; реальные голоса в обучении не участвовали.
- Скрипты рассчитаны на zsh и macOS; на других системах не проверялись.

## Откат

```sh
zsh deploy/rollback_model_fallback.sh "$STATION_IP"            # последний бэкап из $WAKEWORD_ROOT
zsh deploy/rollback_model_fallback.sh "$STATION_IP" <каталог>  # конкретный бэкап
zsh deploy/disable_online_validation.sh "$STATION_IP" restore   # вернуть quasar.cfg
```

`rollback_model_fallback.sh` снимает `chattr +i`, возвращает `spotter.json` из бэкапа и
заводские `flags.txt`/`words.txt` из `/data/opt/sonet-backup-sys`, удаляет свой
`model.tflite` из заводского каталога и перезагружает колонку. Каталоги
`/data/opt/sonet-model` и `/data/opt/sonet-backup-sys` остаются, удаляются вручную. Что после
отката колонка снова скачивает и запускает штатную модель, отдельным тестом не проверялось:
после отката сказать «Алиса» и убедиться, что колонка отвечает.

## Как это работает

Штатная модель. Активационная модель Мини 2 на этой прошивке - обычный TFLite
(`/data/quasar/data/spotter_model/<uuid>/`, скачивается с CDN): int8, только встроенные
операции, сигнатура `serving_default`. Вход `frames` float32 [1,1,6,80]: 6 кадров по 10 мс,
80 признаков = 40 log-mel омни-канала + 40 log-mel канала BNR. Восемь входов состояния
`state_i` (потоковая модель с явной передачей состояния), выход `logits` [1,1,6,2]
(filler / «алиса»). Модель запускается на компьютере стандартным LiteRT (`kws_probe.py`):
16 кГц, окно 25 мс, шаг 10 мс, preemphasis 0.97, окно povey, FFT 512, 40 mel-полос
20 Гц - 8 кГц, логарифм, без CMN, оба канала - один сигнал. Две детали: состояния int8
инициализируются zero point каждого тензора, а не нулями; с CMN модель не работает.

Своя модель (`siesta_train.py`). Потоковая CNN: проекция 80 -> 64, 7 блоков depthwise-свёрток
по времени (ядра 9-33) с остаточными связями, 50 307 параметров, 3 класса. Метка класса стоит
на конце слова (от -120 до +40 мс). Аугментации: темп, громкость, реверберация, наклон
спектра, НЧ-срез, шум, гул, бабл из негативной речи, VTLP (растяжение mel-шкалы 0,88-1,12).
Второй канал признаков - копия первого с небольшим шумом. Данные: три движка синтеза,
10 голосов, ключевые слова отдельными клипами и внутри целых фраз (граница слова по паузам,
`siesta_align_pause.py`), трудные негативы (созвучные слова: «сиестаы», «совет», «монета»,
«Анфиса» и т. п.).

Два экспорта. `export_streaming` - float-модель с той же сигнатурой, что у штатной, для
проверок на компьютере. `export_engine` - под движок колонки: граф 4-D NHWC [1,1,T,C] без
RESHAPE/ADD/FULLY_CONNECTED; сложение остаточной связи - CONCATENATION по каналам и CONV_2D
с постоянным ядром [I;I]; состояние хранится в обратном порядке кадров (срез с шагом -1
конвертер оставляет STRIDED_SLICE, срез с шагом 1 превращает в SLICE, которого в прошивке
нет); int8 с representative dataset даёт те же версии операций, что у штатной модели.
Состояния своей модели: 7 тензоров [1,1,k-1,64].

Загрузка движком. Движок споттера (`libyandex_evil.so`, `spotter_conf_read`) читает
`flags.txt`, `words.txt` и `model.tflite` из каталога модели и требует в metadata модели
`right_context`. Каталог, указанный в `spotter.json`, движок сверяет и при несовпадении
перекачивает. Держится только вариант с заводским каталогом в rootfs и запретом скачивания
(`chattr +i` на `spotter.json` и `spotter_model`).

`flags.txt` (заводской с колонки, правит `siesta_pack.sh`), ключи, которые меняются под своё слово:

| Ключ | Значение | Зачем |
|---|---|---|
| `commands` | `1:2` | какие выходы модели - слова (индексы из `words.txt`) |
| `string-representations` | `алиса:сиеста` | имена слов в журнале (`onPhraseSpotted`) |
| `thresholds` | `0.4:0.5` | пороги «Алиса»:«Сиеста» |
| `subthresholds` | `0.25:0.35` | подпороги в том же порядке |
| `use-tts-blocker` | `false` | у своей модели нет `tts-timings-info` (см. «Ограничения») |
| `version` | `ru-RU-custom-siesta-alisa-...` | метка модели, видна в журнале |

Назначение ключей - по именам флагов и наблюдениям, не по документации. Решение сглаживают
`metric geometric`, `w-max 20`, `w-smooth 11`; модель задают `spotter-npu-model-path
model.tflite` и `spotter-tflite-resolver-tag SPOTTER_ACTIVATION`. Ключи признаков и
частотного фильтра взяты из заводского `flags.txt` этой прошивки. `words.txt`: `filler 0`,
`алиса 1`, `сиеста 2`.

Облачная перепроверка. После срабатывания колонка отправляет звук вокруг слова на сервер
(`SpotterValidation`); пока ждёт ответа, запрос висит, при отказе отменяется. Список типов
споттеров, для которых перепроверка не нужна, - локальный
`onlineSpotterValidationDisabledTypes` в `quasar.cfg`. Типы споттеров по строкам `maind`:
`activation`, `command`, `counteractivation`, `custom-activation`, `interruption`,
`intonation`, `noise-detection`, `vad`.

## Состав

| Файл | Назначение |
|---|---|
| `tools/siesta_data.py` | синтез ключевых слов, трудных негативов, фраз |
| `tools/siesta_sentences.py` | фразы с ключевым словом внутри и те же фразы без него |
| `tools/siesta_align_pause.py` | граница слова во фразах по паузам (`kw_start`, `kw_end`) |
| `tools/siesta_align.py` | то же через whisper (на CPU медленно) |
| `tools/siesta_testset.py` | тестовый набор: целые фразы |
| `tools/segment_voice.py` | нарезка своих голосовых записей по паузам |
| `tools/siesta_train.py` | обучение, `export_streaming`, `export_engine` |
| `tools/siesta_select.py` | отбор эпохи по тестам, float-экспорт |
| `tools/siesta_export.py` | float-экспорт из любых сохранённых весов |
| `tools/siesta_export_engine.py` | int8-экспорт под движок и сверка опкодов |
| `tools/siesta_eval.py` | recall, ложные, срабатывания на длинном потоке |
| `tools/tfl_add_metadata.py` | `right_context`, `min_runtime_version`, `--fix-io` |
| `tools/tfl_inspect.py` | разбор `.tflite` без TensorFlow |
| `tools/kws_probe.py` | прогон wav через штатную модель (2 класса) |
| `tools/siesta_run_all.sh` | первый прогон на склейках (run1) |
| `tools/siesta_run2.sh` | дообучение с фразами (run2) и оценка |
| `tools/siesta_finetune_real.sh` | дообучение на своих записях (не проверялось) |
| `tools/siesta_pack.sh` | сборка `deploy/model-dir` |
| `deploy/install_model_fallback.sh` | установка в заводской каталог и защита от перекачки |
| `deploy/rollback_model_fallback.sh` | откат установки |
| `deploy/disable_online_validation.sh` | `show` / `apply` / `restore` для `quasar.cfg` |
| `deploy/watch_station.sh` | живой журнал споттера |
| `deploy/model-dir/` | `words.txt`; `flags.txt` снимается с колонки (шаг 4), `model.tflite` кладёт `siesta_pack.sh` |

Переменные окружения: `WAKEWORD_ROOT` (рабочий каталог, по умолчанию `./work`), `PYTHON`
(интерпретатор для `tools/*.sh`), `STATION_IP` (IP колонки), `MODEL_DIR` (каталог модели),
`COPY_HOST` и `COPY_DIR` (необязательная копия каталога модели на другую машину из
`siesta_pack.sh`).

## Лицензия

Код - Apache License 2.0 (`LICENSE`, `NOTICE`). Модели, прошивка и файлы Яндекса в
репозиторий не входят; рабочий каталог `work/` с данными и бэкапами с колонки в git не
попадает (`.gitignore`).
