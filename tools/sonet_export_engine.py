#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Экспорт выбранных весов под встроенный TFLite прошивки колонки (int8, только разрешённые опкоды).

Вызывает export_engine() из sonet_train.py; representative dataset для int8 - признаки тестовых
фраз (как в sonet_select.py). После экспорта печатает опкоды с версиями и сверяет их со списком
операций штатных моделей Мини 2 (другие движок не загружает).

  python sonet_export_engine.py --run "$WAKEWORD_ROOT"/sonet/run5 --weights ep28.weights.h5
  -> <run>/model_engine_int8.tflite; дальше tfl_add_metadata.py --fix-io (это делает sonet_pack.sh)

Проверено: повторный экспорт весов run5 (эпоха 28) дал те же опкоды и на тестовом наборе
ведёт себя как установленная модель (см. README, шаг 3).
"""
import argparse, glob, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sonet_train as T  # noqa: E402
from sonet_select import feats  # noqa: E402

# опкоды и версии штатных моделей Мини 2 (прошивка 0.369.1.46)
ENGINE_OPS = {"CONCATENATION": 2, "CONV_2D": 3, "DEPTHWISE_CONV_2D": 3, "DEQUANTIZE": 2, "MUL": 2,
              "QUANTIZE": 1, "SOFTMAX": 2, "STRIDED_SLICE": 2, "SUM": 2, "TRANSPOSE": 2, "HARD_SWISH": 1}


def opcodes(path):
    from tensorflow.lite.python import schema_py_generated as s
    names = {v: k for k, v in vars(s.BuiltinOperator).items() if k.isupper()}
    m = s.Model.GetRootAsModel(open(path, "rb").read(), 0)
    out = []
    for i in range(m.OperatorCodesLength()):
        c = m.OperatorCodes(i)
        code = max(c.BuiltinCode(), c.DeprecatedBuiltinCode())
        out.append((names.get(code, str(code)), c.Version()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--weights", default="best.weights.h5")
    ap.add_argument("--out", default="model_engine_int8.tflite")
    ap.add_argument("--rep", default=os.path.join(T.ROOT, "sonet", "test"), help="wav для representative dataset")
    ap.add_argument("--kernels", default="9,13,17,21,25,29,33"); ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--float", action="store_true", help="без квантования (float32; на колонке не проверялось)")
    a = ap.parse_args()
    wavs = sorted(glob.glob(os.path.join(a.rep, "**", "*.wav"), recursive=True))
    if not a.float and not wavs:
        sys.exit(f"нет wav в {a.rep}: без representative dataset int8 не получится")
    import tensorflow as tf
    kernels = [int(k) for k in a.kernels.split(",")]
    model = T.build(tf, kernels, a.width, len(T.CLASSES))
    model.load_weights(os.path.join(a.run, a.weights))
    rep = None if a.float else [feats(p) for p in wavs]
    path = os.path.join(a.run, a.out)
    T.export_engine(tf, model, kernels, a.width, len(T.CLASSES), path, rep_feats=rep, quantize=not a.float)
    print(path, os.path.getsize(path), "байт; representative dataset:", 0 if a.float else len(wavs), "файлов")
    bad = []
    for name, ver in opcodes(path):
        ok = ENGINE_OPS.get(name) == ver
        print(f"  {name} v{ver}" + ("" if ok else "   <- нет у штатных моделей (или другая версия)"))
        if not ok:
            bad.append(f"{name} v{ver}")
    if bad:
        sys.exit("движок колонки такую модель, скорее всего, не загрузит: " + ", ".join(bad))
    print("опкоды совпадают со штатными моделями")


if __name__ == "__main__":
    main()
