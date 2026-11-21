#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Экспорт выбранных весов под встроенный TFLite прошивки колонки (int8).

  python wakeword_export_engine.py --word votchi --run "$WAKEWORD_ROOT"/votchi/run1 --weights ep28.weights.h5
"""
import argparse, glob, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sonet_train as T
from wakeword_config import get_config
from sonet_select import feats

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
    ap.add_argument("--word", required=True, choices=["siesta", "votchi", "miku", "smith"])
    ap.add_argument("--run", required=True)
    ap.add_argument("--weights", default="best.weights.h5")
    ap.add_argument("--out", default="model_engine_int8.tflite")
    ap.add_argument("--rep", default=None, help="wav каталог (по умолчанию: $WAKEWORD_ROOT/<word>/test)")
    ap.add_argument("--kernels", default="9,13,17,21,25,29,33")
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--float", action="store_true")
    a = ap.parse_args()

    cfg = get_config(a.word)
    T.CLASSES = cfg["classes"]
    T.KIND2CLS = cfg["kind2cls"]

    rep_dir = a.rep or os.path.join(T.ROOT, a.word, "test")
    wavs = sorted(glob.glob(os.path.join(rep_dir, "**", "*.wav"), recursive=True))
    if not a.float and not wavs:
        sys.exit(f"нет wav в {rep_dir}: без representative dataset int8 не получится")
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
        print(f"  {name} v{ver}" + ("" if ok else "   <- нет у штатных моделей"))
        if not ok:
            bad.append(f"{name} v{ver}")
    if bad:
        sys.exit("движок колонки такую модель не загрузит: " + ", ".join(bad))
    print("опкоды совпадают")


if __name__ == "__main__":
    main()
