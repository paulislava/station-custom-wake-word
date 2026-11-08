#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Дописать в .tflite служебные записи metadata, которых требует движок споттера колонки.

Движок libyandex_evil.so (spotter_conf_read) читает из metadata модели ключ `right_context`
(один байт, в заводской модели = 16). TFLiteConverter такой записи не создаёт, поэтому
после экспорта файл надо дополнить. Файл пересобирается через object API схемы TFLite;
векторы байтов выравниваются на 16, как делает штатный конвертер.

  python tfl_add_metadata.py in.tflite out.tflite [--right-context 16] [--min-runtime 1.15.0]
"""
import argparse
import sys

import flatbuffers
import numpy as np
from tensorflow.lite.python import schema_py_generated as s


class _AlignedBuilder(flatbuffers.Builder):
    """Builder, выравнивающий все векторы на 16 байт (как штатный конвертер TFLite)."""

    def StartVector(self, elem_size, num_elems, alignment):
        return super().StartVector(elem_size, num_elems, max(alignment, 16))


def _aligned_builder():
    return _AlignedBuilder(1 << 20)


def set_metadata(model, name: str, data: bytes):
    """Добавить или заменить запись metadata `name` с содержимым `data`."""
    if model.metadata is None:
        model.metadata = []
    for md in model.metadata:
        n = md.name.decode() if isinstance(md.name, (bytes, bytearray)) else md.name
        if n == name:
            model.buffers[md.buffer].data = np.frombuffer(data, dtype=np.uint8).copy()
            return "replaced"
    buf = s.BufferT()
    buf.data = np.frombuffer(data, dtype=np.uint8).copy()
    model.buffers.append(buf)
    md = s.MetadataT()
    md.name = name
    md.buffer = len(model.buffers) - 1
    model.metadata.append(md)
    return "added"


def _name(x):
    return x.decode() if isinstance(x, (bytes, bytearray)) else x


def fix_io(model):
    """Привести входы/выходы подграфа к виду заводской модели.

    Движок колонки берёт тензоры по ПОЗИЦИИ в subgraph.inputs/outputs и по именам:
    inputs = [frames, state_0, state_1, ...], outputs = [logits, state_0_new, ...].
    Конвертер TF даёт имена serving_default_x:0 / PartitionedCall:N и свой порядок,
    поэтому переименовываем тензоры по подписи и переставляем массивы.
    """
    g = model.subgraphs[0]
    sd = model.signatureDefs[0]
    sig_in = {_name(t.name): int(t.tensorIndex) for t in sd.inputs}
    sig_out = {_name(t.name): int(t.tensorIndex) for t in sd.outputs}
    for nm, ti in {**sig_in, **sig_out}.items():
        g.tensors[ti].name = nm
    n_states = sum(1 for k in sig_in if k.startswith("state_"))
    order_in = ["frames"] + [f"state_{i}" for i in range(n_states)]
    order_out = ["logits"] + [f"state_{i}_new" for i in range(n_states)]
    g.inputs = np.array([sig_in[k] for k in order_in], dtype=np.int32)
    g.outputs = np.array([sig_out[k] for k in order_out], dtype=np.int32)
    return order_in, order_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--right-context", type=int, default=16)
    ap.add_argument("--min-runtime", default=None, help="например 1.15.0 (16 байт с нулями)")
    ap.add_argument("--fix-io", action="store_true", help="имена и порядок входов/выходов как у заводской модели")
    a = ap.parse_args()

    raw = open(a.src, "rb").read()
    model = s.ModelT.InitFromObj(s.Model.GetRootAsModel(raw, 0))
    print("right_context:", set_metadata(model, "right_context", bytes([a.right_context])))
    if a.min_runtime:
        v = a.min_runtime.encode().ljust(16, b"\0")[:16]
        print("min_runtime_version:", set_metadata(model, "min_runtime_version", v))
    if a.fix_io:
        oi, oo = fix_io(model)
        print("io order:", oi, "->", oo)

    b = _aligned_builder()
    b.Finish(model.Pack(b), b"TFL3")
    out = bytes(b.Output())
    open(a.dst, "wb").write(out)

    chk = s.Model.GetRootAsModel(out, 0)
    names = []
    for i in range(chk.MetadataLength()):
        md = chk.Metadata(i)
        bl = chk.Buffers(md.Buffer()).DataLength()
        names.append(f"{md.Name().decode()}[{bl}]")
    print(f"{a.dst}: {len(raw)} -> {len(out)} байт; metadata: {', '.join(names)}")


if __name__ == "__main__":
    sys.exit(main())
