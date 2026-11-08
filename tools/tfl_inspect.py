#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Разбор .tflite без TensorFlow: входы/выходы, операторы, состояние стриминга, метаданные, сигнатуры."""
import sys, collections
import tflite  # pure-python flatbuffer classes (pip install tflite)

TYPES = {getattr(tflite.TensorType, n): n for n in dir(tflite.TensorType) if n.isupper()}
OPS = {getattr(tflite.BuiltinOperator, n): n for n in dir(tflite.BuiltinOperator) if n.isupper()}


def shape(t):
    s = [t.Shape(i) for i in range(t.ShapeLength())]
    if t.ShapeSignatureLength():
        s = [t.ShapeSignature(i) for i in range(t.ShapeSignatureLength())]
    return s


def main(path):
    buf = open(path, "rb").read()
    m = tflite.Model.GetRootAsModel(buf, 0)
    print(f"# {path}  ({len(buf)} байт)  version={m.Version()}  desc={m.Description()}")
    print(f"subgraphs={m.SubgraphsLength()} buffers={m.BuffersLength()} metadata={m.MetadataLength()} signatures={m.SignatureDefsLength()}")
    for i in range(m.MetadataLength()):
        md = m.Metadata(i)
        b = m.Buffers(md.Buffer())
        print(f"  metadata[{i}] {md.Name()} buffer_len={b.DataLength()}")
        if b.DataLength() and b.DataLength() < 4000:
            print("    ", bytes(b.DataAsNumpy()).decode("utf-8", "replace")[:1500])
    for i in range(m.SignatureDefsLength()):
        sd = m.SignatureDefs(i)
        ins = [sd.Inputs(j).Name().decode() for j in range(sd.InputsLength())]
        outs = [sd.Outputs(j).Name().decode() for j in range(sd.OutputsLength())]
        print(f"  signature[{i}] {sd.SignatureKey()} subgraph={sd.SubgraphIndex()} in={ins} out={outs}")
    codes = [m.OperatorCodes(i) for i in range(m.OperatorCodesLength())]

    def opname(c):
        bc = c.BuiltinCode() if hasattr(c, "BuiltinCode") else c.DeprecatedBuiltinCode()
        if bc == 0 and c.DeprecatedBuiltinCode():
            bc = c.DeprecatedBuiltinCode()
        n = OPS.get(bc, str(bc))
        if n == "CUSTOM":
            n = "CUSTOM:" + (c.CustomCode().decode() if c.CustomCode() else "?")
        return n

    for si in range(m.SubgraphsLength()):
        g = m.Subgraphs(si)
        print(f"\n## subgraph {si} name={g.Name()} tensors={g.TensorsLength()} ops={g.OperatorsLength()}")
        for label, idxs in (("inputs", [g.Inputs(i) for i in range(g.InputsLength())]),
                            ("outputs", [g.Outputs(i) for i in range(g.OutputsLength())])):
            for ti in idxs:
                t = g.Tensors(ti)
                q = t.Quantization()
                qs = ""
                if q and q.ScaleLength():
                    qs = f" q(scale={q.Scale(0):.6g}, zp={q.ZeroPoint(0) if q.ZeroPointLength() else '-'})"
                print(f"  {label}: #{ti} {t.Name().decode()} {TYPES.get(t.Type(), t.Type())} {shape(t)}{qs}")
        cnt = collections.Counter()
        var = []
        for oi in range(g.OperatorsLength()):
            op = g.Operators(oi)
            cnt[opname(codes[op.OpcodeIndex()])] += 1
        for ti in range(g.TensorsLength()):
            t = g.Tensors(ti)
            if t.IsVariable():
                var.append((ti, t.Name().decode(), TYPES.get(t.Type(), t.Type()), shape(t)))
        print("  ops:", ", ".join(f"{k}×{v}" for k, v in cnt.most_common()))
        print(f"  variable (state) tensors: {len(var)}")
        for v in var[:12]:
            print("    ", v)
        # параметры: суммарный размер весов
        wbytes = 0
        for ti in range(g.TensorsLength()):
            t = g.Tensors(ti)
            b = m.Buffers(t.Buffer())
            wbytes += b.DataLength()
        print(f"  weights bytes: {wbytes}")
        # первые и последние операторы с именами выходных тензоров
        seq = []
        for oi in range(g.OperatorsLength()):
            op = g.Operators(oi)
            outs = [g.Tensors(op.Outputs(k)).Name().decode() for k in range(op.OutputsLength())]
            seq.append(f"{opname(codes[op.OpcodeIndex()])}->{outs[0] if outs else '?'}")
        print("  first ops:", " | ".join(seq[:8]))
        print("  last ops :", " | ".join(seq[-6:]))


if __name__ == "__main__":
    for p in sys.argv[1:]:
        main(p)
        print()
