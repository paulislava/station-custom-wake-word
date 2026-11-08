#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Экспорт tflite из сохранённых весов (в любой момент обучения).
  python sonet_export.py --run "$WAKEWORD_ROOT"/sonet/run1 [--weights best.weights.h5] [--out model_interim.tflite]
"""
import argparse, os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import sonet_train as T

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True); ap.add_argument("--weights", default="best.weights.h5"); ap.add_argument("--out", default="model_interim.tflite")
ap.add_argument("--kernels", default="9,13,17,21,25,29,33"); ap.add_argument("--width", type=int, default=64)
a = ap.parse_args()
import tensorflow as tf
kernels = [int(k) for k in a.kernels.split(",")]
model = T.build(tf, kernels, a.width, len(T.CLASSES))
model.load_weights(os.path.join(a.run, a.weights))
path = os.path.join(a.run, a.out)
m = T.export_streaming(tf, model, kernels, a.width, len(T.CLASSES), path)
X = np.random.default_rng(0).normal(0, 3, (240, 80)).astype(np.float32)
print(path, os.path.getsize(path), "байт; streaming-vs-offline diff", T.verify(tf, model, m, kernels, a.width, X))
