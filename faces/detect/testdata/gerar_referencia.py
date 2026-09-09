"""Gera a referencia de deteccao a partir do cv2.FaceDetectorYN.

A imagem e produzida por formula inteira, reproduzivel bit a bit em Go: o
repositorio nao guarda imagem de rosto, e uma imagem sintetica basta para
conferir a DECODIFICACAO, que e o que esta sob teste.

O array numpy e BGR, que e como o OpenCV carrega imagem. Em Go, onde
image.Image e RGB, os canais 0 e 2 sao trocados na leitura.

Uso:  python gerar_referencia.py caminho/para/yunet.onnx
"""
import sys
import numpy as np
import cv2

H = W = 640
LIMIAR = 0.05
NMS = 0.3
TOPK = 5000

idx = np.arange(H * W * 3, dtype=np.int64)
img = ((idx * 7919) % 251).astype(np.uint8).reshape(H, W, 3)

det = cv2.FaceDetectorYN.create(
    model=sys.argv[1], config="", input_size=(W, H),
    score_threshold=LIMIAR, nms_threshold=NMS, top_k=TOPK,
)
det.setInputSize((W, H))

_, faces = det.detect(img)
faces = np.zeros((0, 15), np.float32) if faces is None else faces.astype(np.float32)

faces.tofile("deteccoes.bin")
print(f"limiar={LIMIAR} nms={NMS}: {len(faces)} deteccoes")
for f in faces[:5]:
    print(f"  score={f[14]:.6f} box=({f[0]:.3f},{f[1]:.3f},{f[2]:.3f},{f[3]:.3f})")
