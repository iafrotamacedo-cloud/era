"""Gera vetores de referencia para o pipeline completo YuNet + SFace.

Usa a mesma imagem sintetica que detect/testdata/gerar_referencia.py e o
caminho oficial do OpenCV: FaceDetectorYN + FaceRecognizerSF.alignCrop +
feature. E o que valida OpcoesSFace e o encadeamento detect -> align -> embed.

Uso:
    python gerar_pipeline.py caminho/yunet.onnx caminho/sface.onnx

Grava embeddings.bin: para cada rosto, 128 float32 em ordem de confianca.
"""
import sys

import cv2
import numpy as np

H = W = 640
LIMIAR = 0.05
NMS = 0.3
TOPK = 5000
DIM = 128

idx = np.arange(H * W * 3, dtype=np.int64)
img = ((idx * 7919) % 251).astype(np.uint8).reshape(H, W, 3)

det = cv2.FaceDetectorYN.create(
    model=sys.argv[1],
    config="",
    input_size=(W, H),
    score_threshold=LIMIAR,
    nms_threshold=NMS,
    top_k=TOPK,
)
det.setInputSize((W, H))

rec = cv2.FaceRecognizerSF.create(sys.argv[2], "")

_, faces = det.detect(img)
if faces is None:
    faces = np.zeros((0, 15), np.float32)

vetores = []
for face in faces:
    alinhado = rec.alignCrop(img, face)
    feat = rec.feature(alinhado)
    vetores.append(feat.reshape(-1).astype(np.float32))

if vetores:
    out = np.stack(vetores, axis=0)
else:
    out = np.zeros((0, DIM), np.float32)

out.tofile("embeddings.bin")
print(f"limiar={LIMIAR} nms={NMS}: {len(vetores)} rostos, {DIM} dimensoes cada")
