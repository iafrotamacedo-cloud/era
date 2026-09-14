"""Replica getSimilarityTransformMatrix do face_recognize.cpp do OpenCV 4.x."""
import struct
import sys

import cv2
import numpy as np

H = W = 640
idx = np.arange(H * W * 3, dtype=np.int64)
img = ((idx * 7919) % 251).astype(np.uint8).reshape(H, W, 3)

det = cv2.FaceDetectorYN.create(sys.argv[1], "", (W, H), 0.05, 0.3, 5000)
_, faces = det.detect(img)
face = faces[0]

src = np.zeros((5, 2), np.float32)
for r in range(5):
    src[r, 0] = face[4 + 2 * r]
    src[r, 1] = face[4 + 2 * r + 1]

dst = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    np.float32,
)

avg0 = src[:, 0].mean()
avg1 = src[:, 1].mean()
src_mean = [avg0, avg1]
dst_mean = [56.0262, 71.9008]

src_demean = src - src_mean
dst_demean = dst - dst_mean

A00 = (dst_demean[:, 0] * src_demean[:, 0]).sum() / 5
A01 = (dst_demean[:, 0] * src_demean[:, 1]).sum() / 5
A10 = (dst_demean[:, 1] * src_demean[:, 0]).sum() / 5
A11 = (dst_demean[:, 1] * src_demean[:, 1]).sum() / 5
A = np.array([[A00, A01], [A10, A11]], dtype=np.float64)

d = [1.0, 1.0]
detA = A00 * A11 - A01 * A10
if detA < 0:
    d[1] = -1

u, s, vt = np.linalg.svd(A)
smax = max(s)
tol = smax * 2 * np.finfo(np.float32).tiny
rank = int(s[0] > tol) + int(s[1] > tol)

det_u = np.linalg.det(u)
det_vt = np.linalg.det(vt)

if rank == 1:
    if det_u * det_vt > 0:
        T2 = u @ vt
    else:
        dd = d[1]
        d[1] = -1
        D = np.diag(d)
        T2 = u @ D @ vt
        d[1] = dd
else:
    D = np.diag(d)
    T2 = u @ D @ vt

var1 = (src_demean[:, 0] ** 2).sum() / 5
var2 = (src_demean[:, 1] ** 2).sum() / 5
scale = 1.0 / (var1 + var2) * (s[0] * d[0] + s[1] * d[1])
TS = T2 @ src_mean
T = np.zeros((3, 3))
T[:2, :2] = T2
T[0, 2] = dst_mean[0] - scale * TS[0]
T[1, 2] = dst_mean[1] - scale * TS[1]
T[0, :2] *= scale
T[1, :2] *= scale

M = T[:2, :]
print("getSimilarityTransformMatrix:")
print(M)
print(f"A={M[0,0]:.8f} B_neg01={-M[0,1]:.8f} Tx={M[0,2]:.4f} Ty={M[1,2]:.4f}")

aligned = cv2.warpAffine(img, M, (112, 112), flags=cv2.INTER_LINEAR)
aligned.tofile("opencv_align0.bin")
print("warpAffine salvo em opencv_align0.bin")
