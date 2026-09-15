#!/usr/bin/env python3
"""Calibra anti-spoof e liveness com a webcam do PC.

Replica o preprocess do facenox/ERA e a metrica de variancia do pacote liveness.
Use para achar Limiar (spoof) e MinVar (movimento) antes do FrotaHub.

Uso (na raiz do ERA):
    python faces/cmd/calibrar/calibrar.py

Requisitos: pip install opencv-python onnxruntime numpy
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

# Raiz do monorepo ERA (faces/cmd/calibrar -> ../../../)
ERA_ROOT = Path(__file__).resolve().parents[3]

# YuNet devolve 15 floats por rosto; pontos 4..13 sao os 5 landmarks.
IDX_OLHO_DIR = (4, 5)
IDX_OLHO_ESQ = (6, 7)
IDX_NARIZ = (8, 9)
IDX_BOCA_DIR = (10, 11)
IDX_BOCA_ESQ = (12, 13)


@dataclass
class Amostra:
    tipo: str  # "real" | "spoof"
    logit_diff: float
    real_logit: float
    spoof_logit: float
    variancia: float
    nota: str = ""


@dataclass
class Estado:
    limiar_prob: float = 0.5
    min_var: float = 0.0005
    janela_frames: int = 15
    amostras: list[Amostra] = field(default_factory=list)
    historico_lm: list[list[tuple[float, float]]] = field(default_factory=list)


def logit_limiar(prob: float) -> float:
    p = max(1e-6, min(1 - 1e-6, prob))
    return math.log(p / (1 - p))


def crop_face(img: np.ndarray, x1: float, y1: float, x2: float, y2: float, expansion: float) -> np.ndarray:
    """Recorte quadrado com expansao — igual facenox/preprocess.py (bbox x1,y1,x2,y2)."""
    h_img, w_img = img.shape[:2]
    w = x2 - x1
    h = y2 - y1
    if w <= 0 or h <= 0:
        raise ValueError("bbox invalida")

    max_dim = max(w, h)
    cx = x1 + w / 2
    cy = y1 + h / 2
    x0 = int(cx - max_dim * expansion / 2)
    y0 = int(cy - max_dim * expansion / 2)
    crop_size = int(max_dim * expansion)

    cx1 = max(0, x0)
    cy1 = max(0, y0)
    cx2 = min(w_img, x0 + crop_size)
    cy2 = min(h_img, y0 + crop_size)

    top = max(0, -y0)
    left = max(0, -x0)
    bottom = max(0, (y0 + crop_size) - h_img)
    right = max(0, (x0 + crop_size) - w_img)

    if cx2 > cx1 and cy2 > cy1:
        patch = img[cy1:cy2, cx1:cx2, :].copy()
    else:
        patch = np.zeros((0, 0, 3), dtype=img.dtype)

    out = cv2.copyMakeBorder(patch, top, bottom, left, right, cv2.BORDER_REFLECT_101)
    if out.shape[0] != crop_size or out.shape[1] != crop_size:
        raise ValueError(f"crop {out.shape} != {crop_size}")
    return out


def preprocess(img: np.ndarray, size: int = 128) -> np.ndarray:
    """Letterbox + [0,1] + CHW — igual facenox."""
    old_h, old_w = img.shape[:2]
    ratio = float(size) / max(old_h, old_w)
    nh = max(1, int(old_h * ratio))
    nw = max(1, int(old_w * ratio))
    interp = cv2.INTER_LANCZOS4 if ratio > 1.0 else cv2.INTER_AREA
    resized = cv2.resize(img, (nw, nh), interpolation=interp)

    pad_t = (size - nh) // 2
    pad_b = size - nh - pad_t
    pad_l = (size - nw) // 2
    pad_r = size - nw - pad_l
    padded = cv2.copyMakeBorder(resized, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_REFLECT_101)

    chw = padded.transpose(2, 0, 1).astype(np.float32) / 255.0
    return chw[np.newaxis, ...]


def landmarks_de_face(face: np.ndarray) -> list[tuple[float, float]]:
    return [
        (float(face[IDX_OLHO_DIR[0]]), float(face[IDX_OLHO_DIR[1]])),
        (float(face[IDX_OLHO_ESQ[0]]), float(face[IDX_OLHO_ESQ[1]])),
        (float(face[IDX_NARIZ[0]]), float(face[IDX_NARIZ[1]])),
        (float(face[IDX_BOCA_DIR[0]]), float(face[IDX_BOCA_DIR[1]])),
        (float(face[IDX_BOCA_ESQ[0]]), float(face[IDX_BOCA_ESQ[1]])),
    ]


def distancia_olhos(lm: list[tuple[float, float]]) -> float:
    dx = lm[1][0] - lm[0][0]
    dy = lm[1][1] - lm[0][1]
    return math.hypot(dx, dy)


def variancia_liveness(seq: list[list[tuple[float, float]]]) -> float:
    """Mesma formula que faces/liveness/liveness.go."""
    if len(seq) < 2:
        return 0.0
    ref = distancia_olhos(seq[0])
    if ref < 1:
        return 0.0

    soma = 0.0
    n = len(seq)
    for p in range(5):
        mx = sum(lm[p][0] for lm in seq) / n
        my = sum(lm[p][1] for lm in seq) / n
        var_x = var_y = 0.0
        for lm in seq:
            dx = (lm[p][0] - mx) / ref
            dy = (lm[p][1] - my) / ref
            var_x += dx * dx
            var_y += dy * dy
        soma += (var_x + var_y) / n
    return soma / 5.0


def inferir_antispoof(sess: ort.InferenceSession, inp_name: str, tensor: np.ndarray) -> tuple[float, float]:
    logits = sess.run(None, {inp_name: tensor})[0][0]
    return float(logits[0]), float(logits[1])


def sugerir_limiar(amostras: list[Amostra]) -> float | None:
    reais = [a.logit_diff for a in amostras if a.tipo == "real"]
    spoofs = [a.logit_diff for a in amostras if a.tipo == "spoof"]
    if not reais or not spoofs:
        return None
    # Ponto medio entre o minimo dos reais e o maximo dos spoofs.
    meio = (min(reais) + max(spoofs)) / 2
    prob = 1.0 / (1.0 + math.exp(-meio))
    return max(0.05, min(0.95, prob))


def sugerir_min_var(amostras: list[Amostra]) -> float | None:
    mov = [a.variancia for a in amostras if a.tipo == "real" and a.variancia > 0]
    parado = [a.variancia for a in amostras if a.tipo == "spoof" and a.variancia > 0]
    if not mov or not parado:
        return None
    return (max(parado) + min(mov)) / 2


def desenhar_hud(
    frame: np.ndarray,
    est: Estado,
    diff: float | None,
    real_l: float | None,
    spoof_l: float | None,
    var: float,
    tem_rosto: bool,
) -> None:
    limiar_logit = logit_limiar(est.limiar_prob)
    linhas = [
        "CALIBRACAO ERA — anti-spoof + liveness",
        f"Limiar spoof (prob): {est.limiar_prob:.2f}  (+/-)   MinVar: {est.min_var:.6f}  ([/])",
        f"Amostras: {len(est.amostras)}  |  [R] real  [F] fake  [S] salvar CSV  [Q] sair",
    ]
    if tem_rosto and diff is not None:
        live_spoof = diff >= limiar_logit
        live_mov = var >= est.min_var
        linhas += [
            f"logit_diff={diff:+.3f}  real={real_l:+.3f}  spoof={spoof_l:+.3f}  -> {'REAL' if live_spoof else 'SPOOF'}",
            f"variancia ({est.janela_frames} frames)={var:.6f}  -> {'MOVIMENTO' if live_mov else 'PARADO'}",
        ]
    else:
        linhas.append("Nenhum rosto — centralize o rosto na camera")

    # Faixa branca no topo para o texto preto ficar legivel em qualquer fundo.
    hud_h = 16 + 24 * len(linhas)
    cv2.rectangle(frame, (0, 0), (frame.shape[1], hud_h), (255, 255, 255), -1)

    y = 22
    for txt in linhas:
        cv2.putText(frame, txt, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
        y += 24


def salvar_csv(amostras: list[Amostra], caminho: Path) -> None:
    with caminho.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["tipo", "logit_diff", "real_logit", "spoof_logit", "variancia", "nota"])
        for a in amostras:
            w.writerow([a.tipo, a.logit_diff, a.real_logit, a.spoof_logit, a.variancia, a.nota])
    print(f"Salvo: {caminho}")


def imprimir_resumo(est: Estado) -> None:
    print("\n--- Resumo da calibracao ---")
    print(f"Amostras: {len(est.amostras)} ({sum(1 for a in est.amostras if a.tipo == 'real')} real, "
          f"{sum(1 for a in est.amostras if a.tipo == 'spoof')} spoof)")

    lim = sugerir_limiar(est.amostras)
    if lim is not None:
        print(f"Limiar sugerido (spoof.Options.Limiar): {lim:.3f}  (logit ~ {logit_limiar(lim):+.3f})")
    else:
        print("Limiar: marque pelo menos 3 reais e 3 fakes com [R] e [F]")

    mv = sugerir_min_var(est.amostras)
    if mv is not None:
        print(f"MinVar sugerido (liveness.Options.MinVar): {mv:.6f}")
    else:
        print("MinVar: marque reais com movimento e fakes parados (foto na tela)")

    print("\nNo Go (FrotaHub / ERA):")
    if lim is not None:
        print(f"  spoof.Options{{ Limiar: {lim:.3f} }}")
    if mv is not None:
        print(f"  liveness.Options{{ MinVar: {mv:.6f} }}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Calibra anti-spoof e liveness com webcam")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--yunet", type=Path, default=ERA_ROOT / "models" / "yunet.onnx")
    ap.add_argument("--antispoof", type=Path, default=ERA_ROOT / "models" / "anti-spoof.onnx")
    ap.add_argument("--saida", type=Path, default=None, help="CSV de saida (padrao: calibracao_YYYYMMDD_HHMM.csv)")
    args = ap.parse_args()

    if not args.yunet.is_file():
        print(f"YuNet nao encontrado: {args.yunet}", file=sys.stderr)
        return 1
    if not args.antispoof.is_file():
        print(f"Anti-spoof nao encontrado: {args.antispoof}", file=sys.stderr)
        return 1

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Nao abri a camera {args.camera}", file=sys.stderr)
        return 1

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    detector = cv2.FaceDetectorYN.create(
        str(args.yunet), "", (640, 480), score_threshold=0.6, nms_threshold=0.3, top_k=5000
    )

    sess = ort.InferenceSession(str(args.antispoof), providers=["CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name

    est = Estado()
    print("Controles: R=rosto real | F=fake (foto/tela) | +/- limiar | [/] MinVar | S=salvar | Q=sair")
    print("Para MinVar: mova a cabeca ao marcar [R]; deixe parado (foto) ao marcar [F].")

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        frame = frame_bgr
        h, w = frame.shape[:2]
        detector.setInputSize((w, h))
        _, faces = detector.detect(frame)

        diff = real_l = spoof_l = None
        var = 0.0
        tem_rosto = faces is not None and len(faces) > 0

        if tem_rosto:
            face = faces[0]
            x, y, bw, bh = face[0], face[1], face[0] + face[2], face[1] + face[3]
            try:
                crop = crop_face(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), x, y, bw, bh, 1.5)
                tensor = preprocess(crop, 128)
                real_l, spoof_l = inferir_antispoof(sess, inp_name, tensor)
                diff = real_l - spoof_l
            except Exception:
                tem_rosto = False

            lm = landmarks_de_face(face)
            est.historico_lm.append(lm)
            if len(est.historico_lm) > est.janela_frames:
                est.historico_lm.pop(0)
            var = variancia_liveness(est.historico_lm)

            if tem_rosto:
                limiar_logit = logit_limiar(est.limiar_prob)
                cor = (0, 200, 0) if diff >= limiar_logit else (0, 0, 220)
                cv2.rectangle(frame, (int(face[0]), int(face[1])), (int(face[0] + face[2]), int(face[1] + face[3])), cor, 2)

        desenhar_hud(frame, est, diff, real_l, spoof_l, var, tem_rosto)
        cv2.imshow("ERA calibrar", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q")):
            break
        if key == ord("+") or key == ord("="):
            est.limiar_prob = min(0.95, est.limiar_prob + 0.05)
        if key in (ord("-"), ord("_")):
            est.limiar_prob = max(0.05, est.limiar_prob - 0.05)
        if key == ord("]"):
            est.min_var *= 1.25
        if key == ord("["):
            est.min_var = max(1e-7, est.min_var / 1.25)

        if key in (ord("r"), ord("R"), ord("f"), ord("F")) and tem_rosto and diff is not None:
            tipo = "real" if key in (ord("r"), ord("R")) else "spoof"
            est.amostras.append(
                Amostra(tipo=tipo, logit_diff=diff, real_logit=real_l, spoof_logit=spoof_l, variancia=var)
            )
            print(f"  + {tipo}: diff={diff:+.3f} var={var:.6f}  (total {len(est.amostras)})")

        if key in (ord("s"), ord("S")):
            out = args.saida or Path(f"calibracao_{datetime.now():%Y%m%d_%H%M}.csv")
            salvar_csv(est.amostras, out)
            imprimir_resumo(est)

    cap.release()
    cv2.destroyAllWindows()

    if est.amostras:
        imprimir_resumo(est)
        resp = input("Salvar CSV antes de sair? [s/N] ").strip().lower()
        if resp == "s":
            out = args.saida or Path(f"calibracao_{datetime.now():%Y%m%d_%H%M}.csv")
            salvar_csv(est.amostras, out)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
