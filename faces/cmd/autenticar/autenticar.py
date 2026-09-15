#!/usr/bin/env python3
"""Teste real de autenticacao facial na webcam.

Cadastra um vetor SFace + verifica com anti-spoof, liveness e similaridade.
Usa os limiares calibrados em docs/faces-modelos.md (2026-09-14).

Na raiz do ERA:
    python faces/cmd/autenticar/autenticar.py cadastrar
    python faces/cmd/autenticar/autenticar.py verificar

Requisitos: pip install opencv-python onnxruntime numpy
"""

from __future__ import annotations

import argparse
import math
import struct
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

ERA_ROOT = Path(__file__).resolve().parents[3]
TEMPLATE_PADRAO = Path(__file__).resolve().parent / "template_local.bin"

# Calibracao PC 2026-09-14 — docs/faces-modelos.md
LIMIAR_SPOOF_PROB = 0.27
MIN_VAR = 0.012
MIN_FRAMES = 15
COSINE_LIMIAR = 0.38

IDX_OLHO_DIR, IDX_OLHO_ESQ = (4, 5), (6, 7)
IDX_NARIZ = (8, 9)
IDX_BOCA_DIR, IDX_BOCA_ESQ = (10, 11), (12, 13)


def logit_limiar(prob: float) -> float:
    p = max(1e-6, min(1 - 1e-6, prob))
    return math.log(p / (1 - p))


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
        vx = vy = 0.0
        for lm in seq:
            dx = (lm[p][0] - mx) / ref
            dy = (lm[p][1] - my) / ref
            vx += dx * dx
            vy += dy * dy
        soma += (vx + vy) / n
    return soma / 5.0


def crop_face(img: np.ndarray, x1: float, y1: float, x2: float, y2: float, expansion: float) -> np.ndarray:
    h_img, w_img = img.shape[:2]
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        raise ValueError("bbox invalida")
    max_dim = max(w, h)
    cx, cy = x1 + w / 2, y1 + h / 2
    x0 = int(cx - max_dim * expansion / 2)
    y0 = int(cy - max_dim * expansion / 2)
    crop_size = int(max_dim * expansion)
    cx1, cy1 = max(0, x0), max(0, y0)
    cx2, cy2 = min(w_img, x0 + crop_size), min(h_img, y0 + crop_size)
    top, left = max(0, -y0), max(0, -x0)
    bottom, right = max(0, (y0 + crop_size) - h_img), max(0, (x0 + crop_size) - w_img)
    patch = img[cy1:cy2, cx1:cx2, :].copy() if cx2 > cx1 and cy2 > cy1 else np.zeros((0, 0, 3), dtype=img.dtype)
    return cv2.copyMakeBorder(patch, top, bottom, left, right, cv2.BORDER_REFLECT_101)


def preprocess_antispoof(img_rgb: np.ndarray, size: int = 128) -> np.ndarray:
    old_h, old_w = img_rgb.shape[:2]
    ratio = float(size) / max(old_h, old_w)
    nh, nw = max(1, int(old_h * ratio)), max(1, int(old_w * ratio))
    interp = cv2.INTER_LANCZOS4 if ratio > 1.0 else cv2.INTER_AREA
    resized = cv2.resize(img_rgb, (nw, nh), interpolation=interp)
    pad_t = (size - nh) // 2
    pad_b = size - nh - pad_t
    pad_l = (size - nw) // 2
    pad_r = size - nw - pad_l
    padded = cv2.copyMakeBorder(resized, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_REFLECT_101)
    chw = padded.transpose(2, 0, 1).astype(np.float32) / 255.0
    return chw[np.newaxis, ...]


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def desenhar_hud(frame: np.ndarray, linhas: list[str]) -> None:
    hud_h = 16 + 24 * len(linhas)
    cv2.rectangle(frame, (0, 0), (frame.shape[1], hud_h), (255, 255, 255), -1)
    y = 22
    for txt in linhas:
        cv2.putText(frame, txt, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
        y += 24


def salvar_template(caminho: Path, vetor: np.ndarray) -> None:
    caminho.write_bytes(struct.pack(f"<{len(vetor)}f", *vetor.astype(np.float32)))


def carregar_template(caminho: Path) -> np.ndarray:
    raw = caminho.read_bytes()
    n = len(raw) // 4
    return np.array(struct.unpack(f"<{n}f", raw), dtype=np.float32)


class Motor:
    def __init__(self, yunet: Path, sface: Path, antispoof: Path):
        self.det = None
        self.rec = cv2.FaceRecognizerSF.create(str(sface), "")
        self.sess = ort.InferenceSession(str(antispoof), providers=["CPUExecutionProvider"])
        self.inp_antispoof = self.sess.get_inputs()[0].name
        self.yunet_path = yunet
        self.limiar_logit = logit_limiar(LIMIAR_SPOOF_PROB)

    def detectar(self, frame_bgr: np.ndarray) -> np.ndarray | None:
        h, w = frame_bgr.shape[:2]
        if self.det is None:
            self.det = cv2.FaceDetectorYN.create(
                str(self.yunet_path), "", (w, h), score_threshold=0.6, nms_threshold=0.3, top_k=5000
            )
        self.det.setInputSize((w, h))
        _, faces = self.det.detect(frame_bgr)
        if faces is None or len(faces) == 0:
            return None
        return faces[0]

    def antispoof(self, frame_bgr: np.ndarray, face: np.ndarray) -> tuple[float, float, float]:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        x, y, bw, bh = face[0], face[1], face[0] + face[2], face[1] + face[3]
        crop = crop_face(rgb, x, y, bw, bh, 1.5)
        tensor = preprocess_antispoof(crop)
        logits = self.sess.run(None, {self.inp_antispoof: tensor})[0][0]
        real, spoof = float(logits[0]), float(logits[1])
        return real - spoof, real, spoof

    def embed(self, frame_bgr: np.ndarray, face: np.ndarray) -> np.ndarray:
        alinhado = self.rec.alignCrop(frame_bgr, face)
        feat = self.rec.feature(alinhado)
        return feat.reshape(-1).astype(np.float32)


def capturar_sequencia(
    motor: Motor,
    cap: cv2.VideoCapture,
    titulo: str,
    segundos_max: float = 12.0,
) -> tuple[np.ndarray, list[np.ndarray]] | None:
    """Coleta MIN_FRAMES com rosto; retorna vetor do ultimo frame e historico de frames."""
    historico_lm: deque = deque(maxlen=MIN_FRAMES)
    frames_buf: deque = deque(maxlen=MIN_FRAMES)
    faces_buf: deque = deque(maxlen=MIN_FRAMES)
    inicio = time.time()

    while time.time() - inicio < segundos_max:
        ok, frame = cap.read()
        if not ok:
            break

        face = motor.detectar(frame)
        var = 0.0
        status = "Centralize o rosto e mexa a cabeca levemente"
        if face is not None:
            lm = landmarks_de_face(face)
            historico_lm.append(lm)
            frames_buf.append(frame.copy())
            faces_buf.append(face.copy())
            var = variancia_liveness(list(historico_lm))
            status = f"Frames {len(historico_lm)}/{MIN_FRAMES}  var={var:.4f}"

        linhas = [titulo, status, "[ESC] cancelar"]
        if len(historico_lm) >= MIN_FRAMES:
            linhas.append("Segure parado 1s para confirmar...")
        desenhar_hud(frame, linhas)
        if face is not None:
            cv2.rectangle(
                frame,
                (int(face[0]), int(face[1])),
                (int(face[0] + face[2]), int(face[1] + face[3])),
                (0, 180, 0),
                2,
            )
        cv2.imshow("ERA autenticar", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            return None

        if len(historico_lm) >= MIN_FRAMES and var >= MIN_VAR:
            # pequena pausa para estabilizar no ultimo frame
            time.sleep(0.4)
            ok, frame = cap.read()
            if not ok:
                return None
            face = motor.detectar(frame)
            if face is None:
                continue
            diff, _, _ = motor.antispoof(frame, face)
            if diff < motor.limiar_logit:
                desenhar_hud(frame, [titulo, "Falhou anti-spoof — tente de novo", f"logit_diff={diff:+.2f}"])
                cv2.imshow("ERA autenticar", frame)
                cv2.waitKey(1500)
                historico_lm.clear()
                frames_buf.clear()
                faces_buf.clear()
                continue
            vec = motor.embed(frame, face)
            return vec, list(frames_buf)

    return None


def cmd_cadastrar(motor: Motor, cap: cv2.VideoCapture, template: Path) -> int:
    print("Cadastro: olhe para a camera e mova a cabeca quando pedir.")
    resultado = capturar_sequencia(motor, cap, "CADASTRO — liveness + anti-spoof")
    if resultado is None:
        print("Cadastro cancelado ou falhou.")
        return 1
    vetor, _ = resultado
    salvar_template(template, vetor)
    print(f"Template salvo em {template} ({len(vetor)} dims)")
    return 0


def cmd_verificar(motor: Motor, cap: cv2.VideoCapture, template: Path) -> int:
    if not template.is_file():
        print(f"Template nao encontrado: {template}. Rode cadastrar primeiro.", file=sys.stderr)
        return 1

    ref = carregar_template(template)
    print("Verificacao: mesma pose do cadastro; mova a cabeca levemente.")

    while True:
        resultado = capturar_sequencia(motor, cap, "VERIFICAR — autenticacao")
        if resultado is None:
            print("Verificacao cancelada.")
            return 1

        vetor, _ = resultado
        score = cosine(vetor, ref)
        ok = score >= COSINE_LIMIAR

        print(f"Similaridade cosseno: {score:.4f}  (limiar {COSINE_LIMIAR})  -> {'AUTENTICADO' if ok else 'NEGADO'}")

        # tela de resultado
        _, frame = cap.read()
        if frame is None:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cor = (0, 180, 0) if ok else (0, 0, 220)
        msg = "AUTENTICADO" if ok else "ACESSO NEGADO"
        desenhar_hud(frame, ["RESULTADO", f"cosine={score:.4f}", msg, "Qualquer tecla: nova tentativa | ESC: sair"])
        cv2.putText(frame, msg, (120, 280), cv2.FONT_HERSHEY_SIMPLEX, 1.2, cor, 3, cv2.LINE_AA)
        cv2.imshow("ERA autenticar", frame)
        key = cv2.waitKey(0) & 0xFF
        if key == 27:
            return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="Teste de autenticacao facial ERA")
    ap.add_argument("modo", choices=["cadastrar", "verificar"], help="cadastrar template ou verificar")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--yunet", type=Path, default=ERA_ROOT / "models" / "yunet.onnx")
    ap.add_argument("--sface", type=Path, default=ERA_ROOT / "models" / "sface.onnx")
    ap.add_argument("--antispoof", type=Path, default=ERA_ROOT / "models" / "anti-spoof.onnx")
    ap.add_argument("--template", type=Path, default=TEMPLATE_PADRAO)
    args = ap.parse_args()

    for p, nome in [(args.yunet, "YuNet"), (args.sface, "SFace"), (args.antispoof, "anti-spoof")]:
        if not p.is_file():
            print(f"{nome} nao encontrado: {p}", file=sys.stderr)
            return 1

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Camera {args.camera} indisponivel", file=sys.stderr)
        return 1
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    motor = Motor(args.yunet, args.sface, args.antispoof)
    try:
        if args.modo == "cadastrar":
            return cmd_cadastrar(motor, cap, args.template)
        return cmd_verificar(motor, cap, args.template)
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
