"""Gera entrada e logits de referencia para o anti-spoof MiniFAS."""

import struct
import sys
from pathlib import Path

import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    print("pip install onnxruntime numpy", file=sys.stderr)
    raise

ROOT = Path(__file__).resolve().parents[3]
MODELO = ROOT / "models" / "anti-spoof.onnx"
OUT_DIR = Path(__file__).resolve().parent


def main():
    if not MODELO.exists():
        print(f"modelo nao encontrado: {MODELO}", file=sys.stderr)
        sys.exit(1)

    rng = np.random.default_rng(42)
    x = rng.random((1, 3, 128, 128), dtype=np.float32)

    sess = ort.InferenceSession(str(MODELO), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name
    logits = sess.run(None, {inp: x})[0].astype(np.float32)

    (OUT_DIR / "entrada.bin").write_bytes(x.tobytes())
    (OUT_DIR / "logits.bin").write_bytes(logits.tobytes())
    print(f"entrada {x.shape} logits {logits.shape}")
    print("logits", logits[0])


if __name__ == "__main__":
    main()
