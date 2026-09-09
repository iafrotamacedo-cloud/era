"""Gera a entrada de teste e o vetor de referencia do onnxruntime.

A entrada e gravada em disco como float32 cru para que o Go leia exatamente
os mesmos bits. Gerar o mesmo pseudo-aleatorio nas duas linguagens seria uma
fonte de divergencia que nao tem nada a ver com o que esta sob teste.
"""

import struct
import sys

import numpy as np
import onnxruntime as ort

modelo = sys.argv[1]
dir_saida = sys.argv[2]

# Entrada deterministica. Uma imagem de rosto normalizada fica mais ou menos
# nessa faixa; o que importa e ser reproduzivel e exercitar valores dos dois
# sinais.
rng = np.random.default_rng(20260908)
entrada = rng.standard_normal((1, 3, 112, 112)).astype(np.float32)

with open(f"{dir_saida}/entrada.bin", "wb") as f:
    f.write(entrada.tobytes())

sess = ort.InferenceSession(modelo, providers=["CPUExecutionProvider"])

nome_in = sess.get_inputs()[0].name
nome_out = sess.get_outputs()[0].name
print(f"entrada: {nome_in} {sess.get_inputs()[0].shape}")
print(f"saida:   {nome_out} {sess.get_outputs()[0].shape}")

saida = sess.run([nome_out], {nome_in: entrada})[0].astype(np.float32)
vetor = saida.reshape(-1)

with open(f"{dir_saida}/referencia.bin", "wb") as f:
    f.write(vetor.tobytes())

print(f"\nvetor de referencia: {vetor.shape[0]} dimensoes")
print("primeiros 8:", " ".join(f"{v:.6f}" for v in vetor[:8]))
print(f"norma L2: {np.linalg.norm(vetor):.6f}")
print(f"min {vetor.min():.6f}  max {vetor.max():.6f}")
