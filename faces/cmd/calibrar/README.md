# Calibrar anti-spoof e liveness (webcam)

Ferramenta interativa para achar os limiares antes de integrar ao FrotaHub.

## Requisitos

- Webcam no PC
- Python 3.8+
- YuNet em `models/yunet.onnx` (OpenCV Zoo)
- Anti-spoof em `models/anti-spoof.onnx` (já versionado no repo)

```bash
pip install opencv-python onnxruntime numpy
```

## Rodar

Na **raiz do ERA**:

```bash
python faces/cmd/calibrar/calibrar.py
```

Outra câmera: `--camera 1`

## Roteiro de calibração (~15 min)

1. **Rosto real** — olhe para a câmera, boa luz, mova levemente a cabeça. Pressione **R** ~10 vezes.
2. **Foto na tela** — abra sua foto no celular/monitor. Pressione **F** ~10 vezes (parado).
3. **Impressão** (se tiver) — mesma coisa com **F**.
4. Ajuste **+/-** o limiar na tela até verde/vermelho bater com o que você sabe ser real/fake.
5. Pressione **S** para salvar CSV e ver sugestões no terminal.

### Teclas

| Tecla | Ação |
|---|---|
| **R** | Marca amostra como rosto **real** |
| **F** | Marca amostra como **fake** (foto/tela) |
| **+** / **-** | Aumenta / diminui limiar de spoof (probabilidade) |
| **[** / **]** | Diminui / aumenta `MinVar` de movimento |
| **S** | Salva CSV + imprime limiares sugeridos |
| **Q** | Sai |

## O que anotar

O script sugere valores para colar no Go:

```go
faces.Open(faces.Config{
    AntiSpoof: &faces.AntiSpoofConfig{
        Options:  spoof.Options{ Limiar: 0.50 },      // ajustar
        Liveness: liveness.Options{ MinVar: 0.0005 }, // ajustar
    },
})
```

Registre os valores finais em `docs/faces-modelos.md` ou no diário quando fechar a calibração.
