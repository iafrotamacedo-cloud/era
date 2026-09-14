# Modelos do motor `faces`

A ERA não embute pesos por padrão — veja `docs/faces-modelos.md`. Exceção:

## `anti-spoof.onnx`

| Campo | Valor |
|---|---|
| **Arquivo** | MiniFAS (`best_model.onnx` do facenox) |
| **Tamanho** | ~1,9 MB |
| **Licença do código** | [Apache 2.0](https://github.com/facenox/face-antispoof-onnx/blob/main/LICENSE) |
| **Origem** | [facenox/face-antispoof-onnx](https://github.com/facenox/face-antispoof-onnx) release v1.0.0 |
| **Treino** | CelebA-Spoof (ver ressalva abaixo) |

### Atribuição

Pesos derivados do projeto [face-antispoof-onnx](https://github.com/facenox/face-antispoof-onnx)
por John Raiven Olazo, licenciado sob Apache 2.0. Arquitetura MiniFAS baseada em
[Silent-Face-Anti-Spoofing](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing)
(Minivision AI, Apache 2.0).

### Ressalva — CelebA-Spoof

O modelo foi treinado no dataset **CelebA-Spoof**, cuja licença restringe uso
**não comercial** e redistribuição de dados derivados. O repositório facenox
distribui os pesos publicamente sob Apache 2.0; para **uso comercial** (ex. FrotaHub
em produção), vale confirmar com jurídico — no mesmo espírito da pendência do SFace
e MS-Celeb-1M em `docs/faces-modelos.md`.

### Demais modelos

`yunet.onnx` e `sface.onnx` continuam fora do git. Baixe do OpenCV Zoo e coloque
aqui localmente, ou use `ERA_YUNET` / `ERA_SFACE`.
