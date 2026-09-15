# faces — escolha dos modelos

A ERA não distribui pesos. Este documento registra **quais modelos foram
avaliados, quais foram escolhidos e por quê** — para que a decisão não dependa
de alguém lembrar da conversa em que ela foi tomada.

Pesquisa feita em 08/09/2026. As licenças foram lidas na fonte, não de
memória. Fontes ao final.

## Contexto da decisão

A ERA vai rodar em sistemas internos da Frota Macedo (**uso comercial**) e
pode ser aberta depois (**open source**). Isso exige pesos com licença
permissiva — o critério mais restritivo dos dois manda.

## Escolhidos

| Papel | Modelo | Licença | Origem |
|---|---|---|---|
| Detecção + 5 pontos | **YuNet** | MIT *(upstream BSD-3)* | OpenCV Zoo |
| Reconhecimento | **SFace** | Apache 2.0 | OpenCV Zoo |
| Anti-spoof passivo | **MiniFAS** (facenox) | Apache 2.0 *(código)* | [face-antispoof-onnx](https://github.com/facenox/face-antispoof-onnx) |

### Por que MiniFAS (facenox)

O alvo original era o **anti-spoof-mn3** do Open Model Zoo (MIT, MobileNetV3,
128×128), mas não há ONNX público confiável — os IR do OpenVINO vinham
corrompidos (~1 KB). O **MiniFAS** do facenox tem ONNX pronto
(`best_model.onnx`, ~1,9 MB), entrada `1×3×128×128` float normalizado,
saída `1×2` logits (real/spoof). Treinado em CelebA-Spoof.

Pré-processamento igual ao `demo.py` do repositório: recorte quadrado com
expansão 1,5× da caixa, letterbox 128×128, RGB ÷ 255, CHW. Classificação por
diferença de logits (`real - spoof`), não softmax.

Variável de ambiente: `ERA_ANTISPOOF`. Padrão: `models/anti-spoof.onnx`.

Este arquivo **está versionado** no repositório (exceção ao `.gitignore`) para
facilitar clone e CI; atribuição e ressalvas em `models/README.md`.

**Ressalva de dataset:** treinado em **CelebA-Spoof** (uso não comercial na
licença do dataset). O facenox distribui os pesos sob Apache 2.0; para produto
comercial, confirmar com jurídico — mesma classe de risco que SFace × MS-Celeb-1M.

### Calibração (webcam PC — 2026-09-14)

Ferramenta: `python faces/cmd/calibrar/calibrar.py` (na raiz do ERA).
Dados: `calibracao_20260914_2101.csv` — 56 amostras reais, 29 fakes
(foto/tela), webcam padrão do PC.

| Métrica | Real (mediana) | Fake (mediana) |
|---|---|---|
| `logit_diff` (real − spoof) | −0,65 | −4,70 |
| Variância de landmarks (15 frames) | 0,010 | 0,005 |

**Anti-spoof:** fakes ficaram bem negativos; rostos reais oscilaram muito
(luz/distância/parado ao marcar). Limiar inicial recomendado:

| Parâmetro | Valor | Efeito na sessão |
|---|---|---|
| `spoof.Options.Limiar` | **0,27** | ~61% reais aceitos, ~3% fakes passam |
| alternativa mais permissiva | **0,18** | ~77% reais, ~7% fakes passam |

Equivale a `logit_diff` ≥ **−1,0** (conservador) ou ≥ **−1,5** (permissivo).
O padrão do código continua **0,5** até o FrotaHub passar `AntiSpoofConfig`.

**Liveness:** o padrão `MinVar: 0,0005` não separou nada nesta câmera.
Usar **`VerifyLive`** com movimento de cabeça, não frame único:

| Parâmetro | Valor | Efeito na sessão |
|---|---|---|
| `liveness.Options.MinFrames` | **15** | janela do script de calibração |
| `liveness.Options.MinVar` | **0,012** | ~48% reais com movimento; ~10% fakes passam |

Exemplo para integração:

```go
AntiSpoof: &faces.AntiSpoofConfig{
    Options:  spoof.Options{ Limiar: 0.27 },
    Liveness: liveness.Options{ MinFrames: 15, MinVar: 0.012 },
},
```

Repetir calibração se mudar câmera, iluminação ou ambiente de obra.

### Por que YuNet

Devolve, por rosto detectado, exatamente o que o alinhamento precisa:

```
x1, y1, w, h          caixa delimitadora
x_re, y_re            olho direito
x_le, y_le            olho esquerdo
x_nt, y_nt            ponta do nariz
x_rcm, y_rcm          canto direito da boca
x_lcm, y_lcm          canto esquerdo da boca
score                 confiança
```

É o conjunto canônico de 5 pontos que o alinhamento do ArcFace usa. Não há
conversão a fazer.

Precisão no WIDER Face: 0,884 (fácil) / 0,866 (médio) / 0,750 (difícil).
Detecta rostos de ~10×10 a ~300×300 pixels.

### Por que SFace

É **MobileFaceNet** treinado com a função de perda SFace — a mesma
arquitetura para a qual o kernel de depthwise da Fase 1 foi otimizado. Os
12,5× conquistados lá se aplicam diretamente a este modelo.

Precisão reportada: 0,9940.

## Descartados

### InsightFace (`buffalo_*`)

**Motivo: os pesos são de pesquisa não-comercial.**

> "The training data containing the annotation (and the models trained with
> these data) are available for non-commercial research purposes only."

O *código* do InsightFace é MIT sem restrição — a confusão é fácil de fazer.
Os *pesos* não são. O `buffalo_l` tem inclusive um canal próprio para
licenciamento comercial (`recognition-oss-pack@insightface.ai`).

São os melhores modelos disponíveis. Não podemos usá-los.

### ONNX Model Zoo — ArcFace LResNet100E-IR

**Motivo: procedência dos dados de treino, e tamanho.**

O arquivo é Apache 2.0, mas foi treinado no **MS-Celeb-1M**, que a Microsoft
retirou do ar em 2019 depois que uma investigação do *Financial Times* mostrou
que o dataset continha jornalistas, artistas e ativistas que nunca
consentiram. A licença do arquivo não resolve a origem dos dados.

Além disso são 248,9 MB de pesos — inferência na casa dos segundos por rosto
em CPU, não dos milissegundos.

## Pendências — verificar antes de produção

**1. Em que dataset os pesos do SFace foram treinados.** O README do OpenCV
Zoo não informa. O artigo original treinou em CASIA-WebFace, VGGFace2 **e
MS-Celeb-1M**. Se os pesos publicados vierem do MS-Celeb-1M, a mesma questão
de procedência do ArcFace se aplica, apesar da licença Apache 2.0.

**2. A dimensão do vetor do SFace.** A documentação da ERA fala em 512
dimensões, herdado do ArcFace. O SFace provavelmente produz **128**. Não muda
a arquitetura — a ERA lê o que o modelo devolver — mas os números na
documentação precisam ser corrigidos assim que o modelo for carregado.

**3. Precisão na população real.** 0,9940 é LFW, um benchmark saturado onde a
diferença entre 99,4% e 99,8% quase não significa nada. O que importa é a
taxa de acerto em obra, com capacete, contraluz e poeira — e isso só medindo.

Vale registrar: os modelos permissivos são de fato mais fracos que os
melhores restritos. É o preço da licença limpa, e é um preço consciente.

## Como isso afeta o código

Quase nenhum modelo é embutido: `yunet.onnx` e `sface.onnx` ficam fora do git
(`.gitignore`); quem usa aponta o caminho ou baixa do OpenCV Zoo. Exceção:
`models/anti-spoof.onnx` (~1,9 MB), versionado por conveniência — ver
`models/README.md`.

O executor de grafo roda qualquer ONNX de reconhecimento facial — trocar de
modelo é trocar um arquivo e reconferir o limiar de similaridade. Se alguma
licença mudar, ou se aparecer um modelo melhor, a troca não mexe em código.

## Fontes

- InsightFace — https://github.com/deepinsight/insightface
- ONNX Model Zoo, ArcFace — https://github.com/onnx/models/tree/main/validated/vision/body_analysis/arcface
- OpenCV Zoo, SFace — https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface
- OpenCV Zoo, YuNet — https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet
- libfacedetection (upstream do YuNet) — https://github.com/ShiqiYu/libfacedetection
- SFace, artigo — https://arxiv.org/abs/2205.12010
- Exposing.ai, sobre o MS-Celeb-1M — https://exposing.ai/msceleb/
- MIT Technology Review, sobre datasets retirados — https://www.technologyreview.com/2021/08/13/1031836/ai-ethics-responsible-data-stewardship/
- OpenCV, `cv::FaceDetectorYN` — https://docs.opencv.org/4.x/df/d20/classcv_1_1FaceDetectorYN.html
