# ERA

[![CI](https://github.com/iafrotamacedo-cloud/era/actions/workflows/ci.yml/badge.svg)](https://github.com/iafrotamacedo-cloud/era/actions/workflows/ci.yml)

Motor de reconhecimento facial em Go puro. Zero dependências.

## O que é

Uma biblioteca que recebe uma imagem e devolve um vetor de 512 números
representando o rosto. Vetores parecidos = mesma pessoa.

Não é um sistema. Não tem banco, tela, login nem nuvem. É o motor que outros
sistemas importam.

## Princípios

Cada um destes é verificado pelo CI a cada push. Promessa sem verificação é
só intenção.

- **Zero dependências.** Só a biblioteca padrão do Go.
- **Sem cgo.** Cross-compila para 11 plataformas, do Raspberry Pi ao WASM.
- **Sem pesos embutidos.** Você traz o modelo `.onnx` que quiser usar. Isso
  mantém a licença do código limpa e desacopla o projeto da licença dos
  modelos.
- **Sem estado.** A biblioteca não guarda nada. Onde salvar é decisão de
  quem usa.

## Como funciona o reconhecimento

Reconhecimento facial não é modelo de linguagem — é geometria vetorial:

1. **Detecção** — achar os rostos e os 5 pontos faciais de cada um
2. **Alinhamento** — recortar e endireitar num quadrado padrão de 112×112
3. **Embedding** — a rede transforma o recorte num vetor de 512 números
4. **Comparação** — similaridade de cosseno entre vetores

A rede é treinada para que fotos da mesma pessoa virem vetores apontando na
mesma direção. Comparar identidades vira, então, medir um ângulo.

| Similaridade | Interpretação |
|---|---|
| 0,85 | mesma pessoa, foto parecida |
| 0,55 | mesma pessoa, condições diferentes |
| **0,38** | **limiar típico de decisão** |
| 0,15 | pessoas diferentes |

## Estado

Fase 1 de 7 concluída.

| Fase | Pacote | Entrega | Estado |
|---|---|---|---|
| 1 | `tensor`, `kernel` | tensor N-d, matmul bloqueado, im2col, Conv2D | **pronto** |
| 2 | `nn` | camadas: Conv2D, BatchNorm, PReLU, Linear | — |
| 3 | `onnx`, `graph` | parser de `.onnx` e executor de grafo | — |
| 4 | `embed` | ArcFace fim a fim | — |
| 5 | `detect`, `align` | detecção e alinhamento | — |
| 6 | `index` | busca 1:N, serialização | — |
| 7 | — | API pública, docs, benchmarks | — |

A Fase 4 é o marco real: quando o vetor gerado aqui bater com o do ONNX
Runtime na quarta casa decimal, a tecnologia está reproduzida.

## Desempenho

Medido num Intel i7-9750H, 6 núcleos / 12 threads. Matmul 512×512×512:

| Versão | GFLOPS | Ganho |
|---|---|---|
| Ingênua | 0,99 | — |
| Bloqueada, 1 thread | 2,60 | 2,6× (cache) |
| Bloqueada, 12 threads | 14,13 | 5,4× (paralelismo) |
| | | **14,2× no total** |

Rode você mesmo:

```
go test ./kernel/ -bench=. -run='^$'
```

### Gargalo conhecido

Convolução depthwise roda a **0,86 GFLOPS**, contra ~12 das demais camadas.
A causa: com `Groups = C`, o `Conv2D` faz uma matmul por grupo com `m = 1`, e
a divisão de trabalho por linhas desliga o paralelismo.

O resultado está correto — os testes garantem — mas o caminho é errado para
esse caso. A correção é um kernel dedicado, que desliza o kernel direto sobre
cada canal e paraleliza por canal. Prevista para a Fase 2, e relevante porque
depthwise é a operação dominante do MobileFaceNet.

## Testes

```
go test ./...              # 69 testes
go test -race ./...        # exige cgo e um compilador C
go vet ./...
```

As implementações otimizadas são conferidas contra versões ingênuas em
`kernel/reference.go`, escritas para serem óbvias em vez de rápidas. É o que
permite afirmar que uma otimização preservou o resultado.

## Licença

A definir antes da primeira release — MIT ou Apache 2.0.

Os **pesos dos modelos** têm licença própria, independente deste código, e
não são distribuídos aqui. Confira a licença do modelo que você for usar:
alguns conjuntos de pesos populares são restritos a pesquisa e não permitem
uso comercial.
