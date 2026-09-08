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

Fases 1 e 2 de 7 concluídas.

| Fase | Pacote | Entrega | Estado |
|---|---|---|---|
| 1 | `tensor`, `kernel` | tensor N-d, matmul bloqueado, im2col, Conv2D, depthwise | **pronto** |
| 2 | `nn` | camadas, workspace reutilizável, fusão de BatchNorm | **pronto** |
| 3 | `onnx`, `graph` | parser de `.onnx` e executor de grafo | — |
| 4 | `embed` | ArcFace fim a fim | — |
| 5 | `detect`, `align` | detecção e alinhamento | — |
| 6 | `index` | busca 1:N, serialização | — |
| 7 | — | API pública, docs, benchmarks | — |

A Fase 4 é o marco real: quando o vetor gerado aqui bater com o do ONNX
Runtime na quarta casa decimal, a tecnologia está reproduzida.

## Desempenho

Medido num Intel i7-9750H, 6 núcleos / 12 threads.

**Matmul 512×512×512** — o efeito de cada otimização, isolado:

| Versão | GFLOPS | Ganho |
|---|---|---|
| Ingênua | 1,08 | — |
| Bloqueada, 1 thread | 3,53 | 3,3× (cache) |
| Bloqueada, 12 threads | 13,72 | 3,9× (paralelismo) |
| | | **12,7× no total** |

**Camadas com a geometria de uma rede de reconhecimento facial:**

| Camada | GFLOPS |
|---|---|
| Entrada 112×112×3, stride 2 | 10,95 |
| Depthwise 3×3, 56×56×64 | 10,85 |
| Pontual 1×1, 56×56, 64→128 | 12,57 |
| 3×3, 14×14, 128→256 | 13,62 |

Rode você mesmo:

```
go test ./kernel/ -bench=. -run='^$'
```

### Depthwise tem kernel próprio

Convolução depthwise não passa por im2col nem por matmul. Com `Groups = C`,
cada grupo viraria uma matmul de uma linha só: o paralelismo por linhas não
teria o que dividir, e o rearranjo de memória do im2col seria pago sem volume
que o amortizasse. Medido nesse caminho: **0,86 GFLOPS**.

O kernel dedicado desliza o filtro direto sobre cada canal e paraleliza por
canal. **10,85 GFLOPS — 12,5× mais rápido.** `Conv2D` reconhece o caso e
despacha sozinho.

Importa porque MobileFaceNet, o modelo alvo, é feito majoritariamente de
depthwise.

Com stride 2 o número cai para ~4,9 GFLOPS: a leitura passa a ser espaçada e
o caminho contíguo rápido não se aplica. Em tempo absoluto ainda é mais barato,
porque há um quarto das posições de saída para calcular.

## Camadas

O pacote `nn` monta as camadas sobre os kernels: `Conv2D`, `BatchNorm`,
`PReLU`, `ReLU`, `Linear`, `MaxPool2D`, `GlobalAvgPool`, `Flatten`, `Add`
e `Sequential`.

Duas decisões atravessam o pacote:

**Workspace.** Uma rede tem dezenas de camadas, cada uma produzindo um tensor
intermediário. Alocar tudo isso por rosto seria pressão de coletor de lixo num
caminho que roda em milissegundos. O `Workspace` aloca na primeira passagem e
reaproveita nas seguintes — zero alocação de dados daí em diante. É feito de
blocos que nunca são realocados, porque um buffer único que crescesse
invalidaria as fatias já entregues.

Restam ~90 alocações pequenas por passagem (≈4 KB): os slices de forma dos
tensores. Eliminá-las exigiria um pool de formas; o ganho não paga a
complexidade por ora.

**Fusão de BatchNorm.** Na inferência, BatchNorm é `y = x*escala + desloc` por
canal, e os dois cabem dentro dos pesos da convolução anterior:

```
conv:  y = W*x + b
bn:    z = y*escala + desloc
       z = (W*escala)*x + (b*escala + desloc)
```

`Sequential.Fuse()` faz a absorção e remove a camada. É uma reescrita
algébrica exata — há teste comparando a rede antes e depois.

O ganho medido é de **~4%**, não mais que isso: BatchNorm custa O(elementos)
enquanto a convolução custa O(elementos × K² × canais), então a camada
eliminada já era barata. Vale por ser de graça, não por ser decisiva.

## Testes

```
go test ./...              # 146 testes
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
