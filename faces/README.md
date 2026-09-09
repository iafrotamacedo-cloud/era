# ERA · faces

Motor de reconhecimento facial em Go puro. Zero dependências.

## O que é

Uma biblioteca que recebe uma imagem e devolve um vetor de números
representando o rosto. Vetores parecidos = mesma pessoa.

O tamanho do vetor é do modelo, não da biblioteca: o SFace produz 128
dimensões, o ArcFace produz 512.

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
3. **Embedding** — a rede transforma o recorte num vetor de números
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

Fases 1 a 5 de 7 concluídas.

| Fase | Pacote | Entrega | Estado |
|---|---|---|---|
| 1 | `tensor`, `kernel` | tensor N-d, matmul bloqueado, im2col, Conv2D, depthwise | **pronto** |
| 2 | `nn` | camadas, workspace reutilizável, fusão de BatchNorm | **pronto** |
| 3 | `onnx`, `graph` | parser de `.onnx` e executor de grafo | **pronto** |
| 4 | validação | modelo real conferido contra o ONNX Runtime | **pronto** |
| 5 | `detect`, `align` | detecção e alinhamento | **pronto** |
| 6 | `index` | busca 1:N, serialização | — |
| 7 | — | API pública, docs, benchmarks | — |

**A Fase 4 fechou o marco do projeto.** O vetor gerado aqui bate com o do ONNX
Runtime — a meta era a quarta casa decimal, o resultado foi a quinta.

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

## Leitura de modelos

O `.onnx` é uma mensagem Protobuf. Como a ERA não tem dependências, o leitor
do formato binário mora em [`internal/protowire`](../internal/protowire) — e é
compartilhado com o motor `maps`, que lê `.osm.pbf`, outro Protobuf de esquema
completamente diferente.

O pacote [`faces/onnx`](onnx/) põe o esquema do ONNX em cima disso e devolve a
estrutura crua: nós, pesos, atributos. Ele **não executa nada** — quem executa
é o `graph`. Separar as duas coisas mantém erro de leitura de arquivo e erro
de execução de rede como problemas distintos.

Os testes montam modelos `.onnx` sintéticos campo a campo, com o escritor do
`protowire`. Isso permite construir os casos difíceis — arquivo truncado, tipo
inesperado, pesos em arquivo externo — que ninguém consegue de propósito no
mundo real.

Suporta `raw_data` e os campos tipados, campos repetidos empacotados ou um a
um, `float16`/`bfloat16`/`float64`/inteiros, e dimensões simbólicas. Pesos em
arquivo externo são detectados e viram **erro explícito** — devolver tensores
vazios em silêncio produziria uma rede que roda e dá resultado errado.

## Execução do grafo

O pacote [`graph`](graph/) pega a estrutura crua do `onnx`, resolve as ligações
entre os nós, monta as camadas do `nn` com os pesos certos e executa.

O ONNX não tem ponteiros: as ligações são **nomes**. A saída `conv1_out` de um
nó é a entrada `conv1_out` do próximo. Executar é manter uma tabela de nome
para tensor, alimentada por três origens — pesos, entradas de quem chama, e
saídas produzidas pelos nós conforme rodam.

**Ordem de execução.** A especificação exige que os nós venham em ordem
topológica. O `graph` não confia nisso e reordena. Custa uma passada e elimina
uma classe de falha difícil de diagnosticar: um exportador fora de ordem
produziria "valor não encontrado" no meio da rede, sem pista da causa.

**Montagem separada da execução.** Forma dos pesos, atributos coerentes,
combinações não suportadas — tudo é validado uma vez, ao carregar. O que sai
dali roda sem conferir nada, uma vez por rosto.

### Operadores

| | |
|---|---|
| Com pesos | `Conv` `BatchNormalization` `PRelu` `Gemm` `MatMul` |
| Espacial | `GlobalAveragePool` `MaxPool` `AveragePool` |
| Ativação | `Relu` `LeakyRelu` `Sigmoid` `Tanh` `Clip` `Softmax` |
| Aritmética | `Add` `Sub` `Mul` `Div`, com transmissão de forma estilo NumPy |
| Forma | `Flatten` `Reshape` `Transpose` `Concat` `Unsqueeze` `Squeeze` |
| Neutros | `Identity` `Dropout` `Constant` |

Um operador ausente é reportado junto com todos os outros que faltarem, para
que carregar um modelo novo não vire uma sequência de tentativas.

### O que vira erro em vez de aproximação

- Saídas de treino (`BatchNormalization` com estatísticas do lote, `Dropout`
  com máscara) — executar só a primeira saída **rodaria**, e daria resultado
  errado
- Padding assimétrico e `auto_pad=SAME_*`
- `Gemm` com `alpha`/`beta`/`transA` fora do usual

A regra é a mesma dos pesos em arquivo externo: uma rede que carrega, roda e
dá resposta errada é o pior desfecho possível numa biblioteca.

## Validação contra o ONNX Runtime

O modelo **SFace** (MobileFaceNet, Apache 2.0) carrega e executa: 88 operações,
9,6 milhões de parâmetros, entrada `[1,3,112,112]`, saída `[1,128]`. Nenhum
operador faltou.

O vetor gerado aqui, comparado com o do ONNX Runtime na mesma entrada:

| | |
|---|---|
| **Similaridade de cosseno** | **1,000000000** |
| Diferença absoluta máxima | 1,9 × 10⁻⁵ |
| Norma L2 | 2,221406 (referência: 2,221428) |
| Tempo por rosto | 83,6 ms |

A meta era bater na quarta casa decimal. Bateu na quinta. A divergência que
resta é acúmulo de arredondamento de `float32` em ordem diferente de soma —
não há como eliminá-la, e ela não muda a identidade que o vetor representa.

O teste vive em `graph/modelo_real_test.go` e **pula quando o modelo não está
presente**, porque o repositório não guarda pesos. Para rodar:

```
# baixe o SFace do OpenCV Zoo para models/sface.onnx
go test ./faces/graph/ -run SFace -v
```

A entrada e o vetor de referência estão em `graph/testdata/`, gerados por
`testdata/gerar_referencia.py`. A entrada é lida de arquivo pelos dois lados —
reproduzir o mesmo pseudo-aleatório em Python e em Go seria uma fonte de
divergência sem relação com o que está sob teste.

### Consumo de memória

O workspace chega a **68 MB** para esse modelo de 37 MB. O alocador é uma
pilha que só reaproveita memória *entre* passagens, não dentro de uma: cada um
dos 88 tensores intermediários recebe memória nova. Análise de tempo de vida
cortaria bastante. Fica registrado como conhecido, não corrigido.

## Detecção e alinhamento

**`align`** endireita o rosto antes do reconhecimento. A rede foi treinada com
rostos numa posição canônica — olhos numa altura fixa, boca noutra, tamanho
padronizado — e alimentá-la com um rosto torto degrada a acurácia muito mais
do que a intuição sugere.

A transformação é de **similaridade**: rotação, escala uniforme e translação.
Sem cisalhamento, de propósito — uma afim completa encaixaria os 5 pontos
exatamente, mas distorcendo o rosto, e a distorção muda a identidade que a
rede enxerga.

O ajuste usa números complexos em vez de decomposição em valores singulares.
Tratando `(x,y)` como `x + yi`, a similaridade vira uma multiplicação e o
mínimo quadrados tem solução fechada. Além de ser menos código que o
algoritmo de Umeyama, a forma complexa **não consegue produzir reflexão** —
multiplicação complexa é sempre rotação pura. Um rosto espelhado seria uma
solução válida para o mínimo quadrados e uma catástrofe para o
reconhecimento.

**`detect`** envolve o YuNet (MIT, 232 KB). O modelo não devolve uma lista de
rostos: devolve doze tensores — classificação, objectness, caixa e pontos —
em três escalas com passos de 8, 16 e 32 pixels. São 8.400 âncoras, e cada
uma responde *"se houvesse um rosto centrado perto de mim, ele estaria
assim"*.

```
score = √(cls × obj)
cx = (coluna + bbox₀) · passo      largura = exp(bbox₂) · passo
cy = (linha  + bbox₁) · passo      altura  = exp(bbox₃) · passo
```

O logaritmo no tamanho não é enfeite: ele faz a rede prever a **razão** entre
o tamanho do rosto e o da âncora, em vez da diferença absoluta — e razão é o
que se mantém estável entre um rosto perto e um longe.

### De onde vem a fórmula

A decodificação não está documentada de forma verificável em lugar nenhum.
As fórmulas foram **derivadas dos dados**: comparando as saídas cruas do
modelo com o que o `cv2.FaceDetectorYN` devolve na mesma imagem, e conferindo
em cinco limiares diferentes.

| Limiar | Detecções | Diferença máxima |
|---|---|---|
| 0,001 | 638 | 5,19 × 10⁻⁴ px |
| 0,005 | 273 | 5,19 × 10⁻⁴ px |
| 0,010 | 236 | 5,19 × 10⁻⁴ px |
| 0,050 | 92 | 4,58 × 10⁻⁴ px |

Contagem idêntica em todos, geometria batendo em meio milésimo de pixel.
`detect_test.go` guarda essa verificação; sem ela, uma refatoração poderia
perder a fórmula sem que nada quebrasse de forma visível.

O YuNet de 2023 tem entrada **fixa** em 640×640. Imagens de outro tamanho
passam por letterbox — redimensionadas preservando a proporção, com o resto
preenchido — e as coordenadas voltam convertidas. Esticar para o quadrado
seria mais simples e degradaria a detecção: um rosto achatado deixa de
parecer rosto.

## Licença

A definir antes da primeira release — MIT ou Apache 2.0.

Os **pesos dos modelos** têm licença própria, independente deste código, e
não são distribuídos aqui. Confira a licença do modelo que você for usar:
alguns conjuntos de pesos populares são restritos a pesquisa e não permitem
uso comercial.
