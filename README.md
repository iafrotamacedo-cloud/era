# ERA

[![CI](https://github.com/iafrotamacedo-cloud/era/actions/workflows/ci.yml/badge.svg)](https://github.com/iafrotamacedo-cloud/era/actions/workflows/ci.yml)

Motores em Go puro. Zero dependências.

## O que é

Uma família de motores que outros sistemas importam. Cada um resolve um
problema que normalmente se resolve pagando uma API ou subindo um container —
e resolve dentro do processo de quem chama, sem rede, sem serviço, sem cota.

Não são sistemas. Não têm banco, tela, login nem nuvem.

| Motor | O que faz | Estado |
|---|---|---|
| [`faces`](faces/) | reconhecimento facial: imagem → vetor de 512 números | Fase 2 de 7 |
| [`maps`](maps/) | distâncias e rotas rodoviárias, roteirização | Fase 1 de 7 |
| [`docs`](docs/) | leitura de documentos | não iniciado |

Cada motor tem seu próprio README com a arquitetura, os números medidos e o
roteiro de fases.

## Princípios

Cada um destes é verificado pelo CI a cada push. Promessa sem verificação é
só intenção.

- **Zero dependências.** Só a biblioteca padrão do Go.
- **Sem cgo.** Cross-compila para 11 plataformas, do Raspberry Pi ao WASM.
- **Sem dados embutidos.** Você traz o `.onnx`, o `.osm.pbf`. Isso mantém a
  licença do código limpa e desacopla o projeto da licença dos dados.
- **Sem estado.** As bibliotecas não guardam nada. Onde salvar é decisão de
  quem usa.
- **Referência antes de otimização.** Toda implementação rápida é conferida
  nos testes contra uma versão óbvia, escrita para ser fácil de ler em vez de
  rápida. É o que permite afirmar que uma otimização preservou o resultado.

## Por que um repositório só

Os motores parecem independentes e não são, em dois pontos concretos:

**Código compartilhado que já existe.** O `.onnx` do `faces` e o `.osm.pbf`
do `maps` são ambos protobuf. O leitor de wire format que os dois precisam é
o mesmo. Em repositórios separados isso vira um quarto repositório ou uma
duplicação que sai de sincronia.

**As mesmas promessas, verificadas do mesmo jeito.** Os dois CIs eram
praticamente o mesmo arquivo — e já tinham começado a divergir antes da
fusão. Um só não diverge.

O que se paga por isso: uma versão vale para os três, e quem importar só o
`maps` baixa o módulo inteiro. O compilador só constrói o que é importado,
então o custo é de download, não de binário.

## Estrutura

```
era/
├── go.mod          module github.com/iafrotamacedo-cloud/era
├── faces/          tensor, kernel, nn, onnx, embed, detect, index
├── maps/           geo, dist, osm, graph, ch, geocode, vrp
└── docs/           a definir
```

Os caminhos de importação seguem as pastas:

```go
import (
    "github.com/iafrotamacedo-cloud/era/faces/tensor"
    "github.com/iafrotamacedo-cloud/era/maps/geo"
)
```

## Testes

```
go test ./...              # todos os motores
go test ./maps/...         # só um
go test -race ./...        # exige cgo e um compilador C
go vet ./...
```

## Licença

A definir antes da primeira release — MIT ou Apache 2.0.

Os **dados** têm licença própria, independente deste código, e não são
distribuídos aqui:

- **Pesos de modelo** (`faces`): confira a licença do modelo que for usar —
  alguns conjuntos populares são restritos a pesquisa e não permitem uso
  comercial.
- **Dados de mapa** (`maps`): o OpenStreetMap é ODbL — uso comercial
  liberado, atribuição obrigatória, e o *share-alike* recai sobre a base de
  dados derivada, não sobre o código que a consulta.
