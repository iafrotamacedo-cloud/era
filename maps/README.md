# ERA · maps

Motor de distâncias e rotas rodoviárias em Go puro. Zero dependências.

## O que é

Uma biblioteca que responde três perguntas de logística sem chamar API
nenhuma: *quanto tempo e quantos quilômetros daqui até ali*, *quem está
perto de quem*, e *em que ordem visitar as paradas*.

Não é um sistema. Não tem banco, tela, login nem nuvem. É o motor que outros
sistemas importam — irmão da [ERA](https://github.com/iafrotamacedo-cloud/era)
e com os mesmos princípios.

## O reframe

Não existe distância rodoviária real sem dado de estrada. A escolha nunca foi
*"com ou sem mapa"* — é **em qual máquina o mapa mora**.

O OpenStreetMap resolve isso: dado livre, baixável, sem cota, sem termo de uso
proibindo cache. Com ele, o mapa mora aqui dentro. É o mesmo movimento da ERA
com os pesos `.onnx` — você traz o dado, o motor é seu.

| Alternativa | Por que não |
|---|---|
| Google Maps | O termo de uso proíbe cachear o resultado e usá-lo fora do mapa deles. É restrição contratual, não preço. |
| OSRM demo, ORS, GraphHopper | Cota de alguns milhares de chamadas por dia e nenhum SLA. Servem a protótipo, morrem em produção. |
| OSRM/Valhalla em container | Funciona bem, mas é um serviço a mais para operar. O motor precisa caber *dentro* do sistema que o usa. |

## Princípios

Cada um destes é verificado pelo CI a cada push. Promessa sem verificação é
só intenção.

- **Zero dependências.** Só a biblioteca padrão do Go.
- **Sem cgo.** Cross-compila para 11 plataformas, do Raspberry Pi ao WASM.
- **Sem dados embutidos.** Você traz o extrato `.osm.pbf` da sua região. Isso
  mantém a licença do código limpa e o repositório leve.
- **Sem estado.** A biblioteca não guarda nada. Onde salvar é decisão de quem
  usa.

## As quatro perguntas

Quase todo mundo trata "calcular distância" como um problema só. São quatro,
e o esforço se distribui de forma bem diferente do que a intuição sugere:

| Problema | O que é | Onde dói |
|---|---|---|
| Geocodificação | endereço → lat/lon | **é aqui que o projeto morre**, não no roteamento |
| Distância / matriz | N×N metros e segundos | resolvido, se houver grafo |
| Rota | o caminho real, mão única, restrição de caminhão | o OSM cobre parcialmente |
| Otimização (VRP) | ordem das paradas, qual veículo | **é aqui que o dinheiro está** |

Matriz perfeita não economiza um litro de diesel. Ordenar as paradas
economiza de 15% a 30%.

## Estado

Fase 1 de 7 concluída.

| Fase | Pacote | Entrega | Estado |
|---|---|---|---|
| 1 | `geo` | ponto, haversine, Vincenty, retângulo, índice em grade | **pronto** |
| 2 | `dist` | interface `Distancer`, cache em disco, fator de desvio | — |
| 3 | `protowire`, `osm` | protobuf na mão + leitor de `.osm.pbf` | — |
| 4 | `graph` | grafo CSR, contração de nós grau 2, Dijkstra bidirecional | — |
| 5 | `ch` | Contraction Hierarchies, matriz muitos-para-muitos | — |
| 6 | `geocode` | endereço → ponto: CEP, CNEFE/IBGE, ruas do OSM | — |
| 7 | `vrp` | Clarke-Wright, 2-opt/Or-opt, capacidade e janela de tempo | — |

A Fase 2 já é útil sozinha: haversine calibrado com o hodômetro da operação
resolve agrupamento e raio de atendimento enquanto o motor é construído.

A Fase 5 é o marco real: quando a rota calculada aqui bater com a do OSRM em
±1%, a tecnologia está reproduzida.

## O que a Fase 1 entrega

O vocabulário geográfico, e o filtro barato que decide quais pares de pontos
merecem o custo de um cálculo de rota de verdade. Numa operação com mil
clientes há um milhão de pares, e quase todos podem ser descartados por
geometria antes de tocar no grafo rodoviário.

```go
d := geo.Haversine(deposito, cliente)          // metros, sobre a esfera

g := geo.NewGridForRadius(10_000)              // grade para buscas de ~10 km
for i, c := range clientes {
    g.Add(i, c.Ponto)
}
perto := g.Within(veiculo, 10_000)             // quem está a 10 km, ordenado
dez := g.Nearest(veiculo, 10)                  // os 10 mais próximos
```

### Três réguas, três propósitos

| Função | Custo | Erro | Para quê |
|---|---|---|---|
| `Ruler.SquaredDistance` | 1,9 ns | ~0,3% em 250 km | ordenar e comparar contra um raio |
| `Ruler.Distance` | 7,8 ns | ~0,3% em 250 km | laço interno, agrupamento |
| `Haversine` | 74 ns | ~0,5% (esfera vs. Terra) | uso geral |
| `Vincenty` | 399 ns | submilimétrico | referência: mede o erro das outras |

O erro do `Ruler`, medido pelos testes contra `Haversine`:

| Raio da região | Erro máximo |
|---|---|
| 50 km | 0,05% |
| 250 km | 0,30% |
| 500 km | 0,75% |
| 1.000 km | 2,09% |

Um estado brasileiro cabe folgado na faixa útil. Distância entre capitais
distantes, não.

### O índice paga

100.000 pontos espalhados por uma área de 300 km de raio, busca de 10 km,
num Intel i7-9750H:

| Como | Tempo | |
|---|---|---|
| Varredura linear | 7,29 ms | — |
| Grade | 34,9 µs | **209× mais rápido** |

Montar o índice custa 14,6 ms para os 100.000 pontos. Numa operação isso é
pago uma vez, no carregamento do cadastro, e amortizado por milhares de
consultas.

Rode você mesmo:

```
go test ./geo/ -bench=. -run='^$'
```

## Por que a haversine e não a lei dos cossenos

As duas fórmulas são algebricamente idênticas. A lei dos cossenos é
numericamente péssima em distâncias curtas: `cos(σ)` fica tão perto de 1 que a
subtração perde quase todos os dígitos significativos.

Medido pelo teste, para dois pontos a **1 cm** um do outro: **849% de erro**.
A 100 km, as duas concordam.

Numa operação logística, onde entregas na mesma rua são a regra e não a
exceção, isso deixa de ser detalhe acadêmico. `LawOfCosinesRef` está no
repositório só para o teste demonstrar a diferença em vez de a afirmar.

## Testes

```
go test ./...              # 74 testes
go test -race ./...        # exige cgo e um compilador C
go vet ./...
```

As implementações rápidas são conferidas contra versões óbvias ou mais exatas
em `geo/reference.go`. É o que permite afirmar que uma otimização preservou o
resultado — e é também o que produz os números de erro desta página: eles são
medidos pelos testes, não estimados.

Dois bugs reais foram encontrados assim durante a Fase 1, ambos em casos que
nenhuma inspeção casual pegaria: um ponto exatamente na borda de um retângulo
de busca caindo de fora por um bit de arredondamento, e uma busca de
abrangência mundial varrendo uma única coluna da grade.

## Licença

A definir antes da primeira release — MIT ou Apache 2.0.

Os **dados de mapa** têm licença própria, independente deste código, e não são
distribuídos aqui. O OpenStreetMap é ODbL: uso comercial liberado, atribuição
obrigatória, e o *share-alike* recai sobre a base de dados derivada, não sobre
o código que a consulta. Quem publicar um serviço com esses dados precisa
creditar o OpenStreetMap.
