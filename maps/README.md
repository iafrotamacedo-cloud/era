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

Fases 1 a 5 de 7 concluídas.

| Fase | Pacote | Entrega | Estado |
|---|---|---|---|
| 1 | `geo` | ponto, haversine, Vincenty, retângulo, índice em grade | **pronto** |
| 2 | `dist` | interface `Distancer`, cache em disco, fator de desvio | **pronto** |
| 3 | `protowire`, `osm` | protobuf na mão + leitor de `.osm.pbf` | **pronto** |
| 4 | `graph` | grafo CSR, contração de nós grau 2, Dijkstra bidirecional | **pronto** |
| 5 | `ch` | Contraction Hierarchies, matriz muitos-para-muitos | **pronto** |
| 6 | `geocode` | endereço → ponto: CEP, CNEFE/IBGE, ruas do OSM | — |
| 7 | `vrp` | Clarke-Wright, 2-opt/Or-opt, capacidade e janela de tempo | — |

A Fase 5 é o marco real: quando a rota calculada aqui bater com a do OSRM em
±1%, a tecnologia está reproduzida.

## O que a Fase 5 entrega

A hierarquia que faz a consulta deixar de percorrer o mapa.

```go
c, _ := ch.Prepare(g, graph.Time)      // caro, uma vez só
custo, ok := c.Cost(de, para)          // microssegundos
m := c.Matrix(paradas, paradas)        // a matriz do dia
```

A intuição vem de como alguém descreve um trajeto longo: sai da rua de casa,
pega uma avenida, entra na rodovia, sai, pega uma avenida, chega. Ninguém
descreve 300 km rua por rua. As Contraction Hierarchies transformam isso em
estrutura — cada nó ganha uma posição, e a busca só anda para cima. As duas
pontas sobem até se encontrarem no cume, e o miolo do mapa nunca é visitado.

### O erro só tem um lado

Decidir se um atalho é necessário exige procurar um caminho alternativo — uma
testemunha. Essa busca tem limite de saltos, senão o pré-processamento não
termina.

O limite a torna incompleta, e é aí que mora a propriedade que faz o algoritmo
funcionar: **não achar uma testemunha que existe cria um atalho desnecessário,
nunca uma resposta errada.** Medido, no mesmo grafo:

| Limite | Atalhos | Respostas diferentes |
|---|---|---|
| 1 salto | 320 | — |
| 12 saltos | 98 | **nenhuma** |

### Contra o Dijkstra da Fase 4

| | Consulta | Matriz 100×100 |
|---|---|---|
| Dijkstra bidirecional | 644 µs | 1.243 ms |
| Hierarquia | **126 µs** | **9,4 ms** |

5,1× na consulta e 132× na matriz. O ganho da matriz é maior porque o
algoritmo de baldes faz cada metade da busca uma vez só: os destinos sobem
deixando bilhetes, as origens sobem recolhendo. O encontro das duas buscas
deixa de ser procurado e passa a ser encontrado.

Os 5,1× são um **piso**, não o número típico. A medição é numa grade, que é o
pior caso para a hierarquia: não há rodovia, não há gargalo, todo caminho tem
mil alternativas equivalentes. Uma malha rodoviária real tem a estrutura que o
algoritmo explora.

### A cadeia de verificação

O CH é conferido contra o Dijkstra bidirecional da Fase 4, que é conferido
contra o Dijkstra óbvio do `reference.go`. Vinte e cinco grafos sorteados,
duas métricas, todos os pares — mais uma grade com mão única onde 2.561 pares
têm ida diferente da volta.

Os caminhos desempacotados também são verificados rua por rua: um custo certo
com um caminho impossível seria pior que um erro, porque parece bom.

### O `ch.Router` fecha a troca de motor

A interface `dist.Distancer`, escrita na Fase 2, aceita agora a terceira
implementação — e trocar entre elas continua sendo uma linha:

```go
dist.NewEstimator(cal)          // Fase 2: haversine calibrado, erra ~10%
graph.NewRouter(g, graph.Time)  // Fase 4: rota de verdade, milissegundos
ch.NewRouter(c)                 // Fase 5: a mesma rota, microssegundos
```

Contra o motor da Fase 4, na mesma grade:

| | `Distance` | `Matrix` 60×60 |
|---|---|---|
| `graph.Router` | 432 µs | 377 ms |
| `ch.Router` | **220 µs** | **10,7 ms** |

Duas vezes na consulta avulsa e **35× na matriz**. A diferença entre os dois
números é o ponto: `Matrix` não é um laço de `Distance`. Usa o algoritmo de
baldes, em que cada metade da busca é feita uma vez só — e é a matriz que uma
roteirização pede.

Para responder `dist.Leg`, que traz metros **e** segundos, a busca acumula a
segunda grandeza junto com a primeira. A alternativa seria desempacotar o
caminho só para somá-la, e desempacotar custa mais que a própria busca.

### O gargalo mudou de lugar, e foi consertado

Com a hierarquia, `geo.Nearest` — encaixar a coordenada no cruzamento mais
próximo — passou a pesar 22 µs por ponta, uns 20% de uma consulta. No motor da
Fase 4 era ruído; com o CH, não era mais.

Foi reescrito (veja **O índice paga**, na Fase 1) e caiu para 3,6 µs. O efeito
no motor:

| | Antes | Depois |
|---|---|---|
| `ch.Router.Distance` | 220 µs | 198 µs |
| `ch.Router.Matrix` 60×60 | 10,7 ms | **7,4 ms** |

A matriz ganha mais porque encaixa cada ponto uma vez e depois só busca.

### O `.eramap` guarda a hierarquia pronta

O preparo é caro e cresce mais que linearmente. Sem gravar o resultado, todo
processo que sobe paga esse custo de novo:

| | Tempo |
|---|---|
| `Prepare` (3.600 cruzamentos) | 461 ms |
| `Load` do `.eramap` | **0,85 ms** |

544×, a 700 MB/s. Extrapolando para um estado, é a diferença entre dezenas de
minutos e uma fração de segundo.

```go
c, _ := ch.Prepare(g, graph.Time)   // uma vez, na máquina que preparar
c.SaveFile("ceara.eramap")

c, _ := ch.LoadFile("ceara.eramap") // em toda partida do processo
```

O arquivo é **autossuficiente**: carregá-lo não exige o `.osm.pbf` que o
gerou. É o ponto inteiro — uma hierarquia que ainda precisasse do mapa
original não economizaria nada. Por isso o `CH` guarda as próprias
coordenadas e identificadores, e `Nearest` funciona depois de carregar.

Escrito à mão, com cabeçalho mágico e byte de versão, como o formato do
`dist`. Não é `gob`: `gob` amarra o arquivo em disco à forma exata das structs
Go, e renomear um campo passaria a quebrar a leitura de arquivos antigos sem
que nada avisasse.

`Load` confere as invariantes, e não só os tamanhos — posições são uma
permutação, índices crescem, todo arco sobe. Um arquivo corrompido do tamanho
certo passaria pela checagem de tamanho e quebraria no meio de uma consulta,
que é muito pior de diagnosticar do que uma recusa na abertura.

## A comparação com o OSRM

Os testes verificam que as três implementações de rota concordam entre si: a
hierarquia bate com o Dijkstra bidirecional, que bate com o Dijkstra óbvio. É
uma cadeia sólida, e ela **não prova o que mais importa**.

Se uma etiqueta do OpenStreetMap tiver sido lida errado — mão única implícita,
restrição de acesso, o que conta como estrada — as três erram juntas e em
silêncio, porque todas leem o mapa pelo mesmo perfil. Só uma referência de
fora pega isso.

O programa está escrito e testado:

```bash
go run ./maps/cmd/osrmcompare -mapa ceara-latest.osm.pbf -n 500
```

Ele sorteia pares de **cruzamentos do próprio grafo**, e não coordenadas
soltas: cada motor encaixa a coordenada à sua maneira — o nosso no cruzamento
mais próximo, o OSRM em qualquer ponto de um trecho —, e sortear às cegas
mediria a diferença entre os dois encaixes, não entre as duas rotas. O quanto
o OSRM se afastou vem na resposta dele, e o par é descartado quando isso
passar do limite.

O relatório traz mediana, p90, p99 e pior caso, separando **erro absoluto** de
**viés**: metade errando 10% para mais e metade 10% para menos dá mediana de
10% e viés zero, e as duas coisas dizem problemas diferentes. E lista os cinco
piores com as coordenadas na mão, para abrir no mapa — é isso que aponta o que
consertar; uma mediana não aponta nada.

### O que a primeira rodada encontrou

Rodado sobre o extrato do Nordeste (421 MB), 149 pares dentro do Ceará, contra
o servidor público do projeto OSRM.

Primeiro, o grafo real. A previsão do README se confirmou:

```
1.750.698 de 6.269.993 vias são estrada
25.026.641 nós de via → 2.916.332 cruzamentos    88,3% contraídos
3.967.854 trechos, em 29 s, 2,17 GB de heap
```

O leitor de `.osm.pbf` engoliu 65,7 milhões de nós em 2,2 s — 194 MB/s, quase
três vezes o que o benchmark sintético media. Dado real comprime melhor que
dado sorteado.

E a comparação:

| | mediana | p90 | p99 | viés |
|---|---|---|---|---|
| **Distância** | **0,39%** | 7,60% | 35,58% | +0,06% |
| Tempo | 20,68% | 27,16% | 35,76% | −20,68% |

**A distância bate.** Mediana de 0,39% e viés de 0,06% dizem que a construção
do grafo, a leitura de mão única, a classificação do que é estrada e os
algoritmos concordam com uma implementação independente. Isso é o que a Fase 5
prometia verificar.

Com a ressalva de que o alvo do roteiro era ±1% e **isso vale para a mediana,
não para a distribuição**: um em cada dez pares erra mais de 7%.

### O que a rodada mostrou estar errado

**A cauda eram estradas de terra.** Investigando o pior caso — 29,7 km aqui
contra 50,9 km no OSRM — a nossa rota passava por 20 vias `unclassified`, das
quais 8 com `surface` de `ground`, `dirt` ou `unpaved`. Estradas de barro do
sertão, percorridas a 40 km/h como se fossem asfalto.

Foi corrigido, e a correção precisou de duas tentativas:

| | Sem `surface` | Multiplicador | **Teto** |
|---|---|---|---|
| Distância, mediana | 0,39% | 0,48% | **0,36%** |
| Distância, p90 | 7,60% | 7,70% | **6,84%** |
| Distância, pior | −41,6% | +48,3% | **−37,3%** |
| Tempo, mediana | 20,68% | 18,94% | 19,81% |

A primeira versão multiplicava a velocidade por um fator. Melhorou o tempo e
**piorou a distância**, com um caso novo de +48%: fugindo do barro por um
desvio de 77 km no asfalto onde o OSRM fazia 52. Penalidade demais.

O erro era o modelo, não os números. Estrada de barro é lenta em termos
absolutos, não em proporção à classe da via — um `primary` de terra não anda a
70% de 70, anda a velocidade de estrada de terra. Trocado por **teto**, todos
os percentis de distância melhoraram e o caso de +48% sumiu.

Vale saber o que isso não resolve: **a maioria das vias do interior não tem a
etiqueta.** Na rota que motivou a mudança, 26 de 49 vias vinham sem `surface`.
O `tracktype` é lido como substituto quando existe, mas para o resto nada muda.

**O tempo continua 21% otimista, sempre para o mesmo lado.** Viés de −19,8%
com mediana de 19,8% — quase todo par erra na mesma direção. São as
velocidades genéricas do `graph.Car()`, que não foram medidas em operação
nenhuma. Não é problema de etiqueta: é calibração, e a saída acordada é o
`dist.Calibrate` da Fase 2, que mede velocidade efetiva a partir do hodômetro.

### As etiquetas de acesso, e o que elas não mudaram

O `access` do OpenStreetMap tem **precedência**, não acumulação: vale a chave
mais específica que existir, e as demais nem são olhadas. Uma servidão com

```
access=private
motor_vehicle=yes
```

está liberada para carro. E uma via `access=yes` com `motor_vehicle=no` está
fechada. O modelo anterior olhava pares chave=valor soltos e errava nos dois
sentidos.

Foi trocado por uma lista de chaves em ordem de especificidade —
`motor_vehicle`, `vehicle`, `access` — e um conjunto de valores que negam. O
que fica de fora é permitido, e **essa lista é uma decisão de logística**:
`destination`, `delivery` e `customers` descrevem exatamente o caminhão que vai
entregar. Bloqueá-los faria a roteirização recusar o condomínio, o posto e a
fazenda — os lugares para onde a entrega vai.

O efeito medido:

| | Sem acesso | Com acesso |
|---|---|---|
| Vias no grafo | 1.784.370 | 1.749.627 |
| Cruzamentos | 2.967.019 | 2.915.025 |
| Rotas diferentes em 150 pares | — | **nenhuma** |

Tira 34.743 vias, 1,9% da malha, e **não muda uma única rota** da amostra. As
vias bloqueadas são servidões e áreas fechadas que nenhuma rota intermunicipal
usa. O valor da correção está nos casos que a amostra não cobre — entrega
urbana em condomínio fechado, via de serviço particular —, onde rotear por
dentro seria erro de verdade.

Vale registrar assim, com o número: a mudança está certa e não melhora nada
que se meça aqui.

### A amostra precisou ficar estável

A primeira versão do comparador sorteava índices de nó do grafo. Servia para
medir a qualidade de uma versão e **não servia para comparar duas**: mudar o
perfil muda quantas vias entram, os índices deslocam, e a amostra vira outra.
A primeira tentativa de medir o `access` deu p90 de 6,84% para 13,56%, e o
que tinha mudado era a amostra.

Agora sorteia coordenadas no retângulo e encaixa depois. Enquanto os
cruzamentos continuarem existindo — que é o caso de quase todos — a amostra é
a mesma entre execuções, e um A/B mede o que se quer medir.

A amostra por coordenada é mais difícil que a por índice: alcança lugares
remotos onde o encaixe fica a quilômetros e a malha é rala. Com ela, a mediana
fica em 0,33% e o p90 em 16,1%. Os números desta seção e das anteriores vêm de
amostragens diferentes e **não devem ser comparados entre si** — daqui para a
frente, podem.

### O erro de método que quase virou diagnóstico errado

A primeira rodada deu mediana de 6,81% com viés de −6,81%: nossas rotas
sistematicamente mais curtas que as do OSRM. Parecia bug de roteamento.

Não era. Eu pedia ao nosso motor a rota mais **curta** e ao OSRM a rota mais
**rápida**, que é o padrão dele. Os dois números mediam a diferença entre dois
objetivos, não entre dois motores. Com as duas pontas minimizando tempo, o
erro mediano de distância caiu de 6,81% para 0,39%.

Fica o registro porque o número errado era convincente: tinha viés
consistente, tinha explicação plausível e apontava para um culpado real. O que
o desmentiu foi olhar o que cada lado estava otimizando.

O programa é o único lugar da ERA que fala HTTP, e ele não é parte do motor —
verificável com `go list -deps ./maps/geo ./maps/dist ./maps/osm ./maps/graph
./maps/ch | grep net/`, que não devolve nada.

## O que a Fase 4 entrega

O mapa vira grafo, e o grafo responde rota.

```go
g, err := graph.BuildFile("ceara.osm.pbf", graph.Car())
de, _, _ := g.Nearest(deposito)
para, _, _ := g.Nearest(cliente)
p, ok := g.Route(de, para, graph.Time)   // p.Meters, p.Seconds, p.Nodes
```

E a Fase 2 finalmente troca de motor sem que o chamador saiba:

```go
d := dist.NewCache(graph.NewRouter(g, graph.Time))   // era dist.NewEstimator(cal)
```

### A contração é a maior decisão do pacote

A maioria dos nós do OpenStreetMap existe só para desenhar a curva da rua.
Contando quantas estradas citam cada nó **antes** de montar, o grafo nunca
chega a existir em tamanho cheio. Numa rua de 11 nós sem cruzamento, sobram 2
nós e 1 trecho — 82% contraídos, e o comprimento do trecho é a soma dos dez
pedaços, não a linha reta entre as pontas.

O `Stats` do grafo traz esse número para o extrato que você usar; num mapa
urbano real a faixa costuma ser de 80% a 90%.

### O bidirecional contra o óbvio

Um Dijkstra comum explora um círculo em volta da origem. Buscando também a
partir do destino, são dois círculos de metade do raio — e dois círculos de
raio r/2 têm metade da área de um de raio r.

Numa grade de 90 mil cruzamentos:

| | Tempo | Alocações |
|---|---|---|
| Dijkstra óbvio (`RouteRef`) | 13,6 ms | 149.700 |
| Bidirecional (`Route`) | **5,6 ms** | 5 |

O `reference.go` não é código morto: é contra ele que o bidirecional é
conferido, em 200 grafos sorteados, duas métricas, todos os pares. O critério
de parada de uma busca bidirecional é a parte que engana — o encontro das duas
frentes **não** garante o melhor caminho —, e é o tipo de erro que produz uma
rota que parece boa e não é.

### A conta do cache se inverteu

A Fase 2 mediu que o `dist.Cache` **perdia** contra o `Estimator`: uma busca
num mapa de 12 MB custa mais que a multiplicação que ela evita. Contra uma
busca no grafo, a mesma busca no mapa é ruído:

| | Sem cache | Com cache |
|---|---|---|
| Contra o `Estimator` | 130 ns | 100 ns |
| Contra o `Router` | 1,13 ms | **63 ns** |

O cache foi escrito na Fase 2 para este momento, e a documentação dele dizia
isso antes de ser verdade. Agora é.

### O que ainda não presta

Uma matriz 500×500 são 250 mil buscas: minutos. É exatamente para isso que a
Fase 5 existe. Até lá o `Router.Matrix` serve para conferir a resposta, não
para rodar em produção.

A memória da construção é proporcional aos nós de via, não ao tamanho do
arquivo — um mapa de identificador para coordenada. Para um estado são
centenas de MB; para o Brasil inteiro aperta, e a saída é o `.eramap` da Fase
5, que guarda o grafo já montado.

A geometria dos trechos é descartada: entre dois cruzamentos sobra o
comprimento, não a lista de curvas. Achar caminho não precisa dela; desenhar
o caminho num mapa precisa.

## O que a Fase 3 entrega

O leitor de `.osm.pbf` — o formato binário em que o OpenStreetMap distribui
extratos. Lê e só: não monta grafo, não filtra estrada, não sabe o que é uma
rua. Mesmo corte que o `faces` faz entre `onnx`, que lê o modelo, e `graph`,
que o executa.

```go
osm.Scan(f, osm.Handler{
    Way: func(w osm.Way) error {
        if _, ok := w.Tags.Get("highway"); ok { ... }
        return nil
    },
})
```

O `internal/protowire` já existia — foi escrito pelo `faces` para ler `.onnx`,
e inclui o zigzag que o ONNX **não** usa e o `.osm.pbf` usa em toda diferença
de coordenada. Era a aposta do monorepo, e ela pagou: a Fase 3 do `maps` foi
só o esquema por cima.

### `nil` no handler quer dizer "não decodifique"

`Handler` é um struct de funções, não uma interface. A diferença é que um
campo `nil` significa pular, e o leitor nunca toca nos `DenseNodes` — que são
a maior parte do arquivo.

Isso importa porque montar um grafo exige duas passadas: primeiro as vias,
para saber quais nós importam; depois esses nós. Medido, sobre o mesmo
arquivo:

| | Tempo | Entrada |
|---|---|---|
| Tudo | 16,8 ms | 71 MB/s |
| Só vias (`Node` nil) | **11,8 ms** | 102 MB/s |
| Nada (handler vazio) | 7,3 ms | 164 MB/s |

13,1 milhões de entidades por segundo na leitura completa, num i7-9750H.

### Paralelo por dentro, ordenado por fora

Os blocos são independentes e o `inflate` é caro, então a decodificação é
paralela. Mas os callbacks são chamados de uma goroutine só, na ordem do
arquivo: o handler não precisa se preocupar com concorrência, e duas leituras
do mesmo arquivo produzem a mesma sequência.

Isso tem um preço, e ele foi medido em vez de estimado:

| | 1 trabalhador | 8 trabalhadores | Ganho |
|---|---|---|---|
| Só o envelope | 43,7 ms | 22,5 ms | 1,95× |
| Leitura completa | 75,1 ms | 51,5 ms | 1,46× |

A parte paralela escala. O teto é a entrega serial — cerca de 26 ns por
entidade. Foi uma troca deliberada: um handler que precisasse ser seguro para
concorrência empurraria essa complexidade para todo mundo que usa a
biblioteca. Se a Fase 4 esbarrar nesse teto, a saída é uma variante que
entregue lotes em vez de uma entidade por vez.

### Falhar alto

Um mapa lido pela metade em silêncio vira rota errada, e rota errada em
logística vira caminhão no lugar errado. Então o leitor recusa, com erro
nomeado: arquivo que exige capacidade não implementada, compressão que a ERA
não tem (lzma, lz4, zstd — todas exigiriam dependência), arquivo truncado,
granularidade zero, índice de tabela apontando para o vazio, listas de
tamanhos diferentes, tipo de membro que não existe.

Como o repositório não guarda dados de mapa, os testes **escrevem** os
`.osm.pbf` que leem. A vantagem escondida é que dá para produzir arquivos que
não existiriam naturalmente — é assim que cada uma dessas recusas é
verificada.

## O que a Fase 2 entrega

A interface que o sistema chamador enxerga, e a primeira implementação dela.

```go
cal, rep, err := dist.Calibrate(historico)   // hodômetro da operação
log.Print(rep)                               // erro mediano 4,0%, p90 7,1%

d, _ := dist.NewEstimator(cal)               // implementa dist.Distancer
leg, _ := d.Distance(ctx, deposito, cliente) // metros e segundos
m, _ := d.Matrix(ctx, paradas, paradas)      // a matriz do dia
```

Na Fase 5 o `Estimator` é trocado pelo motor de grafo e nada mais muda.

### O fator de desvio não é uma constante

Distância rodoviária é a linha reta vezes um fator. Esse fator cai conforme a
distância cresce: ir a duas quadras pode custar seis por causa das mãos
únicas, atravessar o estado segue a rodovia, que já foi traçada para ser
curta. Um fator único erra nas duas pontas ao mesmo tempo.

`Calibrate` mede um fator por faixa de distância, com a **mediana** das razões
— não a média. Histórico real de frota tem motorista que passou na oficina e
hodômetro digitado errado; a média persegue esses pontos, a mediana os ignora.
Medido pelos testes, com 5% das viagens corrompidas, o fator se move 0,17%.

Quanto isso vale, sobre viagens que não entraram na calibração:

| | Erro mediano |
|---|---|
| Fator genérico | 19,7% |
| Calibrado, fator único | 11,4% |
| Calibrado, por faixa | **3,9%** |

`DefaultCalibration` existe para o dia zero e se declara genérica
(`Generic() == true`), para o sistema poder avisar em vez de fingir precisão
que não tem.

### O cache não paga contra o `Estimator`

O cache era a otimização óbvia. Os benchmarks disseram outra coisa:

| | Sem cache | Com cache |
|---|---|---|
| Consulta individual | 130 ns | **100 ns** |
| Matriz 500×500 | **10 ms** | 17 ms |

A matriz cacheada perde, e o motivo é estrutural: meio milhão de buscas num
mapa de 12 MB são meio milhão de idas à memória principal, cada uma mais cara
que a multiplicação que o `Estimator` faria no lugar. Não há cache que ganhe
de uma conta de dezenas de nanossegundos.

O cache está escrito para a Fase 5, quando atrás dele estiver uma busca no
grafo. Com o `Estimator`, use-o para consultas avulsas e deixe as matrizes
passarem direto.

Isso só apareceu porque o benchmark foi escrito antes da conclusão. A primeira
versão era ainda 8× pior: travava o mutex uma vez por célula e recalculava a
chave em cada uma.

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

**`Nearest` foi reescrito na Fase 5**, quando a hierarquia acelerou a busca de
rota o bastante para o encaixe da coordenada virar 20% do custo de uma
consulta. Ele chamava `Within`, que junta todos os candidatos do raio e ordena
todos — para devolver um.

| | Antes | Depois | |
|---|---|---|---|
| `Nearest(·, 1)` | 34,2 µs | **5,9 µs** | 5,8× |
| `Nearest(·, 10)` | 34,2 µs | **14,5 µs** | 2,4× |
| Alocação | 8,6 KB | 32 B | |

Duas mudanças. A varredura das células foi separada da coleta, e `Nearest`
passou a guardar só os k melhores — a alocação caiu 270×, mas o tempo, só
16%. Foi a medição que mostrou onde estava o custo de verdade: **varrer**. O
raio inicial era uma célula inteira, e uma célula dimensionada para buscas de
10 km guarda centenas de pontos.

Agora o raio inicial sai da densidade medida da própria grade. Isso é seguro
por causa da propriedade que já sustentava o algoritmo: achar k pontos dentro
de um raio qualquer significa que são os k mais próximos do planeta. Um chute
ruim custa uma varredura a mais, nunca um ponto errado.

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
go test ./...              # 318 testes no repositorio
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
