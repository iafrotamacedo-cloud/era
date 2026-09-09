// Package graph monta e percorre o grafo rodoviario extraido de um .osm.pbf.
//
// E onde o mapa deixa de ser uma lista de pontos e vias e vira uma coisa que
// responde "como se vai daqui ate ali". O osm da Fase 3 le o arquivo; este
// pacote decide o que e estrada, junta os pedacos e acha caminho.
//
// # A representacao
//
// O grafo e guardado em CSR -- compressed sparse row --, que e um jeito
// antigo e simples de guardar grafo esparso: um vetor com todas as arestas em
// sequencia, e outro dizendo onde comecam as de cada no.
//
//	offsets: [0, 3, 5, 9, ...]
//	arestas: [a a a | b b | c c c c | ...]
//	          no 0    no 1   no 2
//
// A alternativa obvia -- uma fatia de arestas por no -- custaria um cabecalho
// de fatia e uma alocacao por no, e espalharia as arestas pela memoria. Em
// CSR, as arestas de um no sao vizinhas: o processador as traz numa leitura
// so. Numa busca que visita milhoes de nos, isso e a diferenca entre segundos
// e dezenas de segundos.
//
// O preco e que o grafo fica imutavel depois de montado. Para roteirizacao
// isso nao custa nada: o mapa nao muda durante a consulta.
//
// # O que nao esta aqui
//
// A geometria dos trechos e descartada. Entre dois cruzamentos sobra o
// comprimento acumulado, nao a lista de curvas. Achar caminho nao precisa
// dela; desenhar o caminho num mapa precisa, e quem precisar disso vai ter de
// guarda-la -- e mais que dobra a memoria do grafo.
package graph

import (
	"fmt"
	"sync"

	"github.com/iafrotamacedo-cloud/era/maps/geo"
)

// NodeID identifica um no dentro deste grafo.
//
// Nao e o identificador do OpenStreetMap: e o indice compacto que a
// construcao atribuiu, de 0 a Len()-1. int32 e de proposito -- o Brasil
// inteiro tem alguns milhoes de cruzamentos, muito abaixo dos dois bilhoes
// que cabem aqui, e a metade da memoria de um int64 aparece em cada aresta.
type NodeID int32

// NoNode e o valor que significa "nenhum no".
const NoNode NodeID = -1

// Edge e uma ligacao entre dois cruzamentos, ja com os pedacos intermediarios
// somados.
//
// Cada trecho da rua aparece duas vezes no grafo, uma em cada ponta, e as
// bandeiras dizem quais sentidos aquela ponta pode usar. E o que permite a
// busca para tras do Dijkstra bidirecional andar pelas arestas invertidas sem
// que exista um segundo grafo invertido na memoria.
//
// Meters e Seconds em float32: numa aresta de 10 km o float32 distingue
// milimetros, e a metade da memoria decide se o grafo de um estado cabe.
type Edge struct {
	To      NodeID
	Meters  float32
	Seconds float32

	Forward  bool // da para ir de quem guarda esta aresta ate To
	Backward bool // da para vir de To ate quem guarda esta aresta
}

// Metric escolhe o que a busca minimiza.
type Metric uint8

// As duas grandezas que o grafo guarda por aresta.
const (
	// Distance minimiza quilometragem. E o que interessa a combustivel e a
	// desgaste.
	Distance Metric = iota
	// Time minimiza tempo. E o que interessa a janela de entrega e a jornada
	// do motorista, e costuma ser a resposta que o cliente espera.
	Time
)

func (m Metric) String() string {
	if m == Time {
		return "tempo"
	}
	return "distancia"
}

func (e Edge) custo(m Metric) float64 {
	if m == Time {
		return float64(e.Seconds)
	}
	return float64(e.Meters)
}

// Graph e um grafo rodoviario pronto para consulta.
//
// E imutavel depois de construido, e seguro para uso concorrente.
type Graph struct {
	pontos  []geo.Point
	osmIDs  []int64
	offsets []uint32
	arestas []Edge
	stats   Stats

	// O indice espacial e montado na primeira vez que alguem pergunta por
	// proximidade. Quem so roteiriza entre nos ja conhecidos nao paga por ele.
	umaVez sync.Once
	grade  *geo.Grid
}

// Len devolve quantos nos o grafo tem.
func (g *Graph) Len() int { return len(g.pontos) }

// Edges devolve quantas arestas o grafo tem, contando cada trecho uma vez por
// ponta.
func (g *Graph) Edges() int { return len(g.arestas) }

// Point devolve onde um no esta.
func (g *Graph) Point(n NodeID) geo.Point {
	g.conferir(n)
	return g.pontos[n]
}

// OSMID devolve o identificador do no no OpenStreetMap.
//
// Serve para conferir uma rota contra o site do OSM, que e como se depura
// grafo rodoviario na pratica.
func (g *Graph) OSMID(n NodeID) int64 {
	g.conferir(n)
	return g.osmIDs[n]
}

// EdgesOf devolve as arestas de um no.
//
// A fatia aponta para dentro do grafo: leia, nao escreva.
func (g *Graph) EdgesOf(n NodeID) []Edge {
	g.conferir(n)
	return g.arestas[g.offsets[n]:g.offsets[n+1]]
}

func (g *Graph) conferir(n NodeID) {
	if n < 0 || int(n) >= len(g.pontos) {
		panic(fmt.Sprintf("graph: no %d fora de um grafo de %d nos", n, len(g.pontos)))
	}
}

// Nearest acha o no do grafo mais proximo de uma coordenada.
//
// E o que liga um endereco ao grafo: o cliente tem uma coordenada, a rota
// precisa de um cruzamento. Usa o indice em grade da Fase 1, montado na
// primeira chamada.
//
// Devolve a distancia em linha reta ate o no encontrado. Vale olhar para ela:
// centenas de metros quer dizer que o ponto caiu longe de qualquer estrada --
// coordenada errada, ou uma regiao que o extrato nao cobre.
func (g *Graph) Nearest(p geo.Point) (NodeID, float64, bool) {
	if len(g.pontos) == 0 || !p.Valid() {
		return NoNode, 0, false
	}

	g.umaVez.Do(func() {
		// Celulas de uns 500 m: pequenas o bastante para a busca olhar pouca
		// coisa, grandes o bastante para nao virar uma celula por no.
		g.grade = geo.NewGridForRadius(500)
		for i, pt := range g.pontos {
			g.grade.Add(i, pt)
		}
	})

	achados := g.grade.Nearest(p, 1)
	if len(achados) == 0 {
		return NoNode, 0, false
	}
	return NodeID(achados[0].ID), achados[0].Meters, true
}
