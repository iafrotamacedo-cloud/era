// Package ch acelera a busca de rota com Contraction Hierarchies.
//
// E o marco do motor de mapas. Um Dijkstra bidirecional num grafo estadual
// leva milissegundos por consulta, e uma roteirizacao de 500 paradas precisa
// de 250 mil consultas. Com a hierarquia, a mesma consulta leva
// microssegundos, e a matriz que levava minutos passa a levar segundos.
//
// # A ideia
//
// A intuicao vem de como uma pessoa descreve um trajeto longo: sai da rua de
// casa, pega uma avenida, entra na rodovia, sai da rodovia, pega uma avenida,
// chega na rua do destino. Ninguem descreve uma viagem de 300 km rua por rua,
// porque o meio do caminho e sempre feito de vias importantes.
//
// As Contraction Hierarchies transformam isso em estrutura. Cada no ganha uma
// posicao numa hierarquia -- a rua de casa embaixo, a rodovia em cima --, e a
// busca so anda para cima. As duas pontas sobem ate se encontrarem no ponto
// mais alto do caminho, e o miolo, que seria a maior parte do trabalho de um
// Dijkstra, nunca e visitado.
//
// # Como a hierarquia e construida
//
// Contraindo os nos um por um, do menos importante para o mais importante.
// Contrair um no v e remove-lo e perguntar, para cada par de vizinhos u e w
// que passava por ele: sem o v, o caminho de u ate w fica mais longo? Se
// ficar, cria-se um atalho de u ate w com o peso da soma -- um arco que
// lembra que ali existiam dois passos.
//
// Um atalho pode virar meio de outro atalho, e por isso o caminho final e
// desempacotado recursivamente ate voltar a ser feito de ruas.
//
// # Por que isto termina, e por que esta certo
//
// Decidir se o atalho e necessario exige procurar um caminho alternativo --
// uma testemunha. Essa busca tem limite de saltos, senao o pre-processamento
// nao acaba.
//
// O limite torna a busca incompleta, e e ai que mora a propriedade que faz o
// algoritmo funcionar: nao achar uma testemunha que existe cria um atalho
// desnecessario. O grafo fica maior e a consulta mais lenta -- nunca errada.
// O erro so tem um lado, e e o lado barato.
package ch

import (
	"fmt"
	"sync"

	"github.com/iafrotamacedo-cloud/era/maps/geo"
	"github.com/iafrotamacedo-cloud/era/maps/graph"
)

// arco e uma ligacao no grafo hierarquico.
//
// Pode ser uma rua de verdade ou um atalho. A diferenca esta em Meio: um
// atalho lembra por qual no ele passava, e e isso que permite desempacota-lo
// de volta em ruas. Numa aresta original, Meio e graph.NoNode.
//
// Distancia e tempo viajam os dois, mesmo que a hierarquia tenha sido
// construida para minimizar so um: quem pede a rota mais rapida quase sempre
// tambem quer saber quantos quilometros ela tem.
type arco struct {
	Para     graph.NodeID
	Metros   float32
	Segundos float32
	Meio     graph.NodeID
}

func (a arco) custo(m graph.Metric) float64 {
	if m == graph.Time {
		return float64(a.Segundos)
	}
	return float64(a.Metros)
}

// outroCusto devolve a grandeza que a hierarquia nao minimiza.
//
// Ela e acumulada junto durante a busca. A alternativa seria desempacotar o
// caminho para soma-la depois, e desempacotar custa mais que a propria busca
// -- caro demais para quem so quer preencher uma matriz.
func (a arco) outroCusto(m graph.Metric) float64 {
	if m == graph.Time {
		return float64(a.Metros)
	}
	return float64(a.Segundos)
}

// CH e um grafo com hierarquia pronta para consulta.
//
// Guarda, para cada no, apenas os arcos que sobem -- os que levam a nos mais
// importantes. Uma consulta anda so por eles, dos dois lados.
//
// E imutavel depois de construido, e seguro para uso concorrente.
type CH struct {
	m graph.Metric

	// Coordenada e identificador de cada no, copiados do grafo na construcao.
	//
	// A hierarquia guarda os seus proprios, e nao um ponteiro para o grafo de
	// origem, porque ela precisa sobreviver sozinha: depois de Load, o .osm.pbf
	// que a gerou pode nem estar na maquina. Sem isso o formato nao teria
	// sentido -- carregar uma hierarquia que ainda exige o mapa original nao
	// economiza nada.
	pontos []geo.Point
	osmIDs []int64

	// Posicao de cada no na hierarquia. Quem foi contraido primeiro e menos
	// importante e tem posicao menor.
	posicao []int32

	// CSR dos arcos que sobem. Cada arco aparece na ponta de baixo, com as
	// bandeiras dizendo em que sentido ele serve.
	//
	// Mesmo arranjo do maps/graph, e pelo mesmo motivo: a busca para tras
	// precisa dos arcos invertidos, e guardar um segundo grafo invertido
	// dobraria a memoria do que ja e a estrutura mais pesada do motor.
	offsets []uint32
	arcos   []arco
	frente  []bool
	tras    []bool

	stats Stats

	// Indice espacial montado na primeira busca por proximidade. Quem so
	// consulta entre nos ja conhecidos nao paga por ele.
	umaVez sync.Once
	grade  *geo.Grid
}

// Metric devolve a grandeza que esta hierarquia minimiza.
//
// Uma hierarquia serve a uma metrica so: a ordem de contracao depende dos
// pesos, e a mais curta em quilometros nao e a mais rapida em minutos. Quem
// precisar das duas prepara duas.
func (c *CH) Metric() graph.Metric { return c.m }

// Point devolve onde um no esta.
func (c *CH) Point(n graph.NodeID) geo.Point {
	c.conferir(n)
	return c.pontos[n]
}

// OSMID devolve o identificador do no no OpenStreetMap.
func (c *CH) OSMID(n graph.NodeID) int64 {
	c.conferir(n)
	return c.osmIDs[n]
}

// Nearest acha o no da hierarquia mais proximo de uma coordenada.
//
// E o que liga um endereco a estrutura: o cliente tem uma coordenada, a
// consulta precisa de um cruzamento. Vale olhar a distancia devolvida --
// centenas de metros quer dizer que o ponto caiu longe de qualquer estrada.
func (c *CH) Nearest(p geo.Point) (graph.NodeID, float64, bool) {
	if len(c.pontos) == 0 || !p.Valid() {
		return graph.NoNode, 0, false
	}

	c.umaVez.Do(func() {
		c.grade = geo.NewGridForRadius(500)
		for i, pt := range c.pontos {
			c.grade.Add(i, pt)
		}
	})

	achados := c.grade.Nearest(p, 1)
	if len(achados) == 0 {
		return graph.NoNode, 0, false
	}
	return graph.NodeID(achados[0].ID), achados[0].Meters, true
}

// Len devolve quantos nos a hierarquia tem.
func (c *CH) Len() int { return len(c.posicao) }

// Arcs devolve quantos arcos que sobem a hierarquia guarda.
func (c *CH) Arcs() int { return len(c.arcos) }

// Rank devolve a posicao de um no na hierarquia.
func (c *CH) Rank(n graph.NodeID) int32 {
	c.conferir(n)
	return c.posicao[n]
}

func (c *CH) conferir(n graph.NodeID) {
	if n < 0 || int(n) >= len(c.posicao) {
		panic(fmt.Sprintf("ch: no %d fora de uma hierarquia de %d nos", n, len(c.posicao)))
	}
}

// subindo devolve os arcos que sobem a partir de um no.
func (c *CH) subindo(n graph.NodeID) (arcos []arco, frente, tras []bool) {
	i, j := c.offsets[n], c.offsets[n+1]
	return c.arcos[i:j], c.frente[i:j], c.tras[i:j]
}
