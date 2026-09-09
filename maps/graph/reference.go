package graph

import "container/heap"

// Este arquivo tem o Dijkstra obvio: um so sentido, uma fila de prioridade da
// biblioteca padrao, nenhuma esperteza.
//
// Existe para ser conferido contra, e nao para ser usado. O Dijkstra
// bidirecional de dijkstra.go e varias vezes mais rapido, mas tem criterio de
// parada, dois sentidos e reconstrucao de caminho costurada no meio -- tres
// lugares onde um erro produz uma rota que parece boa e nao e. Esta versao e
// simples o bastante para dar para ler e afirmar que esta certa, e os testes
// comparam as duas em grafos sorteados.
//
// Nao otimize este arquivo. O valor dele e ser obvio.

// RouteRef acha o caminho mais curto pelo Dijkstra classico.
//
// E a implementacao de referencia. Use Route, que faz o mesmo mais rapido.
func (g *Graph) RouteRef(de, para NodeID, m Metric) (Path, bool) {
	if !g.valido(de) || !g.valido(para) {
		return Path{}, false
	}
	if de == para {
		return Path{Nodes: []NodeID{de}}, true
	}

	const infinito = 1e300
	dist := make([]float64, g.Len())
	anterior := make([]NodeID, g.Len())
	pronto := make([]bool, g.Len())
	for i := range dist {
		dist[i] = infinito
		anterior[i] = NoNode
	}
	dist[de] = 0

	fila := &filaSimples{{no: de, custo: 0}}
	heap.Init(fila)

	for fila.Len() > 0 {
		item := heap.Pop(fila).(entrada)
		if pronto[item.no] {
			continue
		}
		pronto[item.no] = true
		if item.no == para {
			break
		}

		for _, e := range g.EdgesOf(item.no) {
			if !e.Forward {
				continue
			}
			novo := dist[item.no] + e.custo(m)
			if novo < dist[e.To] {
				dist[e.To] = novo
				anterior[e.To] = item.no
				heap.Push(fila, entrada{no: e.To, custo: novo})
			}
		}
	}

	if dist[para] >= infinito {
		return Path{}, false
	}
	return g.refazer(anterior, de, para, m), true
}

// refazer monta o caminho andando de tras para frente pelos antecessores.
func (g *Graph) refazer(anterior []NodeID, de, para NodeID, m Metric) Path {
	var nos []NodeID
	for n := para; n != NoNode; n = anterior[n] {
		nos = append(nos, n)
		if n == de {
			break
		}
	}
	for i, j := 0, len(nos)-1; i < j; i, j = i+1, j-1 {
		nos[i], nos[j] = nos[j], nos[i]
	}
	return g.medir(nos, m)
}

// medir soma distancia e tempo ao longo de um caminho ja formado.
//
// As duas grandezas sao devolvidas mesmo quando a busca minimizou so uma:
// quem pediu a rota mais rapida quase sempre tambem quer saber quantos
// quilometros ela tem.
//
// Precisa saber qual metrica foi minimizada por causa das ruas paralelas.
// Dois cruzamentos podem estar ligados por mais de um trecho -- uma avenida e
// a rua de tras --, e a busca usou o melhor *naquela* metrica. Escolher aqui
// pelo comprimento, quando a busca escolheu pelo tempo, faria a rota ser
// relatada com numeros de um caminho que ninguem percorreu.
func (g *Graph) medir(nos []NodeID, m Metric) Path {
	p := Path{Nodes: nos}
	for i := 1; i < len(nos); i++ {
		var melhor Edge
		achou := false
		for _, e := range g.EdgesOf(nos[i-1]) {
			if e.To != nos[i] || !e.Forward {
				continue
			}
			if !achou || e.custo(m) < melhor.custo(m) {
				melhor, achou = e, true
			}
		}
		p.Meters += float64(melhor.Meters)
		p.Seconds += float64(melhor.Seconds)
	}
	return p
}

func (g *Graph) valido(n NodeID) bool { return n >= 0 && int(n) < len(g.pontos) }

// entrada e um no na fila de prioridade.
type entrada struct {
	no    NodeID
	custo float64
}

// filaSimples e a fila de prioridade da biblioteca padrao.
//
// Passa por interface e aloca a cada Push -- e por isso que o Dijkstra rapido
// tem a sua propria. Aqui isso nao importa: o que importa e nao ter escrito
// um heap a mao no arquivo que serve de referencia.
type filaSimples []entrada

func (f filaSimples) Len() int           { return len(f) }
func (f filaSimples) Less(i, j int) bool { return f[i].custo < f[j].custo }
func (f filaSimples) Swap(i, j int)      { f[i], f[j] = f[j], f[i] }
func (f *filaSimples) Push(x any)        { *f = append(*f, x.(entrada)) }
func (f *filaSimples) Pop() any {
	velho := *f
	n := len(velho)
	item := velho[n-1]
	*f = velho[:n-1]
	return item
}
