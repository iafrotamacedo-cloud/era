package ch

import (
	"math"

	"github.com/iafrotamacedo-cloud/era/maps/graph"
)

// Route acha o caminho mais curto pela hierarquia.
//
// Aloca as estruturas de busca a cada chamada. Para muitas consultas seguidas
// use NewQuery, que as reaproveita.
func (c *CH) Route(de, para graph.NodeID) (graph.Path, bool) {
	return c.NewQuery().Route(de, para)
}

// Cost devolve so o custo, sem montar o caminho.
//
// Desempacotar os atalhos de volta em ruas custa mais que a propria busca
// quando o caminho e longo. Quem so quer preencher uma matriz de distancias
// nao deveria pagar por isso.
func (c *CH) Cost(de, para graph.NodeID) (float64, bool) {
	return c.NewQuery().Cost(de, para)
}

// Query guarda as estruturas de uma consulta para reaproveita-las.
//
// Nao e seguro para uso concorrente: use uma por goroutine.
type Query struct {
	c *CH

	df, db []float64

	// A outra metrica, acumulada ao longo do mesmo caminho. Custa dois
	// vetores a mais do tamanho do grafo, e poupa desempacotar o caminho so
	// para descobrir quantos quilometros tem a rota mais rapida.
	of, ob []float64

	af, ab []uint32 // arco pelo qual cada no foi alcancado
	pf, pb []graph.NodeID
	vf, vb []uint32

	geracao uint32
	hf, hb  filaCusto

	nos []graph.NodeID // rascunho do caminho desempacotado
}

// NewQuery cria um consultador reutilizavel.
func (c *CH) NewQuery() *Query {
	n := c.Len()
	return &Query{
		c:  c,
		df: make([]float64, n), db: make([]float64, n),
		of: make([]float64, n), ob: make([]float64, n),
		af: make([]uint32, n), ab: make([]uint32, n),
		pf: make([]graph.NodeID, n), pb: make([]graph.NodeID, n),
		vf: make([]uint32, n), vb: make([]uint32, n),
	}
}

// Cost acha o custo do melhor caminho, na metrica da hierarquia.
//
// # Por que a busca so sobe
//
// Toda rota otima na hierarquia tem a forma de um V: sobe da origem ate um no
// mais importante, e desce dali ate o destino. Isso e uma propriedade da
// construcao, nao uma aposta -- contrair um no cria o atalho que preserva as
// distancias por cima dele, entao o caminho que descia e subia de novo sempre
// tem um equivalente que so sobe.
//
// Por isso a busca da frente e a de tras sobem as duas, e se encontram no
// cume. O miolo do mapa, que um Dijkstra visitaria inteiro, nunca e tocado.
//
// # O criterio de parada muda
//
// Num Dijkstra bidirecional comum, para-se quando a soma dos dois topos
// alcanca o melhor candidato. Aqui isso seria errado: os dois lados sobem, e
// o custo de subir nao se compara com o de descer da mesma forma.
//
// O criterio correto e por lado: uma direcao acabou quando o menor custo
// pendente dela ja alcancou o melhor caminho conhecido. A busca so termina
// quando as duas acabaram.
func (q *Query) Cost(de, para graph.NodeID) (float64, bool) {
	melhor, encontro := q.buscar(de, para)
	if encontro == graph.NoNode {
		return 0, false
	}
	return melhor, true
}

// Leg devolve as duas grandezas do melhor caminho, sem montar o caminho.
//
// E o que um dist.Distancer precisa: distancia e tempo, sem pagar o
// desempacotamento que so serve para desenhar a rota.
func (q *Query) Leg(de, para graph.NodeID) (metros, segundos float64, ok bool) {
	custo, encontro := q.buscar(de, para)
	if encontro == graph.NoNode {
		return 0, 0, false
	}
	if de == para {
		return 0, 0, true
	}

	outra := q.of[encontro] + q.ob[encontro]
	if q.c.m == graph.Time {
		return outra, custo, true
	}
	return custo, outra, true
}

// Route acha o caminho e o desempacota de volta em ruas.
func (q *Query) Route(de, para graph.NodeID) (graph.Path, bool) {
	_, encontro := q.buscar(de, para)
	if encontro == graph.NoNode {
		return graph.Path{}, false
	}
	if de == para {
		return graph.Path{Nodes: []graph.NodeID{de}}, true
	}

	var p graph.Path
	q.nos = q.nos[:0]

	// A metade da frente, do encontro para tras ate a origem, e depois
	// invertida.
	inicio := len(q.nos)
	for n := encontro; n != de; {
		a := q.c.arcos[q.af[n]]
		anterior := q.pf[n]
		p.Meters += float64(a.Metros)
		p.Seconds += float64(a.Segundos)

		var miolo []graph.NodeID
		q.c.expandir(anterior, n, a.Meio, &miolo)
		q.nos = append(q.nos, n)
		for i := len(miolo) - 1; i >= 0; i-- {
			q.nos = append(q.nos, miolo[i])
		}
		n = anterior
	}
	q.nos = append(q.nos, de)
	inverter(q.nos[inicio:])

	// A metade de tras ja sai na ordem certa.
	for n := encontro; n != para; {
		a := q.c.arcos[q.ab[n]]
		proximo := q.pb[n]
		p.Meters += float64(a.Metros)
		p.Seconds += float64(a.Segundos)

		var miolo []graph.NodeID
		q.c.expandir(n, proximo, a.Meio, &miolo)
		q.nos = append(q.nos, miolo...)
		q.nos = append(q.nos, proximo)
		n = proximo
	}

	p.Nodes = append([]graph.NodeID(nil), q.nos...)
	return p, true
}

// buscar roda as duas buscas para cima e devolve o melhor custo e o cume.
func (q *Query) buscar(de, para graph.NodeID) (float64, graph.NodeID) {
	c := q.c
	if !valido(c, de) || !valido(c, para) {
		return 0, graph.NoNode
	}
	if de == para {
		return 0, de
	}

	q.geracao++
	q.hf.limpar()
	q.hb.limpar()

	q.df[de], q.vf[de], q.pf[de], q.of[de] = 0, q.geracao, graph.NoNode, 0
	q.db[para], q.vb[para], q.pb[para], q.ob[para] = 0, q.geracao, graph.NoNode, 0
	q.hf.push(de, 0)
	q.hb.push(para, 0)

	melhor := math.Inf(1)
	encontro := graph.NoNode

	for {
		frenteAcabou := q.hf.Len() == 0 || q.hf.topo() >= melhor
		trasAcabou := q.hb.Len() == 0 || q.hb.topo() >= melhor
		if frenteAcabou && trasAcabou {
			break
		}
		if !frenteAcabou {
			q.expandirLado(true, &melhor, &encontro)
		}
		if !trasAcabou {
			q.expandirLado(false, &melhor, &encontro)
		}
	}
	return melhor, encontro
}

func (q *Query) expandirLado(frente bool, melhor *float64, encontro *graph.NodeID) {
	c := q.c
	h, dist, visto, arcoDe, anterior, outra := &q.hb, q.db, q.vb, q.ab, q.pb, q.ob
	outroLadoDist, outroVisto := q.df, q.vf
	if frente {
		h, dist, visto, arcoDe, anterior, outra = &q.hf, q.df, q.vf, q.af, q.pf, q.of
		outroLadoDist, outroVisto = q.db, q.vb
	}

	no, custo := h.pop()
	if visto[no] != q.geracao || custo > dist[no] {
		return
	}

	inicio := c.offsets[no]
	arcos, podeFrente, podeTras := c.subindo(no)

	for i, a := range arcos {
		if frente && !podeFrente[i] {
			continue
		}
		if !frente && !podeTras[i] {
			continue
		}

		novo := custo + a.custo(c.m)
		if visto[a.Para] != q.geracao || novo < dist[a.Para] {
			dist[a.Para] = novo
			outra[a.Para] = outra[no] + a.outroCusto(c.m)
			arcoDe[a.Para] = inicio + uint32(i)
			anterior[a.Para] = no
			visto[a.Para] = q.geracao
			h.push(a.Para, novo)
		}
		// Com o valor ja assentado deste lado, e nao com novo: se um caminho
		// melhor ate este no ja era conhecido, e ele que combina com o outro
		// lado. Usar novo daria um candidato pior que o real.
		if outroVisto[a.Para] == q.geracao && visto[a.Para] == q.geracao {
			if total := dist[a.Para] + outroLadoDist[a.Para]; total < *melhor {
				*melhor = total
				*encontro = a.Para
			}
		}
	}

	// O proprio no pode ser o cume: quando as duas buscas o alcancam, o
	// caminho que passa por ele e candidato mesmo sem nenhuma aresta nova.
	if outroVisto[no] == q.geracao {
		if total := custo + outroLadoDist[no]; total < *melhor {
			*melhor = total
			*encontro = no
		}
	}
}

// expandir troca um atalho pelos dois passos que ele resume, recursivamente.
//
// Escreve em out os nos estritamente entre u e w.
//
// Os dois pedacos de um atalho de u ate w com meio v estao guardados em v: o
// arco de v ate u com a bandeira de tras (que representa u indo para v) e o
// arco de v ate w com a bandeira de frente. E consequencia de v ter sido
// contraido antes dos dois -- tudo que sobe de v alcanca os dois lados.
func (c *CH) expandir(u, w, meio graph.NodeID, out *[]graph.NodeID) {
	if meio == graph.NoNode {
		return // aresta original: nao ha nada entre as pontas
	}

	primeiro, ok1 := c.acharArco(meio, u, false)
	segundo, ok2 := c.acharArco(meio, w, true)
	if !ok1 || !ok2 {
		// Nao deveria acontecer: todo atalho foi criado a partir de dois
		// arcos que existem. Se acontecer, e melhor devolver o caminho sem o
		// miolo do que entrar em panico no meio de uma consulta.
		return
	}

	c.expandir(u, meio, primeiro.Meio, out)
	*out = append(*out, meio)
	c.expandir(meio, w, segundo.Meio, out)
}

// acharArco procura, entre os arcos que sobem de v, o mais barato ate alvo no
// sentido pedido.
func (c *CH) acharArco(v, alvo graph.NodeID, frente bool) (arco, bool) {
	arcos, podeFrente, podeTras := c.subindo(v)

	var melhor arco
	achou := false
	for i, a := range arcos {
		if a.Para != alvo {
			continue
		}
		if frente && !podeFrente[i] {
			continue
		}
		if !frente && !podeTras[i] {
			continue
		}
		if !achou || a.custo(c.m) < melhor.custo(c.m) {
			melhor, achou = a, true
		}
	}
	return melhor, achou
}

func valido(c *CH, n graph.NodeID) bool { return n >= 0 && int(n) < len(c.posicao) }

func inverter(s []graph.NodeID) {
	for i, j := 0, len(s)-1; i < j; i, j = i+1, j-1 {
		s[i], s[j] = s[j], s[i]
	}
}
