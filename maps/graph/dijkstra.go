package graph

import "math"

// Path e um caminho encontrado.
//
// Nodes traz os cruzamentos na ordem em que se passa por eles. Meters e
// Seconds vem sempre os dois, mesmo que a busca tenha minimizado so um: quem
// pede a rota mais rapida quase sempre tambem quer saber quantos quilometros
// ela tem.
type Path struct {
	Nodes   []NodeID
	Meters  float64
	Seconds float64
}

// Route acha o caminho mais curto entre dois cruzamentos.
//
// Aloca as estruturas de busca a cada chamada. Para muitas consultas seguidas
// use NewSearcher, que as reaproveita.
func (g *Graph) Route(de, para NodeID, m Metric) (Path, bool) {
	return g.NewSearcher().Route(de, para, m)
}

// Searcher guarda as estruturas de uma busca para reaproveita-las.
//
// Um Dijkstra precisa de seis vetores do tamanho do grafo -- distancia,
// antecessor e marca de visita, vezes dois sentidos. Aloca-los por consulta
// custa proporcional ao grafo; a busca em si custa proporcional ao tamanho da
// consulta. Quando os dois sao grandes, a alocacao some no meio; quando a
// consulta e curta, ela e quase tudo.
//
// Medido numa grade de 90 mil cruzamentos:
//
//	rota atravessando tudo    5,6 ms contra 5,9 ms   -- 6%
//	rota de dez quarteiroes   9,6 us contra 267 us   -- 28 vezes
//
// E a segunda linha que importa, porque e a consulta que uma roteirizacao faz
// aos milhares.
//
// Nao e seguro para uso concorrente: use um por goroutine.
type Searcher struct {
	g *Graph

	// Distancia e antecessor de cada sentido.
	df, db []float64
	pf, pb []NodeID

	// Marca de geracao no lugar de limpar os vetores.
	//
	// Zerar dezenas de MB a cada consulta custaria mais que a propria busca,
	// que costuma tocar uma fracao minuscula do grafo. Guardando em que
	// consulta cada posicao foi escrita, o lixo da consulta anterior fica
	// visivelmente velho e nao precisa ser apagado.
	vf, vb  []uint32
	geracao uint32

	hf, hb fila
}

// NewSearcher cria um buscador reutilizavel para este grafo.
func (g *Graph) NewSearcher() *Searcher {
	n := g.Len()
	return &Searcher{
		g:  g,
		df: make([]float64, n), db: make([]float64, n),
		pf: make([]NodeID, n), pb: make([]NodeID, n),
		vf: make([]uint32, n), vb: make([]uint32, n),
	}
}

// Route acha o caminho mais curto, buscando dos dois lados ao mesmo tempo.
//
// # Por que dois lados
//
// Um Dijkstra comum explora um circulo em volta da origem ate alcancar o
// destino: a area cresce com o quadrado da distancia. Buscando tambem a
// partir do destino, sao dois circulos de metade do raio -- e dois circulos
// de raio r/2 tem metade da area de um de raio r. Na pratica, num grafo
// rodoviario, isso corta a busca por um fator entre dois e quatro.
//
// # O criterio de parada
//
// E a parte que engana. Quando as duas buscas se encontram num no, o caminho
// que passa por ele nao e necessariamente o melhor: pode existir outro,
// passando por um no que nenhuma das duas chegou a fechar, que seja mais
// curto.
//
// Entao o encontro nao para a busca -- so registra um candidato. A parada vem
// de outra conta: enquanto a soma do menor custo pendente de cada lado for
// menor que o melhor candidato, ainda pode aparecer coisa melhor. Quando essa
// soma alcanca o candidato, nao pode mais, e ai sim acabou.
//
// O candidato tambem e atualizado ao relaxar arestas, e nao so ao fechar nos.
// Sem isso, um caminho cujo ponto de encontro nunca chega a ser fechado pelos
// dois lados passaria despercebido.
func (s *Searcher) Route(de, para NodeID, m Metric) (Path, bool) {
	g := s.g
	if !g.valido(de) || !g.valido(para) {
		return Path{}, false
	}
	if de == para {
		return Path{Nodes: []NodeID{de}}, true
	}

	s.geracao++
	s.hf.limpar()
	s.hb.limpar()

	s.definir(de, 0, NoNode, true)
	s.definir(para, 0, NoNode, false)
	s.hf.push(de, 0)
	s.hb.push(para, 0)

	melhor := math.Inf(1)
	encontro := NoNode

	for s.hf.Len() > 0 && s.hb.Len() > 0 {
		if s.hf.topo()+s.hb.topo() >= melhor {
			break
		}
		// Avanca sempre o lado que esta mais atrasado. Mantem as duas frentes
		// no mesmo raio, que e o que faz a area total ser a menor possivel.
		if s.hf.topo() <= s.hb.topo() {
			s.expandir(m, true, &melhor, &encontro)
		} else {
			s.expandir(m, false, &melhor, &encontro)
		}
	}

	if encontro == NoNode {
		return Path{}, false
	}
	return g.medir(s.costurar(encontro), m), true
}

// expandir fecha o proximo no de um dos lados e relaxa as arestas dele.
func (s *Searcher) expandir(m Metric, frente bool, melhor *float64, encontro *NodeID) {
	h, dist, visto := &s.hb, s.db, s.vb
	outraDist, outroVisto := s.df, s.vf
	if frente {
		h, dist, visto = &s.hf, s.df, s.vf
		outraDist, outroVisto = s.db, s.vb
	}

	no, custo := h.pop()
	if visto[no] != s.geracao || custo > dist[no] {
		return // entrada velha: o no ja foi alcancado por caminho melhor
	}

	for _, e := range s.g.EdgesOf(no) {
		// A busca para tras anda pelas arestas que chegam, nao pelas que
		// saem. E para isso que cada trecho esta guardado nas duas pontas.
		if frente && !e.Forward {
			continue
		}
		if !frente && !e.Backward {
			continue
		}

		novo := custo + e.custo(m)
		if outroVisto[e.To] == s.geracao {
			if total := novo + outraDist[e.To]; total < *melhor {
				*melhor = total
				*encontro = e.To
			}
		}
		if visto[e.To] != s.geracao || novo < dist[e.To] {
			s.definir(e.To, novo, no, frente)
			h.push(e.To, novo)
		}
	}
}

func (s *Searcher) definir(n NodeID, custo float64, anterior NodeID, frente bool) {
	if frente {
		s.df[n], s.pf[n], s.vf[n] = custo, anterior, s.geracao
	} else {
		s.db[n], s.pb[n], s.vb[n] = custo, anterior, s.geracao
	}
}

// costurar junta as duas metades no ponto de encontro.
func (s *Searcher) costurar(encontro NodeID) []NodeID {
	var nos []NodeID
	for n := encontro; n != NoNode; n = s.pf[n] {
		nos = append(nos, n)
	}
	for i, j := 0, len(nos)-1; i < j; i, j = i+1, j-1 {
		nos[i], nos[j] = nos[j], nos[i]
	}
	for n := s.pb[encontro]; n != NoNode; n = s.pb[n] {
		nos = append(nos, n)
	}
	return nos
}

// fila e um heap binario de pares no-custo.
//
// Existe em vez do container/heap porque aquele passa por interface e aloca a
// cada Push. Numa busca que empurra centenas de milhares de entradas, isso
// aparece. A versao com container/heap esta em reference.go, e os testes
// comparam as duas.
//
// Entradas velhas nao sao removidas quando um no e alcancado por caminho
// melhor: sao deixadas na fila e descartadas ao sair. Remover custaria achar
// a posicao, e a fila fica no maximo com uma entrada por relaxamento.
type fila struct {
	nos    []NodeID
	custos []float64
}

func (f *fila) Len() int { return len(f.nos) }

func (f *fila) limpar() {
	f.nos = f.nos[:0]
	f.custos = f.custos[:0]
}

func (f *fila) topo() float64 {
	if len(f.custos) == 0 {
		return math.Inf(1)
	}
	return f.custos[0]
}

func (f *fila) push(n NodeID, custo float64) {
	f.nos = append(f.nos, n)
	f.custos = append(f.custos, custo)

	i := len(f.nos) - 1
	for i > 0 {
		pai := (i - 1) / 2
		if f.custos[pai] <= f.custos[i] {
			break
		}
		f.trocar(pai, i)
		i = pai
	}
}

func (f *fila) pop() (NodeID, float64) {
	no, custo := f.nos[0], f.custos[0]

	ultimo := len(f.nos) - 1
	f.nos[0], f.custos[0] = f.nos[ultimo], f.custos[ultimo]
	f.nos = f.nos[:ultimo]
	f.custos = f.custos[:ultimo]

	i := 0
	for {
		esq, dir := 2*i+1, 2*i+2
		menor := i
		if esq < len(f.nos) && f.custos[esq] < f.custos[menor] {
			menor = esq
		}
		if dir < len(f.nos) && f.custos[dir] < f.custos[menor] {
			menor = dir
		}
		if menor == i {
			break
		}
		f.trocar(i, menor)
		i = menor
	}
	return no, custo
}

func (f *fila) trocar(i, j int) {
	f.nos[i], f.nos[j] = f.nos[j], f.nos[i]
	f.custos[i], f.custos[j] = f.custos[j], f.custos[i]
}
