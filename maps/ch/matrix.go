package ch

import (
	"math"
	"runtime"
	"sync"

	"github.com/iafrotamacedo-cloud/era/maps/graph"
)

// Matrix e o custo de cada par origem-destino.
//
// Infinito quer dizer que nao ha caminho. E um valor legitimo num mapa real:
// ilha sem ponte, condominio mapeado sem ligacao com a rua, pedaco de malha
// que ficou solto no recorte do extrato.
type Matrix struct {
	Origens  []graph.NodeID
	Destinos []graph.NodeID

	metrica graph.Metric

	custos []float64
	outras []float64 // a grandeza que a hierarquia nao minimiza
}

// Rows e Cols devolvem as dimensoes da matriz.
func (m *Matrix) Rows() int { return len(m.Origens) }
func (m *Matrix) Cols() int { return len(m.Destinos) }

// At devolve o custo de ir da origem i ate o destino j.
func (m *Matrix) At(i, j int) float64 { return m.custos[i*len(m.Destinos)+j] }

// Alcancavel informa se ha caminho entre o par.
func (m *Matrix) Alcancavel(i, j int) bool { return !math.IsInf(m.At(i, j), 1) }

// Leg devolve distancia e tempo do par, nesta ordem.
//
// As duas vem juntas mesmo que a hierarquia minimize so uma: quem pede a
// matriz mais rapida quase sempre tambem quer saber a quilometragem.
func (m *Matrix) Leg(i, j int) (metros, segundos float64) {
	k := i*len(m.Destinos) + j
	if m.metrica == graph.Time {
		return m.outras[k], m.custos[k]
	}
	return m.custos[k], m.outras[k]
}

// Matrix calcula o custo de todos os pares de uma vez.
//
// # Por que nao e so um laco de consultas
//
// Uma matriz 500 x 500 sao 250 mil pares. Feita como 250 mil consultas
// separadas, cada destino teria a sua metade da busca refeita 500 vezes.
//
// O algoritmo de baldes faz cada metade uma vez so. Primeiro cada destino
// sobe a hierarquia e vai deixando bilhetes pelo caminho: "daqui ate mim,
// tanto". Depois cada origem sobe, e a cada no que alcanca recolhe os
// bilhetes que estao la e combina com o que ja andou. O encontro das duas
// buscas deixa de ser procurado e passa a ser encontrado.
//
// O ganho e proporcional ao numero de destinos, e e a diferenca entre
// roteirizar um dia de entregas em segundos ou em minutos.
func (c *CH) Matrix(origens, destinos []graph.NodeID) *Matrix {
	m := &Matrix{
		Origens:  origens,
		Destinos: destinos,
		metrica:  c.m,
		custos:   make([]float64, len(origens)*len(destinos)),
		outras:   make([]float64, len(origens)*len(destinos)),
	}
	for i := range m.custos {
		m.custos[i] = math.Inf(1)
		m.outras[i] = math.Inf(1)
	}
	if len(origens) == 0 || len(destinos) == 0 {
		return m
	}

	// Fase 1: cada destino sobe e deixa bilhetes.
	//
	// Em serie, de proposito: os baldes sao escrita compartilhada, e um
	// mutex por no custaria mais do que esta fase inteira. Ela e a barata --
	// proporcional aos destinos, enquanto a fase 2 e proporcional as origens
	// vezes o tamanho de cada balde.
	type bilhete struct {
		destino int
		custo   float64
		outra   float64
	}
	baldes := make([][]bilhete, c.Len())

	q := c.NewQuery()
	for j, t := range destinos {
		if !valido(c, t) {
			continue
		}
		q.varrer(t, false, func(n graph.NodeID, custo, outra float64) {
			baldes[n] = append(baldes[n], bilhete{destino: j, custo: custo, outra: outra})
		})
	}

	// Fase 2: cada origem sobe e recolhe.
	trabalhadores := runtime.GOMAXPROCS(0)
	if trabalhadores > len(origens) {
		trabalhadores = len(origens)
	}

	linhas := make(chan int)
	var wg sync.WaitGroup
	for w := 0; w < trabalhadores; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			q := c.NewQuery()
			for i := range linhas {
				s := origens[i]
				if !valido(c, s) {
					continue
				}
				linha := m.custos[i*len(destinos) : (i+1)*len(destinos)]
				linhaOutra := m.outras[i*len(destinos) : (i+1)*len(destinos)]
				q.varrer(s, true, func(n graph.NodeID, custo, outra float64) {
					for _, b := range baldes[n] {
						if total := custo + b.custo; total < linha[b.destino] {
							linha[b.destino] = total
							linhaOutra[b.destino] = outra + b.outra
						}
					}
				})
			}
		}()
	}
	for i := range origens {
		linhas <- i
	}
	close(linhas)
	wg.Wait()

	return m
}

// varrer roda uma busca para cima a partir de um no e chama visitar para cada
// no fechado, com o custo definitivo dele.
//
// Nao ha poda por melhor caminho conhecido: numa matriz, qualquer no que as
// duas buscas alcancam pode ser o cume de algum par. Podar aqui perderia
// pares. As buscas para cima sao pequenas -- algumas centenas de nos num
// grafo de centenas de milhares --, entao roda-las inteiras sai barato.
func (q *Query) varrer(de graph.NodeID, frente bool, visitar func(graph.NodeID, float64, float64)) {
	c := q.c

	q.geracao++
	h, dist, visto, outra := &q.hb, q.db, q.vb, q.ob
	if frente {
		h, dist, visto, outra = &q.hf, q.df, q.vf, q.of
	}
	h.limpar()

	dist[de], visto[de], outra[de] = 0, q.geracao, 0
	h.push(de, 0)

	for h.Len() > 0 {
		no, custo := h.pop()
		if visto[no] != q.geracao || custo > dist[no] {
			continue
		}
		visitar(no, custo, outra[no])

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
				visto[a.Para] = q.geracao
				h.push(a.Para, novo)
			}
		}
	}
}
