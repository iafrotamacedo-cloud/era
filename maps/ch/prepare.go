package ch

import (
	"fmt"
	"math"
	"time"

	"github.com/iafrotamacedo-cloud/era/maps/geo"
	"github.com/iafrotamacedo-cloud/era/maps/graph"
)

// SaltosPadrao e o limite de saltos da busca de testemunha.
//
// Sem limite, procurar a testemunha de cada par de vizinhos de cada no vira um
// Dijkstra completo por par, e o pre-processamento de um grafo estadual nao
// termina em tempo util.
//
// Cinco e o valor que a literatura usa e que os testes deste pacote
// confirmam: acima disso o numero de atalhos quase nao cai e o preparo fica
// bem mais lento.
const SaltosPadrao = 5

// Options ajusta o pre-processamento.
type Options struct {
	// MaxSaltos limita a busca de testemunha. Zero usa SaltosPadrao.
	//
	// Menos saltos deixa o preparo mais rapido e cria mais atalhos; mais
	// saltos faz o contrario. Nenhum dos dois muda a rota que sai no fim --
	// so o tamanho da estrutura e o tempo de consulta.
	MaxSaltos int
}

// Stats conta o que o pre-processamento produziu.
type Stats struct {
	Nos              int
	ArestasOriginais int
	Atalhos          int
	ArcosSubindo     int
	Duracao          time.Duration
}

// Inchaco devolve quantos arcos existem para cada aresta original.
//
// E o preco da hierarquia: uma estrutura que nao inchasse nao aceleraria
// nada. Num grafo rodoviario real a faixa costuma ficar entre 1,2 e 2.
func (s Stats) Inchaco() float64 {
	if s.ArestasOriginais == 0 {
		return 0
	}
	return float64(s.ArestasOriginais+s.Atalhos) / float64(s.ArestasOriginais)
}

func (s Stats) String() string {
	return fmt.Sprintf("hierarquia: %d nos, %d arestas originais, %d atalhos (inchaco %.2fx), %d arcos subindo, em %s",
		s.Nos, s.ArestasOriginais, s.Atalhos, s.Inchaco(), s.ArcosSubindo, s.Duracao.Round(time.Millisecond))
}

// Stats devolve o que o preparo contou.
func (c *CH) Stats() Stats { return c.stats }

// arcoTrabalho e um arco durante a contracao, quando o grafo ainda muda.
type arcoTrabalho struct {
	vizinho  graph.NodeID
	metros   float64
	segundos float64
	meio     graph.NodeID
}

// Prepare constroi a hierarquia de um grafo.
//
// A metrica escolhida fica gravada: a ordem de contracao depende dos pesos, e
// a hierarquia mais curta em quilometros nao e a mais rapida em minutos. Quem
// precisar das duas prepara duas.
//
// E caro -- e o passo que se faz uma vez e se guarda com Save. Num grafo de
// dezenas de milhares de cruzamentos leva segundos; num estadual, minutos.
func Prepare(g *graph.Graph, m graph.Metric, opts ...Options) (*CH, error) {
	inicio := time.Now()

	var o Options
	if len(opts) > 0 {
		o = opts[0]
	}
	if o.MaxSaltos <= 0 {
		o.MaxSaltos = SaltosPadrao
	}
	if g == nil || g.Len() == 0 {
		return nil, fmt.Errorf("ch: grafo vazio")
	}

	p := novoPreparo(g, m, o.MaxSaltos)
	p.contrairTudo()

	c := p.montar()
	c.stats.Duracao = time.Since(inicio)
	return c, nil
}

// preparo e o estado mutavel da contracao.
type preparo struct {
	g         *graph.Graph
	m         graph.Metric
	maxSaltos int
	n         int

	// Listas de adjacencia mutaveis: a contracao acrescenta atalhos, e CSR
	// nao aceita insercao. O CSR so e montado no fim.
	saida   [][]arcoTrabalho
	entrada [][]arcoTrabalho

	contraido []bool
	posicao   []int32
	ordem     int32

	vizinhosContraidos []int32
	nivel              []int32

	// Estado da busca de testemunha, reaproveitado entre chamadas pela mesma
	// marca de geracao que o maps/graph usa.
	dist    []float64
	saltos  []int32
	visto   []uint32
	geracao uint32

	// Duas filas, e nao uma. A de ordem vive durante toda a contracao; a de
	// testemunha e limpa a cada busca. Compartilhar as duas apagaria a ordem
	// de contracao no meio do caminho.
	filaOrdem      filaCusto
	filaTestemunha filaCusto

	stats Stats
}

func novoPreparo(g *graph.Graph, m graph.Metric, maxSaltos int) *preparo {
	n := g.Len()
	p := &preparo{
		g: g, m: m, maxSaltos: maxSaltos, n: n,
		saida:              make([][]arcoTrabalho, n),
		entrada:            make([][]arcoTrabalho, n),
		contraido:          make([]bool, n),
		posicao:            make([]int32, n),
		vizinhosContraidos: make([]int32, n),
		nivel:              make([]int32, n),
		dist:               make([]float64, n),
		saltos:             make([]int32, n),
		visto:              make([]uint32, n),
	}

	// Copia o grafo para as listas mutaveis, deduplicando arestas paralelas.
	//
	// Duas ruas podem ligar o mesmo par de cruzamentos -- a avenida e a rua de
	// tras. Para a hierarquia so a melhor importa, e carregar as duas faria
	// cada contracao examinar pares que nunca vao render atalho.
	for v := graph.NodeID(0); int(v) < n; v++ {
		for _, e := range g.EdgesOf(v) {
			metros, segundos := float64(e.Meters), float64(e.Seconds)
			if e.Forward {
				p.saida[v] = juntar(p.saida[v], arcoTrabalho{
					vizinho: e.To, metros: metros, segundos: segundos, meio: graph.NoNode,
				}, m)
			}
			if e.Backward {
				p.entrada[v] = juntar(p.entrada[v], arcoTrabalho{
					vizinho: e.To, metros: metros, segundos: segundos, meio: graph.NoNode,
				}, m)
			}
		}
	}
	for v := 0; v < n; v++ {
		p.stats.ArestasOriginais += len(p.saida[v])
	}
	p.stats.Nos = n
	return p
}

// juntar acrescenta um arco a uma lista, guardando so o melhor por vizinho.
func juntar(lista []arcoTrabalho, novo arcoTrabalho, m graph.Metric) []arcoTrabalho {
	custo := func(a arcoTrabalho) float64 {
		if m == graph.Time {
			return a.segundos
		}
		return a.metros
	}
	for i := range lista {
		if lista[i].vizinho != novo.vizinho {
			continue
		}
		if custo(novo) < custo(lista[i]) {
			lista[i] = novo
		}
		return lista
	}
	return append(lista, novo)
}

func (p *preparo) custo(a arcoTrabalho) float64 {
	if p.m == graph.Time {
		return a.segundos
	}
	return a.metros
}

// contrairTudo percorre os nos do menos importante para o mais importante.
//
// A fila e preguicosa: a prioridade de um no depende de quantos vizinhos ja
// foram contraidos, entao ela envelhece o tempo todo. Recalcular a de todos a
// cada contracao seria caro demais. Em vez disso, tira-se o menor, recalcula-
// se so ele, e se ele deixou de ser o menor devolve-se para a fila.
func (p *preparo) contrairTudo() {
	p.filaOrdem.limpar()
	for v := 0; v < p.n; v++ {
		p.filaOrdem.push(graph.NodeID(v), p.prioridade(graph.NodeID(v)))
	}

	for p.filaOrdem.Len() > 0 {
		v, _ := p.filaOrdem.pop()
		novaPrioridade := p.prioridade(v)

		if p.filaOrdem.Len() > 0 && novaPrioridade > p.filaOrdem.topo() {
			p.filaOrdem.push(v, novaPrioridade)
			continue
		}
		p.contrair(v)
	}
}

// prioridade estima o custo de contrair um no agora.
//
// Tres termos, todos com o mesmo peso:
//
//   - diferenca de arestas: atalhos que seriam criados menos arestas que
//     sumiriam. Contrair primeiro quem nao gera atalho e o que mantem a
//     estrutura pequena.
//   - vizinhos ja contraidos: espalha a contracao pelo mapa em vez de deixa-la
//     corroer uma regiao so, o que produziria uma hierarquia torta.
//   - nivel: mantem a hierarquia rasa, para as buscas subirem pouco.
func (p *preparo) prioridade(v graph.NodeID) float64 {
	atalhos := p.simular(v, nil)
	grau := len(p.vivos(p.entrada[v])) + len(p.vivos(p.saida[v]))

	return float64(atalhos-grau) + float64(p.vizinhosContraidos[v]) + float64(p.nivel[v])
}

// contrair remove um no e cria os atalhos que a remocao exige.
func (p *preparo) contrair(v graph.NodeID) {
	p.simular(v, func(u, w graph.NodeID, metros, segundos float64) {
		p.adicionarAtalho(u, w, metros, segundos, v)
	})

	p.contraido[v] = true
	p.posicao[v] = p.ordem
	p.ordem++

	for _, a := range p.entrada[v] {
		if !p.contraido[a.vizinho] {
			p.vizinhosContraidos[a.vizinho]++
			p.nivel[a.vizinho] = max32(p.nivel[a.vizinho], p.nivel[v]+1)
		}
	}
	for _, a := range p.saida[v] {
		if !p.contraido[a.vizinho] {
			p.vizinhosContraidos[a.vizinho]++
			p.nivel[a.vizinho] = max32(p.nivel[a.vizinho], p.nivel[v]+1)
		}
	}
}

// simular percorre os pares de vizinhos de v e decide quais precisam de
// atalho.
//
// Com aplicar nil, so conta -- e o que a prioridade usa. Com aplicar dado,
// cria os atalhos.
func (p *preparo) simular(v graph.NodeID, aplicar func(u, w graph.NodeID, metros, segundos float64)) int {
	entradas := p.vivos(p.entrada[v])
	saidas := p.vivos(p.saida[v])
	if len(entradas) == 0 || len(saidas) == 0 {
		return 0
	}

	atalhos := 0
	for _, ea := range entradas {
		u := ea.vizinho

		// O limite da busca de testemunha e o pior caminho por v a partir
		// deste u. Alem dele nao ha nada que interesse, e cortar cedo e o que
		// torna a contracao viavel.
		limite := 0.0
		for _, sa := range saidas {
			if sa.vizinho == u {
				continue
			}
			if c := p.custo(ea) + p.custo(sa); c > limite {
				limite = c
			}
		}
		if limite == 0 {
			continue
		}

		p.testemunhas(u, v, limite)

		for _, sa := range saidas {
			w := sa.vizinho
			if w == u {
				continue
			}
			direto := p.custo(ea) + p.custo(sa)

			// Sem epsilon, e de proposito. Aceitar uma testemunha que seja um
			// fio mais cara por erro de ponto flutuante deixaria de criar um
			// atalho necessario, e ai a hierarquia responde errado. Recusar
			// uma testemunha valida so cria um atalho a mais.
			if p.visto[w] == p.geracao && p.dist[w] <= direto {
				continue
			}

			atalhos++
			if aplicar != nil {
				aplicar(u, w, ea.metros+sa.metros, ea.segundos+sa.segundos)
			}
		}
	}
	return atalhos
}

// testemunhas roda um Dijkstra curto a partir de u, ignorando v.
//
// Escreve as distancias em p.dist, validas onde p.visto marca a geracao
// atual. Para ao passar do limite de custo ou do limite de saltos.
func (p *preparo) testemunhas(u, ignorar graph.NodeID, limite float64) {
	p.geracao++
	p.filaTestemunha.limpar()

	p.dist[u], p.saltos[u], p.visto[u] = 0, 0, p.geracao
	p.filaTestemunha.push(u, 0)

	for p.filaTestemunha.Len() > 0 {
		no, custo := p.filaTestemunha.pop()
		if p.visto[no] != p.geracao || custo > p.dist[no] {
			continue
		}
		if custo > limite {
			return // dai para frente nada mais serve de testemunha
		}
		if int(p.saltos[no]) >= p.maxSaltos {
			continue
		}

		for _, a := range p.saida[no] {
			w := a.vizinho
			if w == ignorar || p.contraido[w] {
				continue
			}
			novo := custo + p.custo(a)
			if novo > limite {
				continue
			}
			if p.visto[w] != p.geracao || novo < p.dist[w] {
				p.dist[w] = novo
				p.saltos[w] = p.saltos[no] + 1
				p.visto[w] = p.geracao
				p.filaTestemunha.push(w, novo)
			}
		}
	}
}

// adicionarAtalho cria ou melhora o arco de u ate w.
func (p *preparo) adicionarAtalho(u, w graph.NodeID, metros, segundos float64, meio graph.NodeID) {
	novo := arcoTrabalho{vizinho: w, metros: metros, segundos: segundos, meio: meio}
	antes := len(p.saida[u])
	p.saida[u] = juntar(p.saida[u], novo, p.m)
	if len(p.saida[u]) > antes {
		p.stats.Atalhos++
	}

	p.entrada[w] = juntar(p.entrada[w], arcoTrabalho{
		vizinho: u, metros: metros, segundos: segundos, meio: meio,
	}, p.m)
}

// vivos filtra os vizinhos que ainda nao foram contraidos.
func (p *preparo) vivos(lista []arcoTrabalho) []arcoTrabalho {
	fora := 0
	for _, a := range lista {
		if p.contraido[a.vizinho] {
			fora++
		}
	}
	if fora == 0 {
		return lista
	}
	out := make([]arcoTrabalho, 0, len(lista)-fora)
	for _, a := range lista {
		if !p.contraido[a.vizinho] {
			out = append(out, a)
		}
	}
	return out
}

// montar transforma as listas mutaveis no CSR de arcos que sobem.
//
// Um arco so entra se leva a um no mais importante. E o que a consulta
// precisa, e descartar o resto corta a estrutura quase pela metade.
func (p *preparo) montar() *CH {
	c := &CH{
		m:       p.m,
		pontos:  make([]geo.Point, p.n),
		osmIDs:  make([]int64, p.n),
		posicao: p.posicao,
		offsets: make([]uint32, p.n+1),
		stats:   p.stats,
	}
	for v := graph.NodeID(0); int(v) < p.n; v++ {
		c.pontos[v] = p.g.Point(v)
		c.osmIDs[v] = p.g.OSMID(v)
	}

	type pendente struct {
		a            arco
		frente, tras bool
	}
	porNo := make([][]pendente, p.n)

	juntarPendente := func(v graph.NodeID, a arco, frente, tras bool) {
		for i := range porNo[v] {
			q := &porNo[v][i]
			if q.a.Para == a.Para && q.a.Metros == a.Metros && q.a.Segundos == a.Segundos && q.a.Meio == a.Meio {
				q.frente = q.frente || frente
				q.tras = q.tras || tras
				return
			}
		}
		porNo[v] = append(porNo[v], pendente{a: a, frente: frente, tras: tras})
	}

	for v := graph.NodeID(0); int(v) < p.n; v++ {
		for _, a := range p.saida[v] {
			if p.posicao[a.vizinho] <= p.posicao[v] {
				continue
			}
			juntarPendente(v, arco{
				Para: a.vizinho, Metros: float32(a.metros),
				Segundos: float32(a.segundos), Meio: a.meio,
			}, true, false)
		}
		for _, a := range p.entrada[v] {
			if p.posicao[a.vizinho] <= p.posicao[v] {
				continue
			}
			juntarPendente(v, arco{
				Para: a.vizinho, Metros: float32(a.metros),
				Segundos: float32(a.segundos), Meio: a.meio,
			}, false, true)
		}
	}

	total := 0
	for v := 0; v < p.n; v++ {
		c.offsets[v] = uint32(total)
		total += len(porNo[v])
	}
	c.offsets[p.n] = uint32(total)

	c.arcos = make([]arco, 0, total)
	c.frente = make([]bool, 0, total)
	c.tras = make([]bool, 0, total)
	for v := 0; v < p.n; v++ {
		for _, q := range porNo[v] {
			c.arcos = append(c.arcos, q.a)
			c.frente = append(c.frente, q.frente)
			c.tras = append(c.tras, q.tras)
		}
	}

	c.stats.ArcosSubindo = len(c.arcos)
	return c
}

func max32(a, b int32) int32 {
	if a > b {
		return a
	}
	return b
}

// filaCusto e o mesmo heap binario concreto do maps/graph.
type filaCusto struct {
	nos    []graph.NodeID
	custos []float64
}

func (f *filaCusto) Len() int { return len(f.nos) }

func (f *filaCusto) limpar() {
	f.nos = f.nos[:0]
	f.custos = f.custos[:0]
}

func (f *filaCusto) topo() float64 {
	if len(f.custos) == 0 {
		return math.Inf(1)
	}
	return f.custos[0]
}

func (f *filaCusto) push(n graph.NodeID, custo float64) {
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

func (f *filaCusto) pop() (graph.NodeID, float64) {
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

func (f *filaCusto) trocar(i, j int) {
	f.nos[i], f.nos[j] = f.nos[j], f.nos[i]
	f.custos[i], f.custos[j] = f.custos[j], f.custos[i]
}
