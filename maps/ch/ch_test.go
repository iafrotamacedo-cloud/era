package ch

import (
	"math"
	"math/rand"
	"testing"

	"github.com/iafrotamacedo-cloud/era/maps/geo"
	"github.com/iafrotamacedo-cloud/era/maps/graph"
)

// O maps/graph nao expoe um construtor de grafo a mao -- ele monta a partir
// de .osm.pbf. Para testar a hierarquia sem depender do formato, os testes
// escrevem um .osm.pbf minusculo, como o proprio maps/graph faz.
//
// A alternativa seria o graph exportar um construtor so para teste, e isso
// abriria na API publica um caminho que producao nenhuma deveria usar.

func grafoDeGrade(t testing.TB, lado int, maoUnica bool) *graph.Graph {
	t.Helper()

	var nos []noPbf
	var vias []viaPbf
	id := func(i, j int) int64 { return int64(i*lado + j + 1) }

	for i := 0; i < lado; i++ {
		for j := 0; j < lado; j++ {
			nos = append(nos, noPbf{
				id: id(i, j),
				p: geo.Point{
					Lat: -3.7 - float64(i)*0.0009,
					Lon: -38.5 + float64(j)*0.0009,
				},
			})
		}
	}

	via := 1
	for i := 0; i < lado; i++ {
		for j := 0; j < lado; j++ {
			if j+1 < lado {
				tags := map[string]string{"highway": "residential"}
				// Mao unica alternada: e o que separa um grafo direcionado de
				// um nao direcionado, e onde a hierarquia erra se os arcos que
				// sobem forem marcados no sentido trocado.
				if maoUnica && (i+j)%3 == 0 {
					tags["oneway"] = "yes"
				}
				vias = append(vias, viaPbf{id: int64(via), refs: []int64{id(i, j), id(i, j+1)}, tags: tags})
				via++
			}
			if i+1 < lado {
				tags := map[string]string{"highway": "residential"}
				if maoUnica && (i*j)%5 == 0 {
					tags["oneway"] = "yes"
				}
				vias = append(vias, viaPbf{id: int64(via), refs: []int64{id(i, j), id(i+1, j)}, tags: tags})
				via++
			}
		}
	}

	return montarGrafo(t, nos, vias)
}

// grafoAleatorio monta uma malha irregular: ruas de comprimentos e
// velocidades variados, com mao unica sorteada.
func grafoAleatorio(t testing.TB, r *rand.Rand, nNos, nVias int) *graph.Graph {
	t.Helper()

	var nos []noPbf
	for i := 0; i < nNos; i++ {
		nos = append(nos, noPbf{
			id: int64(i + 1),
			p: geo.Point{
				Lat: -3.7 - r.Float64()*0.05,
				Lon: -38.5 - r.Float64()*0.05,
			},
		})
	}

	tipos := []string{"residential", "secondary", "primary", "trunk", "motorway_link"}

	var vias []viaPbf
	for i := 0; i < nVias; i++ {
		a, b := r.Intn(nNos), r.Intn(nNos)
		if a == b {
			continue
		}
		tags := map[string]string{"highway": tipos[r.Intn(len(tipos))]}
		switch r.Intn(5) {
		case 0:
			tags["oneway"] = "yes"
		case 1:
			tags["oneway"] = "-1"
		}
		vias = append(vias, viaPbf{
			id:   int64(i + 1),
			refs: []int64{int64(a + 1), int64(b + 1)},
			tags: tags,
		})
	}
	return montarGrafo(t, nos, vias)
}

// O teste que decide se a fase existe.
//
// A hierarquia tem de devolver exatamente o mesmo custo que o Dijkstra da
// Fase 4 -- que por sua vez ja e conferido contra o Dijkstra obvio. A cadeia
// de verificacao vai do algoritmo esperto ate um que da para ler e afirmar
// que esta certo.
func TestHierarquiaBateComDijkstra(t *testing.T) {
	r := rand.New(rand.NewSource(1))

	for caso := 0; caso < 25; caso++ {
		nNos := 20 + r.Intn(60)
		g := grafoAleatorio(t, r, nNos, nNos*3)
		if g.Len() < 4 {
			continue
		}

		for _, m := range []graph.Metric{graph.Distance, graph.Time} {
			c, err := Prepare(g, m)
			if err != nil {
				t.Fatal(err)
			}
			q := c.NewQuery()

			for de := graph.NodeID(0); int(de) < g.Len(); de++ {
				for para := graph.NodeID(0); int(para) < g.Len(); para++ {
					querido, okRef := g.Route(de, para, m)
					custo, ok := q.Cost(de, para)

					if ok != okRef {
						t.Fatalf("caso %d, %v, %d->%d: ch achou=%v, dijkstra achou=%v",
							caso, m, de, para, ok, okRef)
					}
					if !ok {
						continue
					}

					esperado := querido.Meters
					if m == graph.Time {
						esperado = querido.Seconds
					}
					if math.Abs(custo-esperado) > 1e-4*math.Max(1, esperado) {
						t.Fatalf("caso %d, %v, %d->%d: ch %.6f, dijkstra %.6f",
							caso, m, de, para, custo, esperado)
					}
				}
			}
		}
	}
}

// Numa grade com mao unica alternada o grafo e bem direcionado, e a maioria
// dos pares tem ida diferente da volta.
func TestHierarquiaEmGradeComMaoUnica(t *testing.T) {
	g := grafoDeGrade(t, 9, true)
	c, err := Prepare(g, graph.Distance)
	if err != nil {
		t.Fatal(err)
	}
	t.Log(c.Stats())

	q := c.NewQuery()
	assimetricos := 0

	for de := graph.NodeID(0); int(de) < g.Len(); de++ {
		for para := graph.NodeID(0); int(para) < g.Len(); para++ {
			querido, okRef := g.Route(de, para, graph.Distance)
			custo, ok := q.Cost(de, para)
			if ok != okRef {
				t.Fatalf("%d->%d: ch achou=%v, dijkstra achou=%v", de, para, ok, okRef)
			}
			if ok && math.Abs(custo-querido.Meters) > 1e-4*math.Max(1, querido.Meters) {
				t.Fatalf("%d->%d: ch %.4f, dijkstra %.4f", de, para, custo, querido.Meters)
			}

			if de < para {
				volta, okVolta := g.Route(para, de, graph.Distance)
				if okRef != okVolta || (okRef && math.Abs(querido.Meters-volta.Meters) > 1) {
					assimetricos++
				}
			}
		}
	}

	if assimetricos == 0 {
		t.Error("nenhum par assimetrico: a mao unica nao esta sendo exercitada")
	}
	t.Logf("%d pares com ida diferente da volta", assimetricos)
}

// O caminho desempacotado tem de ser feito de ruas de verdade: cada par
// consecutivo precisa existir no grafo original, no sentido certo.
//
// E onde um erro de desempacotamento aparece. O custo pode estar certo e o
// caminho ser impossivel, e ai a rota some no mapa do motorista.
func TestCaminhoDesempacotadoEhPercorrivel(t *testing.T) {
	r := rand.New(rand.NewSource(2))
	g := grafoAleatorio(t, r, 60, 200)

	c, err := Prepare(g, graph.Distance)
	if err != nil {
		t.Fatal(err)
	}
	q := c.NewQuery()

	conferidos := 0
	for de := graph.NodeID(0); int(de) < g.Len(); de++ {
		for para := graph.NodeID(0); int(para) < g.Len(); para++ {
			p, ok := q.Route(de, para)
			if !ok {
				continue
			}
			conferidos++

			if p.Nodes[0] != de || p.Nodes[len(p.Nodes)-1] != para {
				t.Fatalf("%d->%d: caminho vai de %d a %d", de, para,
					p.Nodes[0], p.Nodes[len(p.Nodes)-1])
			}

			var soma float64
			for i := 1; i < len(p.Nodes); i++ {
				melhor := math.Inf(1)
				for _, e := range g.EdgesOf(p.Nodes[i-1]) {
					if e.To == p.Nodes[i] && e.Forward {
						melhor = math.Min(melhor, float64(e.Meters))
					}
				}
				if math.IsInf(melhor, 1) {
					t.Fatalf("%d->%d: nao existe rua de %d para %d",
						de, para, p.Nodes[i-1], p.Nodes[i])
				}
				soma += melhor
			}

			querido, _ := g.Route(de, para, graph.Distance)
			if math.Abs(soma-querido.Meters) > 1e-3*math.Max(1, querido.Meters) {
				t.Fatalf("%d->%d: o caminho soma %.4f, o otimo e %.4f", de, para, soma, querido.Meters)
			}
		}
	}
	if conferidos < 100 {
		t.Errorf("so %d caminhos conferidos; o grafo de teste esta desconexo demais", conferidos)
	}
}

// O limite de saltos da testemunha muda quantos atalhos nascem, mas nao pode
// mudar nenhuma resposta. E a propriedade que torna o algoritmo viavel: o
// erro da busca incompleta so tem o lado barato.
func TestLimiteDeSaltosNaoMudaResposta(t *testing.T) {
	r := rand.New(rand.NewSource(3))
	g := grafoAleatorio(t, r, 50, 160)

	base, err := Prepare(g, graph.Distance, Options{MaxSaltos: 1})
	if err != nil {
		t.Fatal(err)
	}
	fundo, err := Prepare(g, graph.Distance, Options{MaxSaltos: 12})
	if err != nil {
		t.Fatal(err)
	}

	t.Logf("1 salto:  %s", base.Stats())
	t.Logf("12 saltos: %s", fundo.Stats())

	if base.Stats().Atalhos <= fundo.Stats().Atalhos {
		t.Errorf("procurar menos testemunhas deveria criar mais atalhos: %d contra %d",
			base.Stats().Atalhos, fundo.Stats().Atalhos)
	}

	qb, qf := base.NewQuery(), fundo.NewQuery()
	for de := graph.NodeID(0); int(de) < g.Len(); de++ {
		for para := graph.NodeID(0); int(para) < g.Len(); para++ {
			cb, okb := qb.Cost(de, para)
			cf, okf := qf.Cost(de, para)
			if okb != okf {
				t.Fatalf("%d->%d: 1 salto achou=%v, 12 saltos achou=%v", de, para, okb, okf)
			}
			if okb && math.Abs(cb-cf) > 1e-6*math.Max(1, cb) {
				t.Fatalf("%d->%d: %.6f com 1 salto, %.6f com 12", de, para, cb, cf)
			}
		}
	}
}

// A hierarquia so serve a metrica com que foi construida.
func TestMetricaFicaGravada(t *testing.T) {
	g := grafoDeGrade(t, 5, false)

	porDistancia, err := Prepare(g, graph.Distance)
	if err != nil {
		t.Fatal(err)
	}
	porTempo, err := Prepare(g, graph.Time)
	if err != nil {
		t.Fatal(err)
	}

	if porDistancia.Metric() != graph.Distance || porTempo.Metric() != graph.Time {
		t.Error("a metrica nao ficou gravada")
	}

	qd, qt := porDistancia.NewQuery(), porTempo.NewQuery()
	de, para := graph.NodeID(0), graph.NodeID(g.Len()-1)

	cd, _ := qd.Cost(de, para)
	ct, _ := qt.Cost(de, para)

	rd, _ := g.Route(de, para, graph.Distance)
	rt, _ := g.Route(de, para, graph.Time)

	if math.Abs(cd-rd.Meters) > 1e-4 {
		t.Errorf("por distancia: %.4f, esperado %.4f", cd, rd.Meters)
	}
	if math.Abs(ct-rt.Seconds) > 1e-4 {
		t.Errorf("por tempo: %.4f, esperado %.4f", ct, rt.Seconds)
	}
}

func TestCasosDegenerados(t *testing.T) {
	g := grafoDeGrade(t, 4, false)
	c, err := Prepare(g, graph.Distance)
	if err != nil {
		t.Fatal(err)
	}
	q := c.NewQuery()

	if custo, ok := q.Cost(0, 0); !ok || custo != 0 {
		t.Errorf("no para ele mesmo = %v, %v", custo, ok)
	}
	if p, ok := q.Route(0, 0); !ok || len(p.Nodes) != 1 {
		t.Errorf("caminho de um no so = %v, %v", p, ok)
	}

	for _, n := range []graph.NodeID{-1, graph.NodeID(g.Len()), 9999} {
		if _, ok := q.Cost(0, n); ok {
			t.Errorf("no %d deveria ser recusado", n)
		}
		if _, ok := q.Route(n, 0); ok {
			t.Errorf("no %d como origem deveria ser recusado", n)
		}
	}

	if _, err := Prepare(nil, graph.Distance); err == nil {
		t.Error("grafo nil deveria ser recusado")
	}
}

// Todo no tem de receber uma posicao, e as posicoes tem de ser uma permutacao
// de 0 a n-1. Se duas fossem iguais, a regra de "so sobe" nao teria sentido.
func TestPosicoesSaoUmaPermutacao(t *testing.T) {
	r := rand.New(rand.NewSource(4))
	g := grafoAleatorio(t, r, 80, 240)

	c, err := Prepare(g, graph.Distance)
	if err != nil {
		t.Fatal(err)
	}

	vistas := make([]bool, c.Len())
	for n := graph.NodeID(0); int(n) < c.Len(); n++ {
		p := c.Rank(n)
		if p < 0 || int(p) >= c.Len() {
			t.Fatalf("no %d tem posicao %d, fora de 0..%d", n, p, c.Len()-1)
		}
		if vistas[p] {
			t.Fatalf("posicao %d atribuida duas vezes", p)
		}
		vistas[p] = true
	}
}

// Todo arco guardado tem de subir. Um arco que desce nunca seria usado pela
// busca, e estar la seria memoria gasta a toa.
func TestTodoArcoSobe(t *testing.T) {
	r := rand.New(rand.NewSource(5))
	g := grafoAleatorio(t, r, 60, 200)

	c, err := Prepare(g, graph.Distance)
	if err != nil {
		t.Fatal(err)
	}

	for v := graph.NodeID(0); int(v) < c.Len(); v++ {
		arcos, _, _ := c.subindo(v)
		for _, a := range arcos {
			if c.Rank(a.Para) <= c.Rank(v) {
				t.Fatalf("arco de %d (posicao %d) para %d (posicao %d) nao sobe",
					v, c.Rank(v), a.Para, c.Rank(a.Para))
			}
		}
	}
}
