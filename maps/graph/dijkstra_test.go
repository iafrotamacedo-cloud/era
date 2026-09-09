package graph

import (
	"math"
	"math/rand"
	"testing"

	"github.com/iafrotamacedo-cloud/era/maps/geo"
)

// grafoAleatorio monta um grafo direto, sem passar pelo .osm.pbf.
//
// Sorteia mao unica em um quarto dos trechos: e o caso que separa um grafo
// direcionado de um nao direcionado, e onde a busca para tras erra se as
// bandeiras de sentido estiverem trocadas de lado.
func grafoAleatorio(r *rand.Rand, nNos, nTrechos int) *Graph {
	g := &Graph{}
	for i := 0; i < nNos; i++ {
		g.pontos = append(g.pontos, geo.Point{
			Lat: -3.7 - r.Float64()*0.5,
			Lon: -38.5 - r.Float64()*0.5,
		})
		g.osmIDs = append(g.osmIDs, int64(i))
	}

	var ts []trecho
	for i := 0; i < nTrechos; i++ {
		de, para := NodeID(r.Intn(nNos)), NodeID(r.Intn(nNos))
		if de == para {
			continue
		}
		metros := 1 + r.Float64()*1000
		// A velocidade varia por trecho, senao tempo e distancia seriam a
		// mesma ordem e a metrica Time nao testaria nada.
		kmh := 20 + r.Float64()*80

		frente, tras := true, true
		switch r.Intn(4) {
		case 0:
			tras = false
		case 1:
			frente = false
		}

		ts = append(ts, trecho{
			de: de, para: para,
			metros:   float32(metros),
			segundos: float32(metros / (kmh / 3.6)),
			frente:   frente, tras: tras,
		})
	}
	g.montarCSR(ts)
	return g
}

func custoDe(p Path, m Metric) float64 {
	if m == Time {
		return p.Seconds
	}
	return p.Meters
}

// O teste central da fase: o Dijkstra bidirecional tem de dar exatamente o
// mesmo custo que o Dijkstra obvio, em todo par de todo grafo sorteado.
//
// Custo, e nao caminho: quando ha empate, os dois podem escolher caminhos
// diferentes e ambos estarem certos.
func TestBidirecionalBateComAReferencia(t *testing.T) {
	r := rand.New(rand.NewSource(1))

	for caso := 0; caso < 200; caso++ {
		nNos := 5 + r.Intn(60)
		g := grafoAleatorio(r, nNos, nNos*2)
		s := g.NewSearcher()

		for _, m := range []Metric{Distance, Time} {
			for de := NodeID(0); int(de) < nNos; de++ {
				for para := NodeID(0); int(para) < nNos; para++ {
					querido, okRef := g.RouteRef(de, para, m)
					obtido, ok := s.Route(de, para, m)

					if ok != okRef {
						t.Fatalf("caso %d, %v, %d->%d: bidirecional achou=%v, referencia achou=%v",
							caso, m, de, para, ok, okRef)
					}
					if !ok {
						continue
					}
					if math.Abs(custoDe(obtido, m)-custoDe(querido, m)) > 1e-6 {
						t.Fatalf("caso %d, %v, %d->%d: custo %.6f, referencia %.6f",
							caso, m, de, para, custoDe(obtido, m), custoDe(querido, m))
					}
				}
			}
		}
	}
}

// O caminho devolvido tem de existir de verdade: cada par consecutivo precisa
// ter uma aresta no sentido certo. Um custo certo com um caminho impossivel
// seria pior que um erro, porque parece bom.
func TestCaminhoDevolvidoEhPercorrivel(t *testing.T) {
	r := rand.New(rand.NewSource(2))

	for caso := 0; caso < 100; caso++ {
		nNos := 10 + r.Intn(40)
		g := grafoAleatorio(r, nNos, nNos*2)
		s := g.NewSearcher()

		for i := 0; i < 50; i++ {
			de, para := NodeID(r.Intn(nNos)), NodeID(r.Intn(nNos))
			p, ok := s.Route(de, para, Distance)
			if !ok {
				continue
			}

			if p.Nodes[0] != de || p.Nodes[len(p.Nodes)-1] != para {
				t.Fatalf("caminho vai de %d a %d, pedi de %d a %d",
					p.Nodes[0], p.Nodes[len(p.Nodes)-1], de, para)
			}

			var soma float64
			for j := 1; j < len(p.Nodes); j++ {
				achou := false
				melhor := math.Inf(1)
				for _, e := range g.EdgesOf(p.Nodes[j-1]) {
					if e.To == p.Nodes[j] && e.Forward {
						achou = true
						melhor = math.Min(melhor, float64(e.Meters))
					}
				}
				if !achou {
					t.Fatalf("nao existe aresta de %d para %d no sentido do caminho",
						p.Nodes[j-1], p.Nodes[j])
				}
				soma += melhor
			}
			if math.Abs(soma-p.Meters) > 1e-3 {
				t.Fatalf("o caminho soma %.3f m, Path diz %.3f m", soma, p.Meters)
			}
		}
	}
}

// A marca de geracao substitui limpar os vetores entre consultas. Se ela
// estiver errada, a segunda consulta enxerga o lixo da primeira -- e o erro
// so aparece a partir da segunda.
func TestSearcherReaproveitadoNaoVazaEntreConsultas(t *testing.T) {
	r := rand.New(rand.NewSource(3))
	g := grafoAleatorio(r, 80, 200)

	reaproveitado := g.NewSearcher()
	for i := 0; i < 400; i++ {
		de, para := NodeID(r.Intn(80)), NodeID(r.Intn(80))

		limpo, okLimpo := g.NewSearcher().Route(de, para, Distance)
		usado, okUsado := reaproveitado.Route(de, para, Distance)

		if okLimpo != okUsado {
			t.Fatalf("consulta %d (%d->%d): buscador limpo achou=%v, reaproveitado achou=%v",
				i, de, para, okLimpo, okUsado)
		}
		if okLimpo && math.Abs(limpo.Meters-usado.Meters) > 1e-9 {
			t.Fatalf("consulta %d (%d->%d): %.6f reaproveitado contra %.6f limpo",
				i, de, para, usado.Meters, limpo.Meters)
		}
	}
}

// Mao unica: o caminho de ida existe, o de volta nao.
func TestMaoUnica(t *testing.T) {
	g := &Graph{
		pontos: []geo.Point{{Lat: -3.0, Lon: -38.0}, {Lat: -3.01, Lon: -38.0}},
		osmIDs: []int64{1, 2},
	}
	g.montarCSR([]trecho{{de: 0, para: 1, metros: 1000, segundos: 72, frente: true, tras: false}})

	if _, ok := g.Route(0, 1, Distance); !ok {
		t.Error("a ida deveria existir")
	}
	if _, ok := g.Route(1, 0, Distance); ok {
		t.Error("a volta nao deveria existir numa via de mao unica")
	}

	// E a referencia tem de concordar, senao o teste de equivalencia estaria
	// comparando dois erros.
	if _, ok := g.RouteRef(1, 0, Distance); ok {
		t.Error("a referencia tambem deveria recusar a volta")
	}
}

// Um trecho de mao unica no sentido contrario: guardado como frente=false,
// tras=true. Se as bandeiras forem trocadas de lado ao montar o CSR, este
// teste pega.
func TestMaoUnicaInvertida(t *testing.T) {
	g := &Graph{
		pontos: []geo.Point{{Lat: -3.0, Lon: -38.0}, {Lat: -3.01, Lon: -38.0}},
		osmIDs: []int64{1, 2},
	}
	g.montarCSR([]trecho{{de: 0, para: 1, metros: 1000, segundos: 72, frente: false, tras: true}})

	if _, ok := g.Route(0, 1, Distance); ok {
		t.Error("0->1 nao deveria existir")
	}
	if _, ok := g.Route(1, 0, Distance); !ok {
		t.Error("1->0 deveria existir")
	}
}

// Tempo e distancia podem preferir caminhos diferentes: o desvio pela via
// rapida e mais longo em quilometros e mais curto em minutos. Se as duas
// metricas sempre concordassem, guardar as duas nao teria sentido.
func TestTempoEDistanciaEscolhemCaminhosDiferentes(t *testing.T) {
	// 0 -> 1 -> 2 pela rua de bairro: 2 km a 30 km/h.
	// 0 -> 3 -> 2 pela via expressa: 3 km a 100 km/h.
	g := &Graph{
		pontos: make([]geo.Point, 4),
		osmIDs: []int64{0, 1, 2, 3},
	}
	kmh := func(m, v float64) float32 { return float32(m / (v / 3.6)) }
	g.montarCSR([]trecho{
		{de: 0, para: 1, metros: 1000, segundos: kmh(1000, 30), frente: true, tras: true},
		{de: 1, para: 2, metros: 1000, segundos: kmh(1000, 30), frente: true, tras: true},
		{de: 0, para: 3, metros: 1500, segundos: kmh(1500, 100), frente: true, tras: true},
		{de: 3, para: 2, metros: 1500, segundos: kmh(1500, 100), frente: true, tras: true},
	})

	curto, ok := g.Route(0, 2, Distance)
	if !ok {
		t.Fatal("sem rota por distancia")
	}
	rapido, ok := g.Route(0, 2, Time)
	if !ok {
		t.Fatal("sem rota por tempo")
	}

	if curto.Nodes[1] != 1 {
		t.Errorf("por distancia deveria passar pelo no 1, passou pelo %d", curto.Nodes[1])
	}
	if rapido.Nodes[1] != 3 {
		t.Errorf("por tempo deveria passar pelo no 3, passou pelo %d", rapido.Nodes[1])
	}
	if curto.Meters >= rapido.Meters {
		t.Errorf("a rota curta (%.0f m) deveria ser mais curta que a rapida (%.0f m)",
			curto.Meters, rapido.Meters)
	}
	if rapido.Seconds >= curto.Seconds {
		t.Errorf("a rota rapida (%.0f s) deveria ser mais rapida que a curta (%.0f s)",
			rapido.Seconds, curto.Seconds)
	}
}

func TestGrafoDesconexo(t *testing.T) {
	g := &Graph{pontos: make([]geo.Point, 4), osmIDs: []int64{0, 1, 2, 3}}
	g.montarCSR([]trecho{
		{de: 0, para: 1, metros: 100, segundos: 10, frente: true, tras: true},
		{de: 2, para: 3, metros: 100, segundos: 10, frente: true, tras: true},
	})

	if _, ok := g.Route(0, 3, Distance); ok {
		t.Error("nao deveria haver rota entre componentes separados")
	}
	if _, ok := g.RouteRef(0, 3, Distance); ok {
		t.Error("a referencia tambem nao deveria achar")
	}
	if p, ok := g.Route(0, 1, Distance); !ok || len(p.Nodes) != 2 {
		t.Errorf("dentro do mesmo componente deveria haver rota, veio %v %v", p, ok)
	}
}

func TestCasosDegenerados(t *testing.T) {
	g := &Graph{pontos: make([]geo.Point, 2), osmIDs: []int64{0, 1}}
	g.montarCSR([]trecho{{de: 0, para: 1, metros: 100, segundos: 10, frente: true, tras: true}})

	// Origem igual ao destino: caminho de um no, custo zero.
	p, ok := g.Route(0, 0, Distance)
	if !ok || len(p.Nodes) != 1 || p.Meters != 0 {
		t.Errorf("no para ele mesmo = %v, %v", p, ok)
	}

	// No fora da faixa nao pode entrar em panico, tem de recusar.
	for _, n := range []NodeID{-1, 2, 999} {
		if _, ok := g.Route(0, n, Distance); ok {
			t.Errorf("no %d deveria ser recusado", n)
		}
		if _, ok := g.Route(n, 0, Distance); ok {
			t.Errorf("no %d como origem deveria ser recusado", n)
		}
	}

	// Grafo vazio.
	vazio := &Graph{}
	vazio.montarCSR(nil)
	if _, ok := vazio.Route(0, 0, Distance); ok {
		t.Error("grafo vazio nao tem rota")
	}
}

// A fila e escrita a mao. Um heap com erro de ordenacao ainda devolve
// caminhos -- so que nao os mais curtos --, entao vale testa-la sozinha.
func TestFilaSaiEmOrdem(t *testing.T) {
	r := rand.New(rand.NewSource(4))
	var f fila

	for caso := 0; caso < 200; caso++ {
		f.limpar()
		n := 1 + r.Intn(200)
		for i := 0; i < n; i++ {
			f.push(NodeID(i), r.Float64()*1000)
		}
		if f.Len() != n {
			t.Fatalf("fila com %d entradas, esperava %d", f.Len(), n)
		}

		anterior := math.Inf(-1)
		for i := 0; i < n; i++ {
			if topo := f.topo(); topo < anterior {
				t.Fatalf("topo %.6f menor que o anterior %.6f", topo, anterior)
			}
			_, custo := f.pop()
			if custo < anterior {
				t.Fatalf("saiu %.6f depois de %.6f", custo, anterior)
			}
			anterior = custo
		}
		if f.Len() != 0 {
			t.Fatalf("sobraram %d entradas", f.Len())
		}
		if !math.IsInf(f.topo(), 1) {
			t.Errorf("topo de fila vazia = %v, esperado +Inf", f.topo())
		}
	}
}
