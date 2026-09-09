package graph

import (
	"context"
	"errors"
	"math"
	"testing"

	"github.com/iafrotamacedo-cloud/era/maps/dist"
	"github.com/iafrotamacedo-cloud/era/maps/geo"
)

// A promessa da Fase 2 era que o Estimator sairia e o motor de grafo entraria
// sem que o sistema chamador soubesse. Isto e a checagem dela, em tempo de
// compilacao.
var _ dist.Distancer = (*Router)(nil)

// grade monta um grafo quadriculado de n por n cruzamentos, 100 m de lado.
//
// E um mapa de cidade sem os detalhes: quarteiroes regulares, todas as ruas
// de mao dupla. Serve para medir e para conferir contra uma conta que da para
// fazer de cabeca -- num quadriculado, a menor rota entre dois cruzamentos e
// a distancia de quarteiroes, e nada mais.
func grade(n int) *Graph {
	g := &Graph{}
	const passo = 0.0009 // ~100 m

	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			g.pontos = append(g.pontos, geo.Point{
				Lat: -3.7 - float64(i)*passo,
				Lon: -38.5 + float64(j)*passo,
			})
			g.osmIDs = append(g.osmIDs, int64(i*n+j))
		}
	}

	id := func(i, j int) NodeID { return NodeID(i*n + j) }
	var ts []trecho
	liga := func(a, b NodeID) {
		metros := geo.Haversine(g.pontos[a], g.pontos[b])
		ts = append(ts, trecho{
			de: a, para: b,
			metros:   float32(metros),
			segundos: float32(metros / (30.0 / 3.6)),
			frente:   true, tras: true,
		})
	}
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			if j+1 < n {
				liga(id(i, j), id(i, j+1))
			}
			if i+1 < n {
				liga(id(i, j), id(i+1, j))
			}
		}
	}

	g.montarCSR(ts)
	return g
}

// Num quadriculado a menor rota e a distancia de quarteiroes. Se a busca
// estiver certa, o numero de arestas do caminho tem de ser exatamente
// |di| + |dj|.
func TestGradeDaADistanciaDeQuarteiroes(t *testing.T) {
	const n = 12
	g := grade(n)
	s := g.NewSearcher()

	for i1 := 0; i1 < n; i1 += 3 {
		for j1 := 0; j1 < n; j1 += 3 {
			for i2 := 0; i2 < n; i2 += 3 {
				for j2 := 0; j2 < n; j2 += 3 {
					de := NodeID(i1*n + j1)
					para := NodeID(i2*n + j2)

					p, ok := s.Route(de, para, Distance)
					if !ok {
						t.Fatalf("sem rota de (%d,%d) a (%d,%d)", i1, j1, i2, j2)
					}
					querido := abs(i1-i2) + abs(j1-j2)
					if len(p.Nodes)-1 != querido {
						t.Fatalf("(%d,%d)->(%d,%d): caminho com %d arestas, esperado %d",
							i1, j1, i2, j2, len(p.Nodes)-1, querido)
					}
				}
			}
		}
	}
}

func abs(v int) int {
	if v < 0 {
		return -v
	}
	return v
}

func TestRouterConcordaComRoute(t *testing.T) {
	g := grade(10)
	r := NewRouter(g, Distance)
	ctx := context.Background()

	for _, par := range [][2]int{{0, 99}, {5, 50}, {12, 87}, {0, 0}} {
		de, para := NodeID(par[0]), NodeID(par[1])

		querida, ok := g.Route(de, para, Distance)
		if !ok {
			t.Fatalf("sem rota entre %d e %d", de, para)
		}
		leg, err := r.Distance(ctx, g.Point(de), g.Point(para))
		if err != nil {
			t.Fatal(err)
		}

		if math.Abs(leg.Meters-querida.Meters) > 1e-6 {
			t.Errorf("%d->%d: Router deu %.3f m, Route deu %.3f m", de, para, leg.Meters, querida.Meters)
		}
		if math.Abs(leg.Seconds-querida.Seconds) > 1e-6 {
			t.Errorf("%d->%d: Router deu %.3f s, Route deu %.3f s", de, para, leg.Seconds, querida.Seconds)
		}
	}
}

func TestRouterMatrixConcordaComDistance(t *testing.T) {
	g := grade(8)
	r := NewRouter(g, Time)
	ctx := context.Background()

	origens := []geo.Point{g.Point(0), g.Point(9), g.Point(20), g.Point(63), g.Point(35)}
	destinos := []geo.Point{g.Point(7), g.Point(56), g.Point(30)}

	m, err := r.Matrix(ctx, origens, destinos)
	if err != nil {
		t.Fatal(err)
	}
	if m.Rows() != len(origens) || m.Cols() != len(destinos) {
		t.Fatalf("matriz %dx%d, esperada %dx%d", m.Rows(), m.Cols(), len(origens), len(destinos))
	}

	for i, o := range origens {
		for j, d := range destinos {
			querida, err := r.Distance(ctx, o, d)
			if err != nil {
				t.Fatal(err)
			}
			got := m.At(i, j)
			// A matriz guarda float32; a tolerancia e a precisao dele.
			if math.Abs(got.Seconds-querida.Seconds) > 1e-3*math.Max(1, querida.Seconds) {
				t.Errorf("(%d,%d): matriz %.4f s, Distance %.4f s", i, j, got.Seconds, querida.Seconds)
			}
		}
	}
}

// Uma ilha sem ponte existe de verdade no OpenStreetMap, e o Router precisa
// dizer isso em vez de devolver zero -- que passaria por "esta do lado".
func TestRouterSemCaminho(t *testing.T) {
	g := &Graph{
		pontos: []geo.Point{
			{Lat: -3.70, Lon: -38.50}, {Lat: -3.71, Lon: -38.50},
			{Lat: -3.90, Lon: -38.90}, {Lat: -3.91, Lon: -38.90},
		},
		osmIDs: []int64{1, 2, 3, 4},
	}
	g.montarCSR([]trecho{
		{de: 0, para: 1, metros: 1000, segundos: 120, frente: true, tras: true},
		{de: 2, para: 3, metros: 1000, segundos: 120, frente: true, tras: true},
	})

	r := NewRouter(g, Distance)
	_, err := r.Distance(context.Background(), g.Point(0), g.Point(3))
	if !errors.Is(err, ErrSemCaminho) {
		t.Errorf("erro = %v, esperado ErrSemCaminho", err)
	}
}

func TestRouterRespeitaCancelamento(t *testing.T) {
	g := grade(10)
	r := NewRouter(g, Distance)
	ctx, cancelar := context.WithCancel(context.Background())
	cancelar()

	if _, err := r.Distance(ctx, g.Point(0), g.Point(99)); !errors.Is(err, context.Canceled) {
		t.Errorf("Distance: erro = %v, esperado context.Canceled", err)
	}

	pontos := make([]geo.Point, 20)
	for i := range pontos {
		pontos[i] = g.Point(NodeID(i))
	}
	if _, err := r.Matrix(ctx, pontos, pontos); !errors.Is(err, context.Canceled) {
		t.Errorf("Matrix: erro = %v, esperado context.Canceled", err)
	}
}

// O cache da Fase 2 foi escrito para este momento. Contra o Estimator ele
// perdia; contra uma busca no grafo, a segunda consulta nao paga nada.
func TestCacheNaFrenteDoRouter(t *testing.T) {
	g := grade(20)
	r := NewRouter(g, Distance)
	c := dist.NewCache(r)
	ctx := context.Background()

	a, b := g.Point(0), g.Point(399)

	primeira, err := c.Distance(ctx, a, b)
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 50; i++ {
		got, err := c.Distance(ctx, a, b)
		if err != nil {
			t.Fatal(err)
		}
		if got != primeira {
			t.Fatalf("resposta mudou: %+v depois %+v", primeira, got)
		}
	}

	if c.Hits() != 50 || c.Misses() != 1 {
		t.Errorf("acertos %d e faltas %d, esperava 50 e 1", c.Hits(), c.Misses())
	}
}
