package ch

import (
	"context"
	"errors"
	"math"
	"math/rand"
	"testing"

	"github.com/iafrotamacedo-cloud/era/maps/dist"
	"github.com/iafrotamacedo-cloud/era/maps/geo"
	"github.com/iafrotamacedo-cloud/era/maps/graph"
)

// A interface da Fase 2 aceita a terceira implementacao. Checagem em tempo de
// compilacao.
var _ dist.Distancer = (*Router)(nil)

func routerDeTeste(t *testing.T, m graph.Metric) (*graph.Graph, *CH, *Router) {
	t.Helper()
	g := grafoDeGrade(t, 10, true)
	c, err := Prepare(g, m)
	if err != nil {
		t.Fatal(err)
	}
	return g, c, NewRouter(c)
}

// O Router da hierarquia tem de concordar com o Router do grafo da Fase 4 --
// que ja concorda com o Dijkstra obvio. Sao caminhos de verificacao
// independentes chegando na mesma resposta.
func TestRouterConcordaComOGrafo(t *testing.T) {
	for _, m := range []graph.Metric{graph.Distance, graph.Time} {
		g, c, rapido := routerDeTeste(t, m)
		lento := graph.NewRouter(g, m)
		ctx := context.Background()

		r := rand.New(rand.NewSource(30))
		for i := 0; i < 120; i++ {
			a := c.Point(graph.NodeID(r.Intn(c.Len())))
			b := c.Point(graph.NodeID(r.Intn(c.Len())))

			querida, errLento := lento.Distance(ctx, a, b)
			obtida, errRapido := rapido.Distance(ctx, a, b)

			if (errLento != nil) != (errRapido != nil) {
				t.Fatalf("%v: grafo err=%v, hierarquia err=%v", m, errLento, errRapido)
			}
			if errLento != nil {
				continue
			}

			if math.Abs(obtida.Meters-querida.Meters) > 1e-3*math.Max(1, querida.Meters) {
				t.Fatalf("%v: %.4f m, o grafo diz %.4f m", m, obtida.Meters, querida.Meters)
			}
			if math.Abs(obtida.Seconds-querida.Seconds) > 1e-3*math.Max(1, querida.Seconds) {
				t.Fatalf("%v: %.4f s, o grafo diz %.4f s", m, obtida.Seconds, querida.Seconds)
			}
		}
	}
}

// As duas grandezas sao acumuladas durante a busca, sem desempacotar o
// caminho. Elas tem de bater com as que saem de Route, que desempacota.
func TestLegBateComRoute(t *testing.T) {
	for _, m := range []graph.Metric{graph.Distance, graph.Time} {
		_, c, _ := routerDeTeste(t, m)
		q := c.NewQuery()

		conferidos := 0
		for de := graph.NodeID(0); int(de) < c.Len(); de += 3 {
			for para := graph.NodeID(0); int(para) < c.Len(); para += 3 {
				metros, segundos, ok := q.Leg(de, para)
				p, okRota := q.Route(de, para)
				if ok != okRota {
					t.Fatalf("%v %d->%d: Leg achou=%v, Route achou=%v", m, de, para, ok, okRota)
				}
				if !ok || de == para {
					continue
				}
				conferidos++

				if math.Abs(metros-p.Meters) > 1e-3*math.Max(1, p.Meters) {
					t.Fatalf("%v %d->%d: Leg %.4f m, Route %.4f m", m, de, para, metros, p.Meters)
				}
				if math.Abs(segundos-p.Seconds) > 1e-3*math.Max(1, p.Seconds) {
					t.Fatalf("%v %d->%d: Leg %.4f s, Route %.4f s", m, de, para, segundos, p.Seconds)
				}
			}
		}
		if conferidos < 50 {
			t.Errorf("%v: so %d pares conferidos", m, conferidos)
		}
	}
}

// A matriz de baldes tambem carrega as duas grandezas. Elas tem de bater com
// as consultas par a par.
func TestMatrixDoRouterConcordaComDistance(t *testing.T) {
	_, c, r := routerDeTeste(t, graph.Time)
	ctx := context.Background()

	rnd := rand.New(rand.NewSource(31))
	origens := make([]geo.Point, 7)
	destinos := make([]geo.Point, 5)
	for i := range origens {
		origens[i] = c.Point(graph.NodeID(rnd.Intn(c.Len())))
	}
	for j := range destinos {
		destinos[j] = c.Point(graph.NodeID(rnd.Intn(c.Len())))
	}

	m, err := r.Matrix(ctx, origens, destinos)
	if err != nil {
		t.Fatal(err)
	}

	for i, o := range origens {
		for j, d := range destinos {
			querida, err := r.Distance(ctx, o, d)
			if err != nil {
				continue
			}
			got := m.At(i, j)
			// A dist.Matrix guarda float32; a tolerancia e a precisao dele.
			if math.Abs(got.Meters-querida.Meters) > 1e-3*math.Max(1, querida.Meters) {
				t.Errorf("(%d,%d): matriz %.4f m, Distance %.4f m", i, j, got.Meters, querida.Meters)
			}
			if math.Abs(got.Seconds-querida.Seconds) > 1e-3*math.Max(1, querida.Seconds) {
				t.Errorf("(%d,%d): matriz %.4f s, Distance %.4f s", i, j, got.Seconds, querida.Seconds)
			}
		}
	}
}

func TestRouterSemCaminho(t *testing.T) {
	nos := []noPbf{
		{id: 1, p: pontoEm(-3.70, -38.50)}, {id: 2, p: pontoEm(-3.70, -38.49)},
		{id: 3, p: pontoEm(-4.90, -39.90)}, {id: 4, p: pontoEm(-4.90, -39.89)},
	}
	g := montarGrafo(t, nos, []viaPbf{
		{id: 1, refs: []int64{1, 2}, tags: map[string]string{"highway": "residential"}},
		{id: 2, refs: []int64{3, 4}, tags: map[string]string{"highway": "residential"}},
	})
	c, err := Prepare(g, graph.Distance)
	if err != nil {
		t.Fatal(err)
	}
	r := NewRouter(c)

	_, err = r.Distance(context.Background(), c.Point(0), c.Point(3))
	if !errors.Is(err, ErrSemCaminho) {
		t.Errorf("erro = %v, esperado ErrSemCaminho", err)
	}
}

func TestRouterRespeitaCancelamento(t *testing.T) {
	_, c, r := routerDeTeste(t, graph.Distance)
	ctx, cancelar := context.WithCancel(context.Background())
	cancelar()

	if _, err := r.Distance(ctx, c.Point(0), c.Point(5)); !errors.Is(err, context.Canceled) {
		t.Errorf("Distance: erro = %v, esperado context.Canceled", err)
	}

	pontos := []geo.Point{c.Point(0), c.Point(1)}
	if _, err := r.Matrix(ctx, pontos, pontos); !errors.Is(err, context.Canceled) {
		t.Errorf("Matrix: erro = %v, esperado context.Canceled", err)
	}
}

// O Router carregado de um .eramap funciona igual. E o caminho de producao:
// preparar numa maquina, gravar, e servir consultas em outra.
func TestRouterDepoisDoEramap(t *testing.T) {
	_, original, rOriginal := routerDeTeste(t, graph.Time)

	caminho := t.TempDir() + "/teste.eramap"
	if err := original.SaveFile(caminho); err != nil {
		t.Fatal(err)
	}
	carregada, err := LoadFile(caminho)
	if err != nil {
		t.Fatal(err)
	}
	rCarregado := NewRouter(carregada)

	ctx := context.Background()
	rnd := rand.New(rand.NewSource(32))
	for i := 0; i < 60; i++ {
		a := original.Point(graph.NodeID(rnd.Intn(original.Len())))
		b := original.Point(graph.NodeID(rnd.Intn(original.Len())))

		antes, err1 := rOriginal.Distance(ctx, a, b)
		depois, err2 := rCarregado.Distance(ctx, a, b)

		if (err1 != nil) != (err2 != nil) {
			t.Fatalf("original err=%v, carregado err=%v", err1, err2)
		}
		if err1 == nil && antes != depois {
			t.Fatalf("%+v depois do arquivo, era %+v", depois, antes)
		}
	}
}
