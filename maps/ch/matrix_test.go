package ch

import (
	"math"
	"math/rand"
	"testing"

	"github.com/iafrotamacedo-cloud/era/maps/graph"
)

// A matriz de baldes tem de dar exatamente o mesmo que perguntar par a par.
// E um algoritmo bem diferente da consulta simples -- as duas metades da
// busca sao feitas em fases separadas e so se encontram nos baldes --, entao
// ele merece ser conferido contra a consulta, que ja foi conferida contra o
// Dijkstra.
func TestMatrizBateComAsConsultas(t *testing.T) {
	r := rand.New(rand.NewSource(10))

	for caso := 0; caso < 12; caso++ {
		g := grafoAleatorio(t, r, 30+r.Intn(40), 120)
		if g.Len() < 6 {
			continue
		}

		for _, m := range []graph.Metric{graph.Distance, graph.Time} {
			c, err := Prepare(g, m)
			if err != nil {
				t.Fatal(err)
			}

			origens := sortear(r, g.Len(), 5)
			destinos := sortear(r, g.Len(), 7)

			mat := c.Matrix(origens, destinos)
			q := c.NewQuery()

			for i, o := range origens {
				for j, d := range destinos {
					querido, ok := q.Cost(o, d)

					if !ok {
						if mat.Alcancavel(i, j) {
							t.Fatalf("caso %d %v: (%d,%d) a matriz achou caminho e a consulta nao",
								caso, m, o, d)
						}
						continue
					}
					if !mat.Alcancavel(i, j) {
						t.Fatalf("caso %d %v: (%d,%d) a consulta achou caminho e a matriz nao",
							caso, m, o, d)
					}
					if math.Abs(mat.At(i, j)-querido) > 1e-6*math.Max(1, querido) {
						t.Fatalf("caso %d %v: (%d,%d) matriz %.6f, consulta %.6f",
							caso, m, o, d, mat.At(i, j), querido)
					}
				}
			}
		}
	}
}

// A diagonal de uma matriz de um conjunto contra ele mesmo tem de ser zero, e
// o resto tem de bater com o Dijkstra.
func TestMatrizDeUmConjuntoContraSiMesmo(t *testing.T) {
	g := grafoDeGrade(t, 7, true)
	c, err := Prepare(g, graph.Distance)
	if err != nil {
		t.Fatal(err)
	}

	todos := make([]graph.NodeID, g.Len())
	for i := range todos {
		todos[i] = graph.NodeID(i)
	}

	mat := c.Matrix(todos, todos)
	for i := range todos {
		if mat.At(i, i) != 0 {
			t.Fatalf("diagonal (%d,%d) = %v, esperado 0", i, i, mat.At(i, i))
		}
		for j := range todos {
			querido, ok := g.Route(todos[i], todos[j], graph.Distance)
			if ok != mat.Alcancavel(i, j) {
				t.Fatalf("(%d,%d): matriz alcancavel=%v, dijkstra=%v", i, j, mat.Alcancavel(i, j), ok)
			}
			if ok && math.Abs(mat.At(i, j)-querido.Meters) > 1e-4*math.Max(1, querido.Meters) {
				t.Fatalf("(%d,%d): matriz %.4f, dijkstra %.4f", i, j, mat.At(i, j), querido.Meters)
			}
		}
	}
}

// Par sem caminho tem de vir infinito, e nao zero -- que passaria por "esta
// do lado".
func TestMatrizMarcaOInalcancavel(t *testing.T) {
	// Duas ilhas sem ponte.
	nos := []noPbf{
		{id: 1, p: pontoEm(-3.70, -38.50)}, {id: 2, p: pontoEm(-3.70, -38.49)},
		{id: 3, p: pontoEm(-4.90, -39.90)}, {id: 4, p: pontoEm(-4.90, -39.89)},
	}
	vias := []viaPbf{
		{id: 1, refs: []int64{1, 2}, tags: map[string]string{"highway": "residential"}},
		{id: 2, refs: []int64{3, 4}, tags: map[string]string{"highway": "residential"}},
	}
	g := montarGrafo(t, nos, vias)

	c, err := Prepare(g, graph.Distance)
	if err != nil {
		t.Fatal(err)
	}

	todos := []graph.NodeID{0, 1, 2, 3}
	mat := c.Matrix(todos, todos)

	alcancaveis := 0
	for i := range todos {
		for j := range todos {
			if mat.Alcancavel(i, j) {
				alcancaveis++
			}
		}
	}
	// Duas ilhas de dois nos: 4 pares dentro de cada uma, 8 no total.
	if alcancaveis != 8 {
		t.Errorf("%d pares alcancaveis, esperava 8", alcancaveis)
	}
	if !math.IsInf(mat.At(0, 2), 1) && !math.IsInf(mat.At(0, 3), 1) {
		t.Error("os pares entre ilhas deveriam vir infinitos")
	}
}

func TestMatrizVazia(t *testing.T) {
	g := grafoDeGrade(t, 4, false)
	c, err := Prepare(g, graph.Distance)
	if err != nil {
		t.Fatal(err)
	}

	if m := c.Matrix(nil, []graph.NodeID{0}); len(m.Origens) != 0 || m.Cols() != 1 {
		t.Errorf("matriz sem origens = %dx%d", len(m.Origens), m.Cols())
	}
	if m := c.Matrix([]graph.NodeID{0}, nil); len(m.Destinos) != 0 {
		t.Errorf("matriz sem destinos tem %d destinos", len(m.Destinos))
	}
}

func sortear(r *rand.Rand, n, quantos int) []graph.NodeID {
	if quantos > n {
		quantos = n
	}
	out := make([]graph.NodeID, quantos)
	for i := range out {
		out[i] = graph.NodeID(r.Intn(n))
	}
	return out
}
