package graph

import (
	"context"
	"math/rand"
	"testing"

	"github.com/iafrotamacedo-cloud/era/maps/dist"
	"github.com/iafrotamacedo-cloud/era/maps/geo"
)

var sink float64

// Uma grade de 300 por 300 sao 90 mil cruzamentos e 180 mil trechos -- a
// ordem de grandeza da malha rodoviaria de uma cidade media brasileira depois
// da contracao.
const ladoGrade = 300

func paresAleatorios(n, quantos int) [][2]NodeID {
	r := rand.New(rand.NewSource(7))
	pares := make([][2]NodeID, quantos)
	for i := range pares {
		pares[i] = [2]NodeID{NodeID(r.Intn(n)), NodeID(r.Intn(n))}
	}
	return pares
}

// A comparacao que justifica ter escrito o Dijkstra bidirecional: ele contra
// o Dijkstra obvio, no mesmo grafo e nos mesmos pares.
func BenchmarkRotaBidirecional(b *testing.B) {
	g := grade(ladoGrade)
	pares := paresAleatorios(g.Len(), 256)
	s := g.NewSearcher()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		p, _ := s.Route(pares[i&255][0], pares[i&255][1], Distance)
		sink = p.Meters
	}
}

func BenchmarkRotaReferencia(b *testing.B) {
	g := grade(ladoGrade)
	pares := paresAleatorios(g.Len(), 256)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		p, _ := g.RouteRef(pares[i&255][0], pares[i&255][1], Distance)
		sink = p.Meters
	}
}

// Quanto custa nao reaproveitar as estruturas de busca.
//
// Num grafo de 90 mil nos sao seis vetores desse tamanho por consulta. E o
// que o Searcher existe para evitar.
func BenchmarkRotaSemReaproveitar(b *testing.B) {
	g := grade(ladoGrade)
	pares := paresAleatorios(g.Len(), 256)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		p, _ := g.Route(pares[i&255][0], pares[i&255][1], Distance)
		sink = p.Meters
	}
}

func BenchmarkNearest(b *testing.B) {
	g := grade(ladoGrade)
	r := rand.New(rand.NewSource(8))

	alvos := make([]geo.Point, 256)
	for i := range alvos {
		alvos[i] = geo.Point{
			Lat: -3.7 - r.Float64()*0.27,
			Lon: -38.5 + r.Float64()*0.27,
		}
	}
	// Fora do laco medido: a grade e montada na primeira chamada.
	g.Nearest(alvos[0])

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, m, _ := g.Nearest(alvos[i&255])
		sink = m
	}
}

// A conta do cache invertida.
//
// Contra o dist.Estimator, guardar custava mais que recalcular -- uma busca
// num mapa de milhoes de entradas perde para uma multiplicacao. Contra uma
// busca no grafo, a mesma busca no mapa e ruido.
func BenchmarkRouterSemCache(b *testing.B) {
	g := grade(100)
	r := NewRouter(g, Distance)
	ctx := context.Background()
	a, z := g.Point(0), g.Point(NodeID(g.Len()-1))

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		leg, err := r.Distance(ctx, a, z)
		if err != nil {
			b.Fatal(err)
		}
		sink = leg.Meters
	}
}

func BenchmarkRouterComCache(b *testing.B) {
	g := grade(100)
	c := dist.NewCache(NewRouter(g, Distance))
	ctx := context.Background()
	a, z := g.Point(0), g.Point(NodeID(g.Len()-1))

	if _, err := c.Distance(ctx, a, z); err != nil {
		b.Fatal(err)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		leg, err := c.Distance(ctx, a, z)
		if err != nil {
			b.Fatal(err)
		}
		sink = leg.Meters
	}
}

// paresVizinhos sorteia pares proximos: uma entrega para a seguinte, que e a
// consulta que uma roteirizacao faz aos milhares.
func paresVizinhos(lado, quantos int) [][2]NodeID {
	r := rand.New(rand.NewSource(9))
	pares := make([][2]NodeID, quantos)
	for i := range pares {
		li, lj := r.Intn(lado-10), r.Intn(lado-10)
		pares[i] = [2]NodeID{
			NodeID(li*lado + lj),
			NodeID((li+1+r.Intn(9))*lado + lj + 1 + r.Intn(9)),
		}
	}
	return pares
}

// Onde a reutilizacao das estruturas de busca realmente pesa.
//
// A alocacao e proporcional ao tamanho do grafo, e a busca ao tamanho da
// consulta. Numa consulta que atravessa a cidade as duas sao grandes e a
// alocacao some no meio; numa consulta de dez quarteiroes, ela e quase tudo.
func BenchmarkRotaCurtaReaproveitando(b *testing.B) {
	g := grade(ladoGrade)
	pares := paresVizinhos(ladoGrade, 256)
	s := g.NewSearcher()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		p, _ := s.Route(pares[i&255][0], pares[i&255][1], Distance)
		sink = p.Meters
	}
}

func BenchmarkRotaCurtaSemReaproveitar(b *testing.B) {
	g := grade(ladoGrade)
	pares := paresVizinhos(ladoGrade, 256)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		p, _ := g.Route(pares[i&255][0], pares[i&255][1], Distance)
		sink = p.Meters
	}
}
