package ch

import (
	"fmt"
	"math/rand"
	"testing"

	"github.com/iafrotamacedo-cloud/era/maps/graph"
)

var sink float64

// Uma grade de 120 por 120 sao 14.400 cruzamentos. E menor que uma cidade,
// mas grade e o pior caso para a hierarquia -- nao ha rodovia, nao ha
// gargalo, todo caminho tem mil alternativas equivalentes. O ganho medido
// aqui e um piso do que se ve numa malha rodoviaria real.
const ladoBench = 120

func prepararBench(b *testing.B) (*graph.Graph, *CH) {
	b.Helper()
	g := grafoDeGrade(b, ladoBench, true)
	c, err := Prepare(g, graph.Distance)
	if err != nil {
		b.Fatal(err)
	}
	return g, c
}

func paresBench(n int) [][2]graph.NodeID {
	r := rand.New(rand.NewSource(11))
	pares := make([][2]graph.NodeID, 256)
	for i := range pares {
		pares[i] = [2]graph.NodeID{graph.NodeID(r.Intn(n)), graph.NodeID(r.Intn(n))}
	}
	return pares
}

// A comparacao que justifica a fase: consulta na hierarquia contra o Dijkstra
// bidirecional da Fase 4, no mesmo grafo e nos mesmos pares.
func BenchmarkConsultaCH(b *testing.B) {
	_, c := prepararBench(b)
	pares := paresBench(c.Len())
	q := c.NewQuery()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		custo, _ := q.Cost(pares[i&255][0], pares[i&255][1])
		sink = custo
	}
}

func BenchmarkConsultaDijkstra(b *testing.B) {
	g, _ := prepararBench(b)
	pares := paresBench(g.Len())
	s := g.NewSearcher()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		p, _ := s.Route(pares[i&255][0], pares[i&255][1], graph.Distance)
		sink = p.Meters
	}
}

// Desempacotar o caminho custa mais que achar o custo. Quem so quer preencher
// uma matriz nao deveria pagar por isso, e por isso Cost existe separado.
func BenchmarkConsultaCHComCaminho(b *testing.B) {
	_, c := prepararBench(b)
	pares := paresBench(c.Len())
	q := c.NewQuery()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		p, _ := q.Route(pares[i&255][0], pares[i&255][1])
		sink = p.Meters
	}
}

func BenchmarkPreparo(b *testing.B) {
	g := grafoDeGrade(b, 60, true)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		c, err := Prepare(g, graph.Distance)
		if err != nil {
			b.Fatal(err)
		}
		sink = float64(c.Arcs())
	}
}

// A matriz de baldes contra o laco de consultas.
//
// E a diferenca entre roteirizar um dia de entregas em segundos ou em
// minutos, e o ganho cresce com o numero de destinos.
func BenchmarkMatriz100x100(b *testing.B) {
	_, c := prepararBench(b)
	pontos := sortear(rand.New(rand.NewSource(12)), c.Len(), 100)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		m := c.Matrix(pontos, pontos)
		sink = m.At(0, 0)
	}
}

func BenchmarkMatriz100x100ParAPar(b *testing.B) {
	_, c := prepararBench(b)
	pontos := sortear(rand.New(rand.NewSource(12)), c.Len(), 100)
	q := c.NewQuery()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var acc float64
		for _, o := range pontos {
			for _, d := range pontos {
				custo, _ := q.Cost(o, d)
				acc += custo
			}
		}
		sink = acc
	}
}

// Como o preparo escala. E o numero que decide se a hierarquia de um estado
// inteiro e viavel nesta maquina.
func BenchmarkPreparoEscala(b *testing.B) {
	for _, lado := range []int{30, 45, 60, 80} {
		g := grafoDeGrade(b, lado, true)
		b.Run(fmt.Sprintf("%dnos", g.Len()), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				c, err := Prepare(g, graph.Distance)
				if err != nil {
					b.Fatal(err)
				}
				sink = float64(c.Arcs())
			}
		})
	}
}
