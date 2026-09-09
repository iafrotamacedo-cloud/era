package ch

import (
	"bytes"
	"context"
	"fmt"
	"math/rand"
	"testing"

	"github.com/iafrotamacedo-cloud/era/maps/geo"
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

// A comparacao que justifica o .eramap existir: carregar contra preparar.
//
// Sao a mesma hierarquia, chegando pelos dois caminhos possiveis.
func BenchmarkCarregarEramap(b *testing.B) {
	g := grafoDeGrade(b, 60, true)
	c, err := Prepare(g, graph.Distance)
	if err != nil {
		b.Fatal(err)
	}
	var buf bytes.Buffer
	if err := c.Save(&buf); err != nil {
		b.Fatal(err)
	}
	dados := buf.Bytes()
	b.SetBytes(int64(len(dados)))

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		lida, err := Load(bytes.NewReader(dados))
		if err != nil {
			b.Fatal(err)
		}
		sink = float64(lida.Arcs())
	}
}

func BenchmarkGravarEramap(b *testing.B) {
	g := grafoDeGrade(b, 60, true)
	c, err := Prepare(g, graph.Distance)
	if err != nil {
		b.Fatal(err)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var buf bytes.Buffer
		if err := c.Save(&buf); err != nil {
			b.Fatal(err)
		}
		sink = float64(buf.Len())
	}
}

// Os tres dist.Distancer na mesma pergunta.
//
// A interface foi escrita na Fase 2 para que o motor pudesse ser trocado sem
// que o chamador soubesse. Estes tres numeros sao a prova de que ela aguentou.
func BenchmarkDistancerCH(b *testing.B) {
	_, c := prepararBench(b)
	r := NewRouter(c)
	ctx := context.Background()
	// Pares sorteados, e nao um par fixo de canto a canto. O canto a canto e
	// o pior caso da grade e mede umas seis vezes a consulta media -- compara-
	// lo com os benchmarks de consulta, que sorteiam, seria comparar coisas
	// diferentes.
	pares := paresAlcancaveis(c, 256)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		leg, err := r.Distance(ctx, pares[i&255][0], pares[i&255][1])
		if err != nil {
			b.Fatal(err)
		}
		sink = leg.Meters
	}
}

func BenchmarkDistancerGrafo(b *testing.B) {
	g, c := prepararBench(b)
	r := graph.NewRouter(g, graph.Distance)
	ctx := context.Background()
	pares := paresAlcancaveis(c, 256)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		leg, err := r.Distance(ctx, pares[i&255][0], pares[i&255][1])
		if err != nil {
			b.Fatal(err)
		}
		sink = leg.Meters
	}
}

// A matriz de um dia de entregas, pelos dois motores que sabem responder
// rota de verdade.
func BenchmarkMatrixDistancerCH(b *testing.B) {
	_, c := prepararBench(b)
	r := NewRouter(c)
	ctx := context.Background()
	pontos := pontosDe(c, 60)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		m, err := r.Matrix(ctx, pontos, pontos)
		if err != nil {
			b.Fatal(err)
		}
		sink = m.At(0, 0).Meters
	}
}

func BenchmarkMatrixDistancerGrafo(b *testing.B) {
	g, c := prepararBench(b)
	r := graph.NewRouter(g, graph.Distance)
	ctx := context.Background()
	pontos := pontosDe(c, 60)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		m, err := r.Matrix(ctx, pontos, pontos)
		if err != nil {
			b.Fatal(err)
		}
		sink = m.At(0, 0).Meters
	}
}

// paresAlcancaveis sorteia pares que tem caminho.
//
// A grade tem mao unica, e nem todo par e alcancavel -- como num mapa de
// verdade. Um benchmark que sorteasse cegamente mediria uma mistura de rotas
// achadas e buscas que percorrem tudo sem achar nada, e o numero nao diria
// respeito a nenhuma das duas.
func paresAlcancaveis(c *CH, quantos int) [][2]geo.Point {
	r := rand.New(rand.NewSource(13))
	q := c.NewQuery()

	pares := make([][2]geo.Point, 0, quantos)
	for len(pares) < quantos {
		de := graph.NodeID(r.Intn(c.Len()))
		para := graph.NodeID(r.Intn(c.Len()))
		if de == para {
			continue
		}
		if _, ok := q.Cost(de, para); !ok {
			continue
		}
		pares = append(pares, [2]geo.Point{c.Point(de), c.Point(para)})
	}
	return pares
}

func pontosDe(c *CH, quantos int) []geo.Point {
	r := rand.New(rand.NewSource(13))
	out := make([]geo.Point, quantos)
	for i := range out {
		out[i] = c.Point(graph.NodeID(r.Intn(c.Len())))
	}
	return out
}

// Quanto do custo de Distance e o encaixe da coordenada no cruzamento mais
// proximo, e nao a busca da rota.
//
// Depois que a hierarquia acelerou a busca, o gargalo pode ter mudado de
// lugar -- e vale medir em vez de supor.
func BenchmarkNearest(b *testing.B) {
	_, c := prepararBench(b)
	pontos := pontosDe(c, 256)
	c.Nearest(pontos[0]) // a grade e montada na primeira chamada

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, m, _ := c.Nearest(pontos[i&255])
		sink = m
	}
}
