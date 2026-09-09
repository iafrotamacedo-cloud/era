package geo

import (
	"math"
	"math/rand"
	"testing"
)

// Os pares de pontos sao sorteados uma vez e reusados por todos os
// benchmarks de distancia, para que a comparacao entre eles seja justa: o
// que varia e a formula, nao a entrada.
func paresDeTeste(n int, alcanceM float64) (a, b []Point) {
	r := rand.New(rand.NewSource(99))
	a = make([]Point, n)
	b = make([]Point, n)
	for i := 0; i < n; i++ {
		a[i] = Destination(fortaleza, r.Float64()*360, r.Float64()*alcanceM)
		b[i] = Destination(fortaleza, r.Float64()*360, r.Float64()*alcanceM)
	}
	return a, b
}

func BenchmarkHaversine(b *testing.B) {
	pa, pb := paresDeTeste(1024, 300_000)
	b.ResetTimer()
	var acc float64
	for i := 0; i < b.N; i++ {
		j := i & 1023
		acc += Haversine(pa[j], pb[j])
	}
	sink = acc
}

func BenchmarkRuler(b *testing.B) {
	pa, pb := paresDeTeste(1024, 300_000)
	r := NewRuler(fortaleza)
	b.ResetTimer()
	var acc float64
	for i := 0; i < b.N; i++ {
		j := i & 1023
		acc += r.Distance(pa[j], pb[j])
	}
	sink = acc
}

func BenchmarkRulerSquared(b *testing.B) {
	pa, pb := paresDeTeste(1024, 300_000)
	r := NewRuler(fortaleza)
	b.ResetTimer()
	var acc float64
	for i := 0; i < b.N; i++ {
		j := i & 1023
		acc += r.SquaredDistance(pa[j], pb[j])
	}
	sink = acc
}

func BenchmarkVincenty(b *testing.B) {
	pa, pb := paresDeTeste(1024, 300_000)
	b.ResetTimer()
	var acc float64
	for i := 0; i < b.N; i++ {
		j := i & 1023
		d, _ := Vincenty(pa[j], pb[j])
		acc += d
	}
	sink = acc
}

// sink impede que o compilador descarte os calculos dos benchmarks por
// perceber que o resultado nao e usado.
var sink float64

// A comparacao que justifica a existencia da grade: mesma pergunta, mesma
// resposta, feita das duas maneiras. Sem indice, o custo cresce com o
// tamanho da base; com indice, com a densidade local.
const pontosNaBase = 100_000

// espalharUniforme sorteia um ponto com densidade uniforme *por area* dentro
// de um disco.
//
// A raiz quadrada nao e decoracao. Sortear o raio direto amontoa os pontos no
// centro: a area de um anel cresce com o raio, entao raio uniforme significa
// densidade proporcional a 1/r. Num benchmark de indice espacial isso mente
// duas vezes -- infla o numero de vizinhos por consulta e concentra tudo em
// poucas celulas --, e o resultado deixa de representar uma carteira de
// clientes espalhada por um estado.
func espalharUniforme(r *rand.Rand, centro Point, raioM float64) Point {
	return Destination(centro, r.Float64()*360, raioM*math.Sqrt(r.Float64()))
}

func baseDeTeste(b *testing.B) ([]int, []Point, []Point) {
	b.Helper()
	r := rand.New(rand.NewSource(100))

	// Uma area do tamanho de um estado, que e a escala de operacao real.
	const alcance = 300_000.0

	ids := make([]int, pontosNaBase)
	pts := make([]Point, pontosNaBase)
	for i := range pts {
		ids[i] = i
		pts[i] = espalharUniforme(r, fortaleza, alcance)
	}

	consultas := make([]Point, 1024)
	for i := range consultas {
		consultas[i] = espalharUniforme(r, fortaleza, alcance)
	}
	return ids, pts, consultas
}

func BenchmarkGridWithin10km(b *testing.B) {
	ids, pts, consultas := baseDeTeste(b)

	g := NewGridForRadius(10_000)
	for i := range pts {
		g.Add(ids[i], pts[i])
	}

	b.ResetTimer()
	var n int
	for i := 0; i < b.N; i++ {
		n += len(g.Within(consultas[i&1023], 10_000))
	}
	sinkInt = n
}

func BenchmarkVarreduraLinear10km(b *testing.B) {
	ids, pts, consultas := baseDeTeste(b)

	b.ResetTimer()
	var n int
	for i := 0; i < b.N; i++ {
		n += len(varreduraLinear(ids, pts, consultas[i&1023], 10_000))
	}
	sinkInt = n
}

func BenchmarkGridNearest10(b *testing.B) {
	ids, pts, consultas := baseDeTeste(b)

	g := NewGridForRadius(10_000)
	for i := range pts {
		g.Add(ids[i], pts[i])
	}

	b.ResetTimer()
	var n int
	for i := 0; i < b.N; i++ {
		n += len(g.Nearest(consultas[i&1023], 10))
	}
	sinkInt = n
}

// Montar o indice tambem custa. Numa operacao logistica isso e pago uma vez,
// no carregamento do cadastro, e amortizado por milhares de consultas -- mas
// o numero precisa estar visivel para essa afirmacao ser conferivel.
func BenchmarkGridConstrucao(b *testing.B) {
	ids, pts, _ := baseDeTeste(b)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		g := NewGridForRadius(10_000)
		for j := range pts {
			g.Add(ids[j], pts[j])
		}
		sinkInt = g.Len()
	}
}

var sinkInt int

// O caso que o motor de rotas faz o tempo todo: encaixar uma coordenada no
// ponto mais proximo. E k=1, e era o que mais sofria com a versao que juntava
// e ordenava todos os candidatos do raio.
func BenchmarkGridNearest1(b *testing.B) {
	ids, pts, consultas := baseDeTeste(b)

	g := NewGridForRadius(10_000)
	for i := range pts {
		g.Add(ids[i], pts[i])
	}

	b.ResetTimer()
	var n int
	for i := 0; i < b.N; i++ {
		n += len(g.Nearest(consultas[i&1023], 1))
	}
	sinkInt = n
}
