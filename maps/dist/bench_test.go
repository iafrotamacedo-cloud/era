package dist

import (
	"context"
	"math"
	"math/rand"
	"testing"

	"github.com/iafrotamacedo-cloud/era/maps/geo"
)

var sink float64
var sinkMatrix *Matrix

// paradas devolve pontos espalhados com densidade uniforme por area -- a
// raiz quadrada e o que impede que se amontoem no centro.
func paradas(n int, alcanceM float64) []geo.Point {
	r := rand.New(rand.NewSource(42))
	p := make([]geo.Point, n)
	for i := range p {
		p[i] = geo.Destination(fortaleza, r.Float64()*360, alcanceM*math.Sqrt(r.Float64()))
	}
	return p
}

func BenchmarkDistance(b *testing.B) {
	e, _ := NewEstimator(DefaultCalibration())
	ps := paradas(1024, 200_000)
	ctx := context.Background()

	b.ResetTimer()
	var acc float64
	for i := 0; i < b.N; i++ {
		l, _ := e.Distance(ctx, ps[i&1023], ps[(i*7+3)&1023])
		acc += l.Meters
	}
	sink = acc
}

// Uma matriz do tamanho de um dia de roteirizacao: 500 paradas.
func BenchmarkMatrix500(b *testing.B) {
	e, _ := NewEstimator(DefaultCalibration())
	ps := paradas(500, 200_000)
	ctx := context.Background()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		m, err := e.Matrix(ctx, ps, ps)
		if err != nil {
			b.Fatal(err)
		}
		sinkMatrix = m
	}
}

// A comparacao que justifica o cache: a mesma matriz pedida de novo.
//
// Numa operacao real e o caso comum -- as paradas do dia mudam pouco, os
// depositos nao mudam, e a roteirizacao pede a matriz varias vezes enquanto
// testa cenarios.
func BenchmarkMatrix500Cacheada(b *testing.B) {
	e, _ := NewEstimator(DefaultCalibration())
	c := NewCache(e)
	ps := paradas(500, 200_000)
	ctx := context.Background()

	if _, err := c.Matrix(ctx, ps, ps); err != nil {
		b.Fatal(err)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		m, err := c.Matrix(ctx, ps, ps)
		if err != nil {
			b.Fatal(err)
		}
		sinkMatrix = m
	}
}

func BenchmarkCacheDistanceAcerto(b *testing.B) {
	e, _ := NewEstimator(DefaultCalibration())
	c := NewCache(e)
	ps := paradas(1024, 200_000)
	ctx := context.Background()

	for i := range ps {
		if _, err := c.Distance(ctx, ps[i], ps[(i*7+3)&1023]); err != nil {
			b.Fatal(err)
		}
	}

	b.ResetTimer()
	var acc float64
	for i := 0; i < b.N; i++ {
		j := i & 1023
		l, _ := c.Distance(ctx, ps[j], ps[(j*7+3)&1023])
		acc += l.Meters
	}
	sink = acc
}

func BenchmarkCalibrate(b *testing.B) {
	r := rand.New(rand.NewSource(43))
	obs := frotaSintetica(r, 50_000, []float64{1.6, 1.4, 1.25, 1.15}, 0.1)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		cal, _, err := Calibrate(obs)
		if err != nil {
			b.Fatal(err)
		}
		sink = cal.Factor
	}
}
