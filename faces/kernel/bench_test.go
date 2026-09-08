package kernel

import (
	"fmt"
	"math/rand"
	"testing"
)

// Um matmul m x k x n executa 2*m*k*n operacoes de ponto flutuante: uma
// multiplicacao e uma soma por termo. Dividindo pelo tempo, sai o GFLOPS --
// o unico numero que diz se uma otimizacao funcionou ou se foi impressao.
func gflops(b *testing.B, m, k, n int, ns float64) {
	b.ReportMetric(2*float64(m)*float64(k)*float64(n)/ns, "GFLOPS")
}

func BenchmarkMatMul(b *testing.B) {
	r := rand.New(rand.NewSource(1))

	for _, n := range []int{128, 256, 512, 1024} {
		b.Run(fmt.Sprintf("%dx%dx%d", n, n, n), func(b *testing.B) {
			x := randSlice(r, n*n)
			y := randSlice(r, n*n)
			z := make([]float32, n*n)

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				MatMul(x, y, z, n, n, n)
			}
			gflops(b, n, n, n, float64(b.Elapsed().Nanoseconds())/float64(b.N))
		})
	}
}

// BenchmarkMatMulSerial mede um nucleo so. Comparado com BenchmarkMatMul,
// mostra o quanto o paralelismo esta realmente rendendo.
func BenchmarkMatMulSerial(b *testing.B) {
	r := rand.New(rand.NewSource(1))
	const n = 512

	original := Threads()
	SetThreads(1)
	defer SetThreads(original)

	x := randSlice(r, n*n)
	y := randSlice(r, n*n)
	z := make([]float32, n*n)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		MatMul(x, y, z, n, n, n)
	}
	gflops(b, n, n, n, float64(b.Elapsed().Nanoseconds())/float64(b.N))
}

// BenchmarkMatMulRef mede a implementacao ingenua. A razao entre este
// numero e o do BenchmarkMatMul e o ganho do bloqueio mais paralelismo.
func BenchmarkMatMulRef(b *testing.B) {
	r := rand.New(rand.NewSource(1))
	const n = 512

	x := randSlice(r, n*n)
	y := randSlice(r, n*n)
	z := make([]float32, n*n)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		MatMulRef(x, y, z, n, n, n)
	}
	gflops(b, n, n, n, float64(b.Elapsed().Nanoseconds())/float64(b.N))
}

func BenchmarkIm2Col(b *testing.B) {
	r := rand.New(rand.NewSource(1))
	p := ConvParams{C: 64, H: 56, W: 56, KH: 3, KW: 3, PadH: 1, PadW: 1}

	src := randSlice(r, p.C*p.H*p.W)
	dst := make([]float32, p.ColSize())

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Im2Col(src, p, dst)
	}
}

// BenchmarkConv2D usa camadas com a cara das que aparecem numa rede de
// reconhecimento facial de verdade.
func BenchmarkConv2D(b *testing.B) {
	casos := []struct {
		nome string
		outC int
		p    ConvParams
	}{
		{"entrada_112x112x3", 64, ConvParams{C: 3, H: 112, W: 112, KH: 3, KW: 3, PadH: 1, PadW: 1, StrideH: 2, StrideW: 2}},
		{"depthwise_56x56x64", 64, ConvParams{C: 64, H: 56, W: 56, KH: 3, KW: 3, PadH: 1, PadW: 1, Groups: 64}},
		{"pontual_1x1_56x56", 128, ConvParams{C: 64, H: 56, W: 56, KH: 1, KW: 1}},
		{"meio_da_rede_14x14", 256, ConvParams{C: 128, H: 14, W: 14, KH: 3, KW: 3, PadH: 1, PadW: 1}},
	}

	r := rand.New(rand.NewSource(1))

	for _, c := range casos {
		b.Run(c.nome, func(b *testing.B) {
			p := c.p
			groups := max(p.Groups, 1)

			src := randSlice(r, p.C*p.H*p.W)
			weights := randSlice(r, c.outC*(p.C/groups)*p.KH*p.KW)
			dst := make([]float32, c.outC*p.OutH()*p.OutW())

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if err := Conv2D(src, weights, nil, c.outC, p, dst); err != nil {
					b.Fatal(err)
				}
			}

			// MACs da convolucao, contando 2 flops por MAC.
			macs := float64(c.outC) * float64(p.C/groups) * float64(p.KH*p.KW) *
				float64(p.OutH()*p.OutW())
			b.ReportMetric(2*macs/(float64(b.Elapsed().Nanoseconds())/float64(b.N)), "GFLOPS")
		})
	}
}
