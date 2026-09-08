package kernel

import (
	"math/rand"
	"testing"
)

// TestDepthwiseContraReferencia confere o kernel dedicado contra a
// convolucao feita pela definicao. O caminho rapido calcula de antemao em
// que faixa de colunas a janela cai dentro da imagem, em vez de testar
// posicao por posicao -- e e exatamente ai que uma aritmetica de borda
// errada se esconde. Por isso a tabela insiste em padding, stride e
// dilatacao combinados.
func TestDepthwiseContraReferencia(t *testing.T) {
	casos := []struct {
		nome string
		p    ConvParams
	}{
		{"3x3 pad 1", ConvParams{C: 8, H: 8, W: 8, KH: 3, KW: 3, PadH: 1, PadW: 1}},
		{"3x3 sem pad", ConvParams{C: 4, H: 7, W: 7, KH: 3, KW: 3}},
		{"stride 2", ConvParams{C: 6, H: 9, W: 9, KH: 3, KW: 3, PadH: 1, PadW: 1, StrideH: 2, StrideW: 2}},
		{"stride 2 so na horizontal", ConvParams{C: 4, H: 8, W: 9, KH: 3, KW: 3, PadH: 1, PadW: 1, StrideW: 2}},
		{"stride 2 so na vertical", ConvParams{C: 4, H: 9, W: 8, KH: 3, KW: 3, PadH: 1, PadW: 1, StrideH: 2}},
		{"dilatacao 2", ConvParams{C: 3, H: 10, W: 10, KH: 3, KW: 3, DilH: 2, DilW: 2}},
		{"dilatacao com pad", ConvParams{C: 3, H: 10, W: 10, KH: 3, KW: 3, PadH: 2, PadW: 2, DilH: 2, DilW: 2}},
		{"kernel 1x1", ConvParams{C: 5, H: 6, W: 6, KH: 1, KW: 1}},
		{"kernel 5x5 pad 2", ConvParams{C: 3, H: 8, W: 8, KH: 5, KW: 5, PadH: 2, PadW: 2}},
		{"kernel 3x1", ConvParams{C: 4, H: 8, W: 6, KH: 3, KW: 1, PadH: 1}},
		{"kernel 1x3", ConvParams{C: 4, H: 6, W: 8, KH: 1, KW: 3, PadW: 1}},
		{"pad maior que o kernel", ConvParams{C: 2, H: 4, W: 4, KH: 3, KW: 3, PadH: 3, PadW: 3}},
		{"entrada 1x1 com pad", ConvParams{C: 3, H: 1, W: 1, KH: 3, KW: 3, PadH: 1, PadW: 1}},
		{"canal unico", ConvParams{C: 1, H: 5, W: 5, KH: 3, KW: 3, PadH: 1, PadW: 1}},
		{"nao quadrada, tudo junto", ConvParams{C: 6, H: 11, W: 7, KH: 3, KW: 3, PadH: 2, PadW: 1, StrideH: 2, StrideW: 3, DilH: 2}},
		{"muitos canais", ConvParams{C: 64, H: 14, W: 14, KH: 3, KW: 3, PadH: 1, PadW: 1}},
	}

	r := rand.New(rand.NewSource(31337))

	for _, c := range casos {
		t.Run(c.nome, func(t *testing.T) {
			p := c.p
			p.Groups = p.C // depthwise: um filtro por canal

			if err := p.Validate(); err != nil {
				t.Fatalf("caso invalido: %v", err)
			}

			src := randSlice(r, p.C*p.H*p.W)
			weights := randSlice(r, p.C*p.KH*p.KW)
			bias := randSlice(r, p.C)

			n := p.C * p.OutH() * p.OutW()
			got := make([]float32, n)
			want := make([]float32, n)

			if err := DepthwiseConv2D(src, weights, bias, p, got); err != nil {
				t.Fatalf("DepthwiseConv2D: %v", err)
			}
			Conv2DRef(src, weights, bias, p.C, p, want)

			for i := range want {
				if !closeEnough(got[i], want[i]) {
					spatial := p.OutH() * p.OutW()
					t.Fatalf("saida[%d] = %v, referencia diz %v (canal %d, y %d, x %d)",
						i, got[i], want[i],
						i/spatial, (i%spatial)/p.OutW(), i%p.OutW())
				}
			}
		})
	}
}

// TestConv2DDespachaParaDepthwise garante que o caminho rapido nao ficou
// desligado por engano: chamar Conv2D com Groups == C tem que dar o mesmo
// resultado que chamar DepthwiseConv2D diretamente.
func TestConv2DDespachaParaDepthwise(t *testing.T) {
	p := ConvParams{C: 8, H: 9, W: 9, KH: 3, KW: 3, PadH: 1, PadW: 1, Groups: 8}

	r := rand.New(rand.NewSource(5))
	src := randSlice(r, p.C*p.H*p.W)
	weights := randSlice(r, p.C*p.KH*p.KW)
	bias := randSlice(r, p.C)

	n := p.C * p.OutH() * p.OutW()
	viaConv2D := make([]float32, n)
	direto := make([]float32, n)

	if err := Conv2D(src, weights, bias, p.C, p, viaConv2D); err != nil {
		t.Fatalf("Conv2D: %v", err)
	}
	if err := DepthwiseConv2D(src, weights, bias, p, direto); err != nil {
		t.Fatalf("DepthwiseConv2D: %v", err)
	}

	for i := range direto {
		if viaConv2D[i] != direto[i] {
			t.Fatalf("saida[%d]: Conv2D deu %v, DepthwiseConv2D deu %v (despacho quebrado?)",
				i, viaConv2D[i], direto[i])
		}
	}
}

func TestDepthwiseSemBias(t *testing.T) {
	p := ConvParams{C: 4, H: 6, W: 6, KH: 3, KW: 3, PadH: 1, PadW: 1, Groups: 4}

	r := rand.New(rand.NewSource(11))
	src := randSlice(r, p.C*p.H*p.W)
	weights := randSlice(r, p.C*p.KH*p.KW)

	n := p.C * p.OutH() * p.OutW()
	got := make([]float32, n)
	want := make([]float32, n)

	if err := DepthwiseConv2D(src, weights, nil, p, got); err != nil {
		t.Fatalf("DepthwiseConv2D: %v", err)
	}
	Conv2DRef(src, weights, nil, p.C, p, want)

	for i := range want {
		if !closeEnough(got[i], want[i]) {
			t.Fatalf("saida[%d] = %v, quero %v", i, got[i], want[i])
		}
	}
}

// TestDepthwiseNaoDeixaLixo pega um bug que so aparece com buffer reusado:
// o kernel acumula na saida, entao ele precisa inicializa-la por completo.
// Posicoes cobertas apenas por peso zero, ou alcancadas so pelo padding,
// sao justamente as que passariam despercebidas.
func TestDepthwiseNaoDeixaLixo(t *testing.T) {
	p := ConvParams{C: 2, H: 5, W: 5, KH: 3, KW: 3, PadH: 1, PadW: 1, Groups: 2}

	src := make([]float32, p.C*p.H*p.W)
	for i := range src {
		src[i] = 1
	}
	weights := make([]float32, p.C*p.KH*p.KW) // todos zero

	dst := make([]float32, p.C*p.OutH()*p.OutW())
	for i := range dst {
		dst[i] = 999
	}

	if err := DepthwiseConv2D(src, weights, nil, p, dst); err != nil {
		t.Fatalf("DepthwiseConv2D: %v", err)
	}

	for i, v := range dst {
		if v != 0 {
			t.Fatalf("dst[%d] = %v, quero 0 (pesos zerados, mas o lixo ficou)", i, v)
		}
	}
}

func TestDepthwiseMesmoResultadoEmQualquerParalelismo(t *testing.T) {
	// A divisao e por canal, e canais nao se tocam. O resultado nao pode
	// depender de como o trabalho foi repartido.
	p := ConvParams{C: 17, H: 11, W: 13, KH: 3, KW: 3, PadH: 1, PadW: 1, Groups: 17}

	r := rand.New(rand.NewSource(777))
	src := randSlice(r, p.C*p.H*p.W)
	weights := randSlice(r, p.C*p.KH*p.KW)
	bias := randSlice(r, p.C)

	original := Threads()
	defer SetThreads(original)

	n := p.C * p.OutH() * p.OutW()

	SetThreads(1)
	serial := make([]float32, n)
	if err := DepthwiseConv2D(src, weights, bias, p, serial); err != nil {
		t.Fatalf("DepthwiseConv2D: %v", err)
	}

	for _, nt := range []int{2, 3, 8, 64} {
		SetThreads(nt)
		got := make([]float32, n)
		if err := DepthwiseConv2D(src, weights, bias, p, got); err != nil {
			t.Fatalf("DepthwiseConv2D: %v", err)
		}
		for i := range serial {
			if got[i] != serial[i] {
				t.Fatalf("%d threads: saida[%d] = %v, serial deu %v", nt, i, got[i], serial[i])
			}
		}
	}
}

func TestDepthwiseRecusaEntradaInvalida(t *testing.T) {
	p := ConvParams{C: 4, H: 6, W: 6, KH: 3, KW: 3, PadH: 1, PadW: 1, Groups: 4}
	n := p.C * p.OutH() * p.OutW()

	casos := []struct {
		nome    string
		p       ConvParams
		src     int
		weights int
		bias    []float32
		dst     int
	}{
		{"Groups != C", ConvParams{C: 4, H: 6, W: 6, KH: 3, KW: 3, PadH: 1, PadW: 1, Groups: 2}, 144, 36, nil, n},
		{"src curto", p, 3, 36, nil, n},
		{"weights curto", p, 144, 3, nil, n},
		{"bias curto", p, 144, 36, make([]float32, 2), n},
		{"dst curto", p, 144, 36, nil, 3},
		{"geometria invalida", ConvParams{C: 2, H: 2, W: 2, KH: 5, KW: 5, Groups: 2}, 8, 50, nil, 100},
	}

	for _, c := range casos {
		t.Run(c.nome, func(t *testing.T) {
			err := DepthwiseConv2D(
				make([]float32, c.src),
				make([]float32, c.weights),
				c.bias, c.p,
				make([]float32, c.dst),
			)
			if err == nil {
				t.Error("DepthwiseConv2D deveria ter devolvido erro")
			}
		})
	}
}

func TestCeilDiv(t *testing.T) {
	casos := []struct{ a, b, quero int }{
		{0, 1, 0},
		{0, 3, 0},
		{1, 1, 1},
		{3, 2, 2},
		{4, 2, 2},
		{5, 2, 3},
		{-1, 2, 0},
		{-2, 2, -1},
		{-3, 2, -1},
		{-4, 2, -2},
		{7, 3, 3},
		{-7, 3, -2},
	}

	for _, c := range casos {
		if got := ceilDiv(c.a, c.b); got != c.quero {
			t.Errorf("ceilDiv(%d, %d) = %d, quero %d", c.a, c.b, got, c.quero)
		}
	}
}

func BenchmarkDepthwise(b *testing.B) {
	casos := []struct {
		nome string
		p    ConvParams
	}{
		{"56x56x64", ConvParams{C: 64, H: 56, W: 56, KH: 3, KW: 3, PadH: 1, PadW: 1}},
		{"28x28x128", ConvParams{C: 128, H: 28, W: 28, KH: 3, KW: 3, PadH: 1, PadW: 1}},
		{"56x56x64_stride2", ConvParams{C: 64, H: 56, W: 56, KH: 3, KW: 3, PadH: 1, PadW: 1, StrideH: 2, StrideW: 2}},
		{"14x14x256", ConvParams{C: 256, H: 14, W: 14, KH: 3, KW: 3, PadH: 1, PadW: 1}},
	}

	r := rand.New(rand.NewSource(1))

	for _, c := range casos {
		b.Run(c.nome, func(b *testing.B) {
			p := c.p
			p.Groups = p.C

			src := randSlice(r, p.C*p.H*p.W)
			weights := randSlice(r, p.C*p.KH*p.KW)
			dst := make([]float32, p.C*p.OutH()*p.OutW())

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if err := DepthwiseConv2D(src, weights, nil, p, dst); err != nil {
					b.Fatal(err)
				}
			}

			macs := float64(p.C) * float64(p.KH*p.KW) * float64(p.OutH()*p.OutW())
			b.ReportMetric(2*macs/(float64(b.Elapsed().Nanoseconds())/float64(b.N)), "GFLOPS")
		})
	}
}
