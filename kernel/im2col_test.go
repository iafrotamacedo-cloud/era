package kernel

import (
	"fmt"
	"math/rand"
	"reflect"
	"testing"
)

func TestIm2ColConhecido(t *testing.T) {
	// Entrada 1 canal, 3x3:
	//   1 2 3
	//   4 5 6
	//   7 8 9
	// Kernel 2x2, stride 1, sem padding -> saida 2x2, ou seja 4 janelas.
	src := []float32{1, 2, 3, 4, 5, 6, 7, 8, 9}
	p := ConvParams{C: 1, H: 3, W: 3, KH: 2, KW: 2}

	if got := p.OutH(); got != 2 {
		t.Fatalf("OutH = %d, quero 2", got)
	}
	if got := p.ColSize(); got != 16 {
		t.Fatalf("ColSize = %d, quero 16 (4 linhas x 4 janelas)", got)
	}

	dst := make([]float32, p.ColSize())
	Im2Col(src, p, dst)

	// Cada LINHA e uma posicao dentro do kernel; cada COLUNA, uma janela.
	//   linha 0 = canto superior esquerdo de cada janela
	//   linha 1 = canto superior direito
	//   linha 2 = canto inferior esquerdo
	//   linha 3 = canto inferior direito
	want := []float32{
		1, 2, 4, 5,
		2, 3, 5, 6,
		4, 5, 7, 8,
		5, 6, 8, 9,
	}
	if !reflect.DeepEqual(dst, want) {
		t.Errorf("Im2Col =\n%v\nquero\n%v", dst, want)
	}
}

func TestIm2ColPaddingViraZero(t *testing.T) {
	src := []float32{5} // 1x1x1
	p := ConvParams{C: 1, H: 1, W: 1, KH: 3, KW: 3, PadH: 1, PadW: 1}

	if got := p.OutH(); got != 1 {
		t.Fatalf("OutH = %d, quero 1", got)
	}

	dst := make([]float32, p.ColSize())
	Im2Col(src, p, dst)

	// So o centro do kernel 3x3 pega o pixel; as 8 bordas caem no padding.
	want := []float32{0, 0, 0, 0, 5, 0, 0, 0, 0}
	if !reflect.DeepEqual(dst, want) {
		t.Errorf("Im2Col = %v, quero %v", dst, want)
	}
}

func TestIm2ColNaoDeixaLixo(t *testing.T) {
	// Um buffer reusado precisa ser sobrescrito por inteiro, inclusive nas
	// posicoes que caem no padding.
	src := []float32{5}
	p := ConvParams{C: 1, H: 1, W: 1, KH: 3, KW: 3, PadH: 1, PadW: 1}

	dst := make([]float32, p.ColSize())
	for i := range dst {
		dst[i] = 999
	}
	Im2Col(src, p, dst)

	for i, v := range dst {
		if v == 999 {
			t.Fatalf("dst[%d] ficou com lixo do buffer anterior", i)
		}
	}
}

func TestConvParamsGeometria(t *testing.T) {
	casos := []struct {
		nome       string
		p          ConvParams
		outH, outW int
	}{
		{"1x1", ConvParams{C: 1, H: 8, W: 8, KH: 1, KW: 1}, 8, 8},
		{"3x3 com pad 1 preserva", ConvParams{C: 1, H: 8, W: 8, KH: 3, KW: 3, PadH: 1, PadW: 1}, 8, 8},
		{"3x3 sem pad encolhe 2", ConvParams{C: 1, H: 8, W: 8, KH: 3, KW: 3}, 6, 6},
		{"stride 2 divide", ConvParams{C: 1, H: 8, W: 8, KH: 3, KW: 3, PadH: 1, PadW: 1, StrideH: 2, StrideW: 2}, 4, 4},
		{"dilatacao 2 alarga o kernel", ConvParams{C: 1, H: 8, W: 8, KH: 3, KW: 3, DilH: 2, DilW: 2}, 4, 4},
		{"nao quadrado", ConvParams{C: 1, H: 10, W: 6, KH: 3, KW: 1}, 8, 6},
	}

	for _, c := range casos {
		t.Run(c.nome, func(t *testing.T) {
			if got := c.p.OutH(); got != c.outH {
				t.Errorf("OutH = %d, quero %d", got, c.outH)
			}
			if got := c.p.OutW(); got != c.outW {
				t.Errorf("OutW = %d, quero %d", got, c.outW)
			}
			if err := c.p.Validate(); err != nil {
				t.Errorf("Validate: %v", err)
			}
		})
	}
}

func TestConvParamsValidateRecusaLixo(t *testing.T) {
	casos := []struct {
		nome string
		p    ConvParams
	}{
		{"sem canais", ConvParams{C: 0, H: 8, W: 8, KH: 3, KW: 3}},
		{"kernel zero", ConvParams{C: 1, H: 8, W: 8, KH: 0, KW: 3}},
		{"padding negativo", ConvParams{C: 1, H: 8, W: 8, KH: 3, KW: 3, PadH: -1}},
		{"grupos nao dividem", ConvParams{C: 3, H: 8, W: 8, KH: 3, KW: 3, Groups: 2}},
		{"kernel maior que a entrada", ConvParams{C: 1, H: 2, W: 2, KH: 5, KW: 5}},
	}

	for _, c := range casos {
		t.Run(c.nome, func(t *testing.T) {
			if err := c.p.Validate(); err == nil {
				t.Error("Validate deveria ter recusado")
			}
		})
	}
}

// TestConv2DContraReferencia e o teste que fecha a Fase 1: prova que
// Im2Col + MatMul produzem exatamente a mesma coisa que a convolucao feita
// pela definicao, deslizando o kernel posicao a posicao.
func TestConv2DContraReferencia(t *testing.T) {
	casos := []struct {
		nome string
		outC int
		p    ConvParams
	}{
		{"1x1 troca de canais", 8, ConvParams{C: 4, H: 6, W: 6, KH: 1, KW: 1}},
		{"3x3 classica", 5, ConvParams{C: 3, H: 8, W: 8, KH: 3, KW: 3, PadH: 1, PadW: 1}},
		{"3x3 sem padding", 2, ConvParams{C: 2, H: 7, W: 7, KH: 3, KW: 3}},
		{"stride 2", 6, ConvParams{C: 3, H: 9, W: 9, KH: 3, KW: 3, PadH: 1, PadW: 1, StrideH: 2, StrideW: 2}},
		{"dilatacao 2", 4, ConvParams{C: 2, H: 9, W: 9, KH: 3, KW: 3, DilH: 2, DilW: 2}},
		{"kernel nao quadrado", 3, ConvParams{C: 2, H: 8, W: 6, KH: 3, KW: 1, PadH: 1}},
		{"depthwise", 6, ConvParams{C: 6, H: 8, W: 8, KH: 3, KW: 3, PadH: 1, PadW: 1, Groups: 6}},
		{"agrupada em 2", 8, ConvParams{C: 4, H: 6, W: 6, KH: 3, KW: 3, PadH: 1, PadW: 1, Groups: 2}},
		{"canal unico", 1, ConvParams{C: 1, H: 5, W: 5, KH: 3, KW: 3, PadH: 1, PadW: 1}},
		{"stride e padding assimetricos", 4, ConvParams{C: 2, H: 10, W: 7, KH: 3, KW: 3, PadH: 2, PadW: 1, StrideH: 3, StrideW: 2}},
	}

	r := rand.New(rand.NewSource(2024))

	for _, c := range casos {
		t.Run(c.nome, func(t *testing.T) {
			p := c.p
			if err := p.Validate(); err != nil {
				t.Fatalf("caso invalido: %v", err)
			}

			groups := max(p.Groups, 1)
			src := randSlice(r, p.C*p.H*p.W)
			weights := randSlice(r, c.outC*(p.C/groups)*p.KH*p.KW)
			bias := randSlice(r, c.outC)

			n := c.outC * p.OutH() * p.OutW()
			got := make([]float32, n)
			want := make([]float32, n)

			if err := Conv2D(src, weights, bias, c.outC, p, got); err != nil {
				t.Fatalf("Conv2D: %v", err)
			}
			Conv2DRef(src, weights, bias, c.outC, p, want)

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

func TestConv2DSemBias(t *testing.T) {
	p := ConvParams{C: 2, H: 5, W: 5, KH: 3, KW: 3, PadH: 1, PadW: 1}
	const outC = 3

	r := rand.New(rand.NewSource(1))
	src := randSlice(r, p.C*p.H*p.W)
	weights := randSlice(r, outC*p.C*p.KH*p.KW)

	n := outC * p.OutH() * p.OutW()
	got := make([]float32, n)
	want := make([]float32, n)

	if err := Conv2D(src, weights, nil, outC, p, got); err != nil {
		t.Fatalf("Conv2D: %v", err)
	}
	Conv2DRef(src, weights, nil, outC, p, want)

	for i := range want {
		if !closeEnough(got[i], want[i]) {
			t.Fatalf("saida[%d] = %v, quero %v", i, got[i], want[i])
		}
	}
}

func TestConv2DRecusaEntradaInvalida(t *testing.T) {
	p := ConvParams{C: 4, H: 6, W: 6, KH: 3, KW: 3, PadH: 1, PadW: 1, Groups: 2}
	spatial := p.OutH() * p.OutW()

	casos := []struct {
		nome    string
		outC    int
		weights int
		bias    []float32
		dst     int
	}{
		{"outC nao divide nos grupos", 3, 999, nil, 999},
		{"weights curto", 4, 5, nil, 4 * spatial},
		{"bias curto", 4, 4 * 2 * 9, make([]float32, 2), 4 * spatial},
		{"dst curto", 4, 4 * 2 * 9, nil, 3},
	}

	src := make([]float32, p.C*p.H*p.W)
	for _, c := range casos {
		t.Run(c.nome, func(t *testing.T) {
			err := Conv2D(src, make([]float32, c.weights), c.bias, c.outC, p, make([]float32, c.dst))
			if err == nil {
				t.Error("Conv2D deveria ter devolvido erro")
			}
		})
	}
}

// TestConv2DIdentidade confere um caso em que da para prever a saida de
// cabeca: um kernel 1x1 de peso 1 por canal copia a entrada.
func TestConv2DIdentidade(t *testing.T) {
	p := ConvParams{C: 3, H: 4, W: 4, KH: 1, KW: 1}
	const outC = 3

	src := make([]float32, p.C*p.H*p.W)
	for i := range src {
		src[i] = float32(i)
	}

	// weights[oc][ic] = 1 se oc == ic, senao 0
	weights := make([]float32, outC*p.C)
	for oc := 0; oc < outC; oc++ {
		weights[oc*p.C+oc] = 1
	}

	got := make([]float32, outC*p.OutH()*p.OutW())
	if err := Conv2D(src, weights, nil, outC, p, got); err != nil {
		t.Fatalf("Conv2D: %v", err)
	}

	if !reflect.DeepEqual(got, src) {
		t.Errorf("convolucao identidade mudou a entrada:\n%v\nquero\n%v", got, src)
	}
}

func ExampleConv2D() {
	// Detector de borda vertical num quadrado 4x4 partido ao meio.
	src := []float32{
		0, 0, 9, 9,
		0, 0, 9, 9,
		0, 0, 9, 9,
		0, 0, 9, 9,
	}
	p := ConvParams{C: 1, H: 4, W: 4, KH: 1, KW: 2}
	weights := []float32{-1, 1} // diferenca entre vizinhos horizontais

	dst := make([]float32, 1*p.OutH()*p.OutW())
	if err := Conv2D(src, weights, nil, 1, p, dst); err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(dst)
	// Output: [0 9 0 0 9 0 0 9 0 0 9 0]
}
