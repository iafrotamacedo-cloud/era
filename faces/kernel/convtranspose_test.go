package kernel

import (
	"math/rand"
	"reflect"
	"testing"
)

// TestConvTransposeConhecido confere o caso real que motivou esta op: um
// upsample aprendido 2x -- kernel 2x2, stride 2, sem padding -- com um
// exemplo pequeno o bastante para calcular a mao.
//
// Entrada 1 canal 2x2:
//
//	1 2
//	3 4
//
// Kernel 1->1 canal, 2x2, todo peso 1: cada entrada "espalha" seu valor
// pelos 4 pixels do bloco 2x2 correspondente na saida (sem sobreposicao,
// porque stride == kernel), entao a saida 4x4 e cada valor repetido num
// bloco 2x2:
//
//	1 1 2 2
//	1 1 2 2
//	3 3 4 4
//	3 3 4 4
func TestConvTransposeConhecido(t *testing.T) {
	src := []float32{1, 2, 3, 4}
	weights := []float32{1, 1, 1, 1} // [C=1, outC/G=1, KH=2, KW=2]
	p := ConvTransposeParams{C: 1, H: 2, W: 2, KH: 2, KW: 2, StrideH: 2, StrideW: 2}

	if got := p.OutH(); got != 4 {
		t.Fatalf("OutH = %d, quero 4", got)
	}

	want := []float32{
		1, 1, 2, 2,
		1, 1, 2, 2,
		3, 3, 4, 4,
		3, 3, 4, 4,
	}

	got := make([]float32, 16)
	if err := ConvTranspose2D(src, weights, nil, 1, p, got); err != nil {
		t.Fatalf("ConvTranspose2D: %v", err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("saida = %v, quero %v", got, want)
	}

	// A referencia (que distribui, em vez de reunir) tem que bater com a
	// mesma conta feita a mao.
	gotRef := make([]float32, 16)
	ConvTranspose2DRef(src, weights, nil, 1, p, gotRef)
	if !reflect.DeepEqual(gotRef, want) {
		t.Errorf("referencia = %v, quero %v", gotRef, want)
	}
}

func TestConvTransposeComVies(t *testing.T) {
	src := []float32{1, 2, 3, 4}
	weights := []float32{1, 1, 1, 1}
	bias := []float32{10}
	p := ConvTransposeParams{C: 1, H: 2, W: 2, KH: 2, KW: 2, StrideH: 2, StrideW: 2}

	got := make([]float32, 16)
	if err := ConvTranspose2D(src, weights, bias, 1, p, got); err != nil {
		t.Fatalf("ConvTranspose2D: %v", err)
	}
	want := []float32{
		11, 11, 12, 12,
		11, 11, 12, 12,
		13, 13, 14, 14,
		13, 13, 14, 14,
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("saida = %v, quero %v", got, want)
	}
}

// TestConvTransposeContraReferencia varre geometria (stride, padding,
// dilatacao, grupos, canais multiplos) e confere que as duas
// implementacoes -- uma que reune, outra que distribui -- concordam.
func TestConvTransposeContraReferencia(t *testing.T) {
	casos := []struct {
		nome string
		p    ConvTransposeParams
	}{
		{"kernel 2x2 stride 2, o caso real", ConvTransposeParams{C: 4, H: 5, W: 5, KH: 2, KW: 2, StrideH: 2, StrideW: 2}},
		{"stride 1, equivale a convolucao com kernel espelhado", ConvTransposeParams{C: 3, H: 6, W: 6, KH: 3, KW: 3}},
		{"com padding", ConvTransposeParams{C: 3, H: 6, W: 6, KH: 3, KW: 3, PadH: 1, PadW: 1}},
		{"stride 2 com padding", ConvTransposeParams{C: 3, H: 5, W: 5, KH: 3, KW: 3, StrideH: 2, StrideW: 2, PadH: 1, PadW: 1}},
		{"stride assimetrico", ConvTransposeParams{C: 2, H: 4, W: 6, KH: 3, KW: 3, StrideH: 2, StrideW: 3}},
		{"dilatacao 2", ConvTransposeParams{C: 2, H: 5, W: 5, KH: 3, KW: 3, DilH: 2, DilW: 2}},
		{"kernel 1x1", ConvTransposeParams{C: 4, H: 4, W: 4, KH: 1, KW: 1}},
		{"grupos", ConvTransposeParams{C: 4, H: 4, W: 4, KH: 2, KW: 2, StrideH: 2, StrideW: 2, Groups: 2}},
		{"nao quadrada, tudo junto", ConvTransposeParams{C: 6, H: 7, W: 5, KH: 3, KW: 3, PadH: 1, PadW: 2, StrideH: 2, StrideW: 1, DilH: 2, Groups: 3}},
		{"canal unico", ConvTransposeParams{C: 1, H: 3, W: 3, KH: 2, KW: 2, StrideH: 2, StrideW: 2}},
	}

	r := rand.New(rand.NewSource(271828))
	outCPorGrupo := 3

	for _, c := range casos {
		t.Run(c.nome, func(t *testing.T) {
			p := c.p
			if err := p.Validate(); err != nil {
				t.Fatalf("caso invalido: %v", err)
			}
			outC := outCPorGrupo * p.norm().Groups

			src := randSlice(r, p.C*p.H*p.W)
			weights := randSlice(r, p.C*(outC/p.norm().Groups)*p.KH*p.KW)
			bias := randSlice(r, outC)

			n := outC * p.OutH() * p.OutW()
			got := make([]float32, n)
			want := make([]float32, n)

			if err := ConvTranspose2D(src, weights, bias, outC, p, got); err != nil {
				t.Fatalf("ConvTranspose2D: %v", err)
			}
			ConvTranspose2DRef(src, weights, bias, outC, p, want)

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

func TestConvTransposeSemBias(t *testing.T) {
	src := []float32{1, 1, 1, 1}
	weights := []float32{2, 2, 2, 2}
	p := ConvTransposeParams{C: 1, H: 2, W: 2, KH: 2, KW: 2, StrideH: 2, StrideW: 2}

	got := make([]float32, 16)
	if err := ConvTranspose2D(src, weights, nil, 1, p, got); err != nil {
		t.Fatalf("ConvTranspose2D: %v", err)
	}
	for i, v := range got {
		if v != 2 {
			t.Fatalf("got[%d] = %v, quero 2 (sem bias)", i, v)
		}
	}
}

func TestConvTransposeRecusaEntradaInvalida(t *testing.T) {
	p := ConvTransposeParams{C: 2, H: 4, W: 4, KH: 2, KW: 2, StrideH: 2, StrideW: 2}
	dst := make([]float32, 2*8*8)

	casos := []struct {
		nome   string
		src, w []float32
		bias   []float32
		outC   int
		params ConvTransposeParams
	}{
		{"src curto", make([]float32, 4), make([]float32, 2*3*2*2), nil, 3, p},
		{"weights curto", make([]float32, 32), make([]float32, 2), nil, 3, p},
		{"bias curto", make([]float32, 32), make([]float32, 2*3*2*2), make([]float32, 1), 3, p},
		{"canais nao dividem em grupos", make([]float32, 32), make([]float32, 32), nil, 3, ConvTransposeParams{C: 3, H: 4, W: 4, KH: 2, KW: 2, Groups: 2}},
	}
	for _, c := range casos {
		t.Run(c.nome, func(t *testing.T) {
			if err := ConvTranspose2D(c.src, c.w, c.bias, c.outC, c.params, dst); err == nil {
				t.Error("deveria falhar")
			}
		})
	}
}

func TestConvTransposeParamsValidate(t *testing.T) {
	casos := []struct {
		nome string
		p    ConvTransposeParams
		ok   bool
	}{
		{"valido", ConvTransposeParams{C: 2, H: 4, W: 4, KH: 2, KW: 2, StrideH: 2, StrideW: 2}, true},
		{"canal zero", ConvTransposeParams{C: 0, H: 4, W: 4, KH: 2, KW: 2}, false},
		{"kernel zero", ConvTransposeParams{C: 2, H: 4, W: 4, KH: 0, KW: 2}, false},
		{"stride negativo", ConvTransposeParams{C: 2, H: 4, W: 4, KH: 2, KW: 2, StrideH: -1}, false},
		{"padding negativo", ConvTransposeParams{C: 2, H: 4, W: 4, KH: 2, KW: 2, PadH: -1}, false},
		{"grupos nao dividem", ConvTransposeParams{C: 3, H: 4, W: 4, KH: 2, KW: 2, Groups: 2}, false},
	}
	for _, c := range casos {
		t.Run(c.nome, func(t *testing.T) {
			err := c.p.Validate()
			if c.ok && err != nil {
				t.Errorf("deveria ser valido, deu erro: %v", err)
			}
			if !c.ok && err == nil {
				t.Error("deveria dar erro")
			}
		})
	}
}
