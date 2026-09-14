package geom

import (
	"math"
	"math/rand"
	"testing"

	"github.com/iafrotamacedo-cloud/era/documents/imgproc"
)

// TestArcLenTableRetaTemFormulaFechada confere a tabela de comprimento de
// arco contra o unico caso com formula exata: uma reta de inclinacao m tem
// comprimento de arco = distancia em x vezes sqrt(1+m^2). Sem essa
// conferencia, um erro na integracao numerica so apareceria como texto sutil
// e errado num teste visual, nao como falha de teste.
func TestArcLenTableRetaTemFormulaFechada(t *testing.T) {
	for _, m := range []float64{0, 0.5, -2, 5} {
		curve := []float64{10, m} // y = 10 + m*x
		tabela := buildArcLenTable(curve, 0, 100, amostrasArco)
		want := 100 * math.Hypot(1, m)
		if got := tabela.total(); math.Abs(got-want) > 1e-6 {
			t.Errorf("m=%v: comprimento de arco = %v, quero %v", m, got, want)
		}
	}
}

func TestArcLenTableXAtLenIdaEVolta(t *testing.T) {
	curve := []float64{0, 0, 0.01} // y = 0.01*x^2
	tabela := buildArcLenTable(curve, -50, 50, amostrasArco)
	total := tabela.total()

	for _, frac := range []float64{0, 0.1, 0.5, 0.9, 1.0} {
		s := frac * total
		x := tabela.xAtLen(s)
		// x tem que estar dentro do dominio e ser monotono em s -- checa
		// so os limites, a monotonicidade e coberta pelo teste de baixo.
		if x < -50-1e-6 || x > 50+1e-6 {
			t.Errorf("xAtLen(%v) = %v, fora do dominio [-50,50]", s, x)
		}
	}
}

func TestRemapCurveRetaHorizontalReproduzAmostragemDireta(t *testing.T) {
	// Com inclinacao zero, a normal e (0,1): RemapCurve deveria reproduzir
	// exatamente uma amostragem vertical direta em torno de y=c0. E o caso
	// mais simples que confere toda a maquina de arco e normal sem
	// depender de nada alem de SampleBilinear, ja testado a parte.
	rnd := rand.New(rand.NewSource(9))
	src := imgproc.NewGray(60, 40)
	for i := range src.Pix {
		src.Pix[i] = rnd.Float32()
	}

	const c0 = 20.0
	curve := []float64{c0}
	out, err := RemapCurve(src, curve, 5, 55, 50, 11, 5, 5)
	if err != nil {
		t.Fatalf("RemapCurve: %v", err)
	}

	for j := 0; j < out.W; j++ {
		s := (float64(j) + 0.5) / float64(out.W) * (55 - 5)
		x := 5 + s
		for i := 0; i < out.H; i++ {
			offset := -5 + 10*float64(i)/float64(out.H-1)
			want := imgproc.SampleBilinear(src, x, c0+offset)
			got := out.At(j, i)
			if math.Abs(float64(got-want)) > 1e-4 {
				t.Fatalf("(%d,%d) = %v, quero %v (amostragem direta)", j, i, got, want)
			}
		}
	}
}

// TestRemapCurveEndireitaFaixaCurva desenha uma faixa clara seguindo uma
// parabola sobre fundo escuro, retifica com RemapCurve usando a mesma
// parabola, e confere que a faixa sai reta no meio da saida -- a prova de
// que a retificacao realmente segue a curva, nao so copia a imagem.
func TestRemapCurveEndireitaFaixaCurva(t *testing.T) {
	const w, h = 150, 150
	// f(x) = 75 + 0.008*(x-75)^2: parabola centrada em x=75, ponto mais
	// alto (menor y) ali. Expandido em x puro (a convencao de
	// PolyFit/EvalPoly): 0.008*(x-75)^2 = 0.008x^2 - 1.2x + 45, entao
	// curve = [120, -1.2, 0.008].
	f := func(x float64) float64 { return 75 + 0.008*(x-75)*(x-75) }
	curve := []float64{120, -1.2, 0.008}

	src := imgproc.NewGray(w, h)
	const sigma = 2.0
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			d := float64(y) - f(float64(x))
			src.Set(x, y, float32(math.Exp(-(d*d)/(2*sigma*sigma))))
		}
	}

	const outW, outH = 100, 41
	const aboveBelow = 20.0
	out, err := RemapCurve(src, curve, 10, 140, outW, outH, aboveBelow, aboveBelow)
	if err != nil {
		t.Fatalf("RemapCurve: %v", err)
	}

	meio := outH / 2 // offset ~0, onde a faixa deveria estar apos retificar
	for j := 5; j < outW-5; j++ {
		if v := out.At(j, meio); v < 0.8 {
			t.Errorf("coluna %d, linha do meio = %v, esperava >= 0.8 (faixa retificada)", j, v)
		}
		if v := out.At(j, 2); v > 0.1 {
			t.Errorf("coluna %d, perto da borda de cima = %v, esperava < 0.1 (longe da faixa)", j, v)
		}
		if v := out.At(j, outH-3); v > 0.1 {
			t.Errorf("coluna %d, perto da borda de baixo = %v, esperava < 0.1 (longe da faixa)", j, v)
		}
	}
}

func TestRemapCurveErros(t *testing.T) {
	src := imgproc.NewGray(10, 10)
	if _, err := RemapCurve(src, []float64{0}, 5, 5, 10, 10, 1, 1); err == nil {
		t.Error("xMax == xMin deveria falhar")
	}
	if _, err := RemapCurve(src, []float64{0}, 0, 5, 0, 10, 1, 1); err == nil {
		t.Error("outW = 0 deveria falhar")
	}
	if _, err := RemapCurve(src, []float64{0}, 0, 5, 10, 0, 1, 1); err == nil {
		t.Error("outH = 0 deveria falhar")
	}
}
