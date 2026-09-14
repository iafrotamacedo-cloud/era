package dewarp

import (
	"math"
	"testing"

	"github.com/iafrotamacedo-cloud/era/documents/geom"
	"github.com/iafrotamacedo-cloud/era/documents/imgproc"
)

func TestRectifyLineErros(t *testing.T) {
	src := imgproc.NewGray(10, 10)
	if _, err := RectifyLine(src, []geom.Point{pt(0, 0)}, 10, 10, 5, 5); err == nil {
		t.Error("baseline com 1 ponto deveria falhar")
	}
	if _, err := RectifyLine(src, nil, 10, 10, 5, 5); err == nil {
		t.Error("baseline vazia deveria falhar")
	}
}

// TestRectifyLineComQuadrilateroSoDoisPontos confere que o caminho de
// fallback para grau 1 (o caso de um poligono de 4 vertices, so 2 pontos de
// baseline -- o mesmo caso que MeasureLine trata sem erro) tambem funciona
// aqui: nao deveria dar erro, so endireitar sem curvatura nenhuma.
func TestRectifyLineComQuadrilateroSoDoisPontos(t *testing.T) {
	src := imgproc.NewGray(60, 40)
	for i := range src.Pix {
		src.Pix[i] = 0.5
	}
	baseline := []geom.Point{pt(5, 20), pt(55, 20)}
	out, err := RectifyLine(src, baseline, 40, 11, 5, 5)
	if err != nil {
		t.Fatalf("RectifyLine: %v", err)
	}
	if out.W != 40 || out.H != 11 {
		t.Fatalf("dimensoes = %dx%d, quero 40x11", out.W, out.H)
	}
}

// TestRectifyLineEndireitaLinhaCurvaDeVerdade e o teste do marco: uma
// baseline de verdade, amostrada de uma parabola (nao a curva exata passada
// a mao, como em geom.RemapCurve -- aqui PolyFit tem que primeiro
// redescobrir a curva a partir de poucos pontos, igual um detector real
// devolveria), sobre uma imagem com uma faixa clara seguindo essa mesma
// parabola. Depois de RectifyLine, a faixa tem que sair reta.
func TestRectifyLineEndireitaLinhaCurvaDeVerdade(t *testing.T) {
	const w, h = 150, 150
	f := func(x float64) float64 { return 75 + 0.008*(x-75)*(x-75) }

	src := imgproc.NewGray(w, h)
	const sigma = 2.0
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			d := float64(y) - f(float64(x))
			src.Set(x, y, float32(math.Exp(-(d*d)/(2*sigma*sigma))))
		}
	}

	// baseline com poucos pontos, como um detector real devolveria (o
	// meio de cada palavra na linha), nao uma amostragem densa.
	baseline := []geom.Point{
		pt(15, f(15)), pt(45, f(45)), pt(75, f(75)), pt(105, f(105)), pt(135, f(135)),
	}

	const outW, outH = 100, 41
	const aboveBelow = 20.0
	out, err := RectifyLine(src, baseline, outW, outH, aboveBelow, aboveBelow)
	if err != nil {
		t.Fatalf("RectifyLine: %v", err)
	}

	meio := outH / 2
	for j := 5; j < outW-5; j++ {
		if v := out.At(j, meio); v < 0.8 {
			t.Errorf("coluna %d, linha do meio = %v, esperava >= 0.8 (faixa retificada)", j, v)
		}
		if v := out.At(j, 2); v > 0.1 {
			t.Errorf("coluna %d, perto da borda de cima = %v, esperava < 0.1", j, v)
		}
		if v := out.At(j, outH-3); v > 0.1 {
			t.Errorf("coluna %d, perto da borda de baixo = %v, esperava < 0.1", j, v)
		}
	}
}
