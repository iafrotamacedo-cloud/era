package geom

import (
	"math"
	"math/rand"
	"testing"
)

// TestPolyFitRecuperaPolinomioExato gera pontos de um polinomio conhecido,
// sem ruido nenhum, e confere que PolyFit recupera os coeficientes exatos.
// E o teste mais forte possivel para um ajuste: sem ruido, o erro
// quadratico minimo e zero, entao qualquer desvio e bug.
func TestPolyFitRecuperaPolinomioExato(t *testing.T) {
	// y = 2 + 3x - x^2
	want := []float64{2, 3, -1}
	var pts []Point
	for x := -5.0; x <= 5.0; x++ {
		y := want[0] + want[1]*x + want[2]*x*x
		pts = append(pts, Point{X: x, Y: y})
	}

	got, err := PolyFit(pts, 2)
	if err != nil {
		t.Fatalf("PolyFit: %v", err)
	}
	for i := range want {
		if math.Abs(got[i]-want[i]) > 1e-6 {
			t.Errorf("coeffs[%d] = %v, quero %v", i, got[i], want[i])
		}
	}
}

func TestPolyFitReta(t *testing.T) {
	// y = 1 + 2x, pontos exatos -- caso de grau 1, o mais simples.
	pts := []Point{{0, 1}, {1, 3}, {2, 5}, {3, 7}, {4, 9}}
	got, err := PolyFit(pts, 1)
	if err != nil {
		t.Fatalf("PolyFit: %v", err)
	}
	if math.Abs(got[0]-1) > 1e-9 || math.Abs(got[1]-2) > 1e-9 {
		t.Errorf("coeffs = %v, quero [1, 2]", got)
	}
}

// TestPolyFitMinimizaErroComRuido confere a propriedade de minimos
// quadrados sem exigir recuperacao exata: com ruido gaussiano em torno de
// uma reta conhecida, o coeficiente ajustado tem que ficar proximo do
// verdadeiro, e o residuo do ajuste tem que ser menor que o residuo de
// qualquer reta horizontal (a media) -- caso contrario o ajuste nao estaria
// ajustando nada.
func TestPolyFitMinimizaErroComRuido(t *testing.T) {
	rnd := rand.New(rand.NewSource(11))
	const a, b = 5.0, -2.0 // y = 5 - 2x
	var pts []Point
	for x := 0.0; x < 50; x++ {
		y := a + b*x + (rnd.Float64()-0.5)*0.5
		pts = append(pts, Point{X: x, Y: y})
	}

	got, err := PolyFit(pts, 1)
	if err != nil {
		t.Fatalf("PolyFit: %v", err)
	}
	if math.Abs(got[0]-a) > 0.5 || math.Abs(got[1]-b) > 0.05 {
		t.Errorf("coeffs = %v, quero perto de [%v, %v]", got, a, b)
	}

	var residuoAjuste, residuoMedia, mediaY float64
	for _, p := range pts {
		mediaY += p.Y
	}
	mediaY /= float64(len(pts))
	for _, p := range pts {
		e := p.Y - EvalPoly(got, p.X)
		residuoAjuste += e * e
		em := p.Y - mediaY
		residuoMedia += em * em
	}
	if residuoAjuste >= residuoMedia {
		t.Errorf("residuo do ajuste (%v) deveria ser bem menor que o da media (%v)", residuoAjuste, residuoMedia)
	}
}

func TestPolyFitErros(t *testing.T) {
	if _, err := PolyFit([]Point{{0, 0}}, -1); err == nil {
		t.Error("grau negativo deveria falhar")
	}
	if _, err := PolyFit([]Point{{0, 0}, {1, 1}}, 2); err == nil {
		t.Error("pontos insuficientes para o grau deveria falhar")
	}
}

func TestEvalPolyConstante(t *testing.T) {
	if got := EvalPoly([]float64{7}, 100); got != 7 {
		t.Errorf("EvalPoly constante = %v, quero 7", got)
	}
}
