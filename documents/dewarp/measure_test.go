package dewarp

import (
	"math"
	"testing"

	"github.com/iafrotamacedo-cloud/era/documents/geom"
)

func TestMeasureLineReta(t *testing.T) {
	baseline := []geom.Point{pt(0, 5), pt(50, 5), pt(100, 5)}
	got, err := MeasureLine(baseline)
	if err != nil {
		t.Fatalf("MeasureLine: %v", err)
	}
	if math.Abs(got.Angle) > 1e-9 {
		t.Errorf("Angle = %v, quero 0 (linha horizontal)", got.Angle)
	}
	if got.LinearBow > 1e-9 {
		t.Errorf("LinearBow = %v, quero 0 (pontos exatamente na reta)", got.LinearBow)
	}
	if got.CurveBow > 1e-9 {
		t.Errorf("CurveBow = %v, quero 0", got.CurveBow)
	}
}

func TestMeasureLineInclinada(t *testing.T) {
	// y = 0.2*x: inclinacao de atan(0.2) ~ 11.3 graus.
	var baseline []geom.Point
	for x := 0.0; x <= 100; x += 10 {
		baseline = append(baseline, geom.Point{X: x, Y: 0.2 * x})
	}
	got, err := MeasureLine(baseline)
	if err != nil {
		t.Fatalf("MeasureLine: %v", err)
	}
	wantAngle := math.Atan(0.2)
	if math.Abs(got.Angle-wantAngle) > 1e-6 {
		t.Errorf("Angle = %v, quero %v", got.Angle, wantAngle)
	}
	if got.LinearBow > 1e-6 {
		t.Errorf("LinearBow = %v, quero ~0 (reta exata, so inclinada)", got.LinearBow)
	}
}

func TestMeasureLineSoDoisPontosCaiParaReta(t *testing.T) {
	// o caso de um quadrilatero simples: so 2 pontos de baseline, sem
	// informacao nenhuma de curvatura.
	baseline := []geom.Point{pt(0, 10), pt(100, 14)}
	got, err := MeasureLine(baseline)
	if err != nil {
		t.Fatalf("MeasureLine: %v", err)
	}
	if got.LinearBow != 0 {
		t.Errorf("LinearBow com 2 pontos = %v, quero 0 (reta exata por definicao)", got.LinearBow)
	}
	if got.CurveBow != got.LinearBow {
		t.Errorf("CurveBow = %v, quero igual a LinearBow (%v) -- sem pontos para ajustar curva", got.CurveBow, got.LinearBow)
	}
}

// TestMeasureLineCurvaSuave usa uma parabola exata como baseline: a reta
// ajustada tem que sobrar um residuo grande (a linha claramente nao e
// reta), mas a curva de grau 2 recupera a parabola exatamente, entao o
// residuo dela tem que ficar perto de zero. E a assinatura numerica de N2.
func TestMeasureLineCurvaSuave(t *testing.T) {
	var baseline []geom.Point
	for x := 0.0; x <= 100; x += 10 {
		y := 0.01 * (x - 50) * (x - 50) // parabola, sagita de 25px
		baseline = append(baseline, geom.Point{X: x, Y: y})
	}
	got, err := MeasureLine(baseline)
	if err != nil {
		t.Fatalf("MeasureLine: %v", err)
	}
	if got.LinearBow < 5 {
		t.Errorf("LinearBow = %v, esperava bem maior que 0 (parabola nao e reta)", got.LinearBow)
	}
	if got.CurveBow > 1e-6 {
		t.Errorf("CurveBow = %v, quero ~0 (curva de grau 2 recupera a parabola exata)", got.CurveBow)
	}
	if got.CurveBow >= got.LinearBow {
		t.Errorf("CurveBow (%v) deveria ser bem menor que LinearBow (%v) numa curva suave", got.CurveBow, got.LinearBow)
	}
}

// TestMeasureLineIrregularNaoBaixaComCurva simula um vinco: um degrau no
// meio da linha, que nem uma reta nem uma parabola explicam bem. CurveBow
// tem que continuar grande -- e a assinatura de N3.
func TestMeasureLineIrregularNaoBaixaComCurva(t *testing.T) {
	var baseline []geom.Point
	for x := 0.0; x <= 100; x += 10 {
		y := 0.0
		if x >= 50 {
			y = 15 // degrau abrupto
		}
		baseline = append(baseline, geom.Point{X: x, Y: y})
	}
	got, err := MeasureLine(baseline)
	if err != nil {
		t.Fatalf("MeasureLine: %v", err)
	}
	if got.CurveBow < 3 {
		t.Errorf("CurveBow = %v, esperava continuar grande (degrau nao e curva suave)", got.CurveBow)
	}
}

func TestMeasureLinePoucosPontos(t *testing.T) {
	if _, err := MeasureLine([]geom.Point{pt(0, 0)}); err == nil {
		t.Error("MeasureLine com 1 ponto deveria falhar")
	}
	if _, err := MeasureLine(nil); err == nil {
		t.Error("MeasureLine sem pontos deveria falhar")
	}
}
