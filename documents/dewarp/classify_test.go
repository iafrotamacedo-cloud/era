package dewarp

import (
	"math"
	"testing"
)

func TestClassify(t *testing.T) {
	th := DefaultThresholds()
	casos := []struct {
		nome  string
		shape LineShape
		want  Level
	}{
		{"reta e alinhada", LineShape{Angle: 0, LinearBow: 0, CurveBow: 0}, N0},
		{"reta mas torta", LineShape{Angle: 10 * math.Pi / 180, LinearBow: 0, CurveBow: 0}, N1},
		{"curva suave", LineShape{Angle: 0, LinearBow: 20, CurveBow: 0.1}, N2},
		{"irregular", LineShape{Angle: 0, LinearBow: 20, CurveBow: 10}, N3},
		{"angulo no limite exato nao conta como torto", LineShape{Angle: th.AnguloRad, LinearBow: 0, CurveBow: 0}, N0},
		{"angulo logo acima do limite conta", LineShape{Angle: th.AnguloRad + 1e-9, LinearBow: 0, CurveBow: 0}, N1},
	}
	for _, c := range casos {
		t.Run(c.nome, func(t *testing.T) {
			if got := Classify(c.shape, th); got != c.want {
				t.Errorf("Classify(%+v) = %v, quero %v", c.shape, got, c.want)
			}
		})
	}
}

func TestPageLevelPiorCaso(t *testing.T) {
	th := DefaultThresholds()
	shapes := []LineShape{
		{Angle: 0, LinearBow: 0, CurveBow: 0},                  // N0
		{Angle: 10 * math.Pi / 180, LinearBow: 0, CurveBow: 0}, // N1
		{Angle: 0, LinearBow: 20, CurveBow: 0.1},               // N2
	}
	if got := PageLevel(shapes, th); got != N2 {
		t.Errorf("PageLevel = %v, quero N2 (pior caso entre as tres linhas)", got)
	}
}

func TestPageLevelVazioEN0(t *testing.T) {
	if got := PageLevel(nil, DefaultThresholds()); got != N0 {
		t.Errorf("PageLevel(vazio) = %v, quero N0", got)
	}
}

func TestPageLevelUmaLinhaN3ManDaTudo(t *testing.T) {
	th := DefaultThresholds()
	shapes := []LineShape{
		{Angle: 0, LinearBow: 0, CurveBow: 0},   // N0
		{Angle: 0, LinearBow: 20, CurveBow: 10}, // N3
		{Angle: 0, LinearBow: 0, CurveBow: 0},   // N0
	}
	if got := PageLevel(shapes, th); got != N3 {
		t.Errorf("PageLevel = %v, quero N3 (uma linha ruim basta)", got)
	}
}
