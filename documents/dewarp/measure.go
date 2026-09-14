package dewarp

import (
	"fmt"
	"math"

	"github.com/iafrotamacedo-cloud/era/documents/geom"
)

// CurveDegree e o grau do polinomio usado para representar o que uma
// retificacao N2 consegue absorver. Grau 2 (parabola) e o mais simples que
// captura curvatura de verdade sem ainda arriscar o mal condicionamento que
// geom.PolyFit documenta para grau alto.
const CurveDegree = 2

// LineShape resume a forma de uma linha de texto, medida a partir da sua
// baseline.
type LineShape struct {
	Angle     float64 // inclinacao da reta ajustada aos pontos, em radianos
	LinearBow float64 // desvio maximo (px) da baseline em relacao a essa reta
	CurveBow  float64 // desvio maximo (px) da baseline em relacao a uma curva de grau CurveDegree

}

// MeasureLine mede a forma de uma linha a partir dos pontos da sua
// baseline (ver ExtractBaseline), ajustando uma reta e, quando ha pontos
// suficientes, uma curva de grau CurveDegree.
//
// LinearBow grande com CurveBow pequeno e a assinatura de uma curva suave
// (N2 resolve). Os dois grandes e a assinatura de algo que uma curva unica
// nao explica -- um vinco, uma dobra abrupta (candidato a N3).
//
// Baseline com menos de CurveDegree+1 pontos (o caso normal de um
// quadrilatero de 4 vertices, que so da 2 pontos de base) nao tenta ajustar
// curva -- nao ha o que ajustar com 2 pontos -- e CurveBow sai igual a
// LinearBow. Isso e o comportamento certo, nao uma degradacao: um poligono
// de 4 vertices e o proprio detector dizendo que considera a linha reta.
func MeasureLine(baseline []geom.Point) (LineShape, error) {
	if len(baseline) < 2 {
		return LineShape{}, fmt.Errorf("dewarp: %d ponto(s) na baseline nao bastam para medir uma linha (preciso de pelo menos 2)", len(baseline))
	}

	reta, err := geom.PolyFit(baseline, 1)
	if err != nil {
		return LineShape{}, fmt.Errorf("dewarp: ajuste de reta: %w", err)
	}
	angle := math.Atan(reta[1])

	linearBow := maxResidual(baseline, reta)
	curveBow := linearBow
	if len(baseline) >= CurveDegree+1 {
		if curva, err := geom.PolyFit(baseline, CurveDegree); err == nil {
			curveBow = maxResidual(baseline, curva)
		}
		// se o ajuste de curva falhar (pontos degenerados em x), mantem o
		// fallback linearBow -- nao ha curva melhor para tentar.
	}

	return LineShape{Angle: angle, LinearBow: linearBow, CurveBow: curveBow}, nil
}

// maxResidual devolve o maior |y - poly(x)| entre os pontos dados.
func maxResidual(pts []geom.Point, coeffs []float64) float64 {
	var max float64
	for _, p := range pts {
		d := math.Abs(p.Y - geom.EvalPoly(coeffs, p.X))
		if d > max {
			max = d
		}
	}
	return max
}
