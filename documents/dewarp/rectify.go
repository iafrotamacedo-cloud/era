package dewarp

import (
	"fmt"

	"github.com/iafrotamacedo-cloud/era/documents/geom"
	"github.com/iafrotamacedo-cloud/era/documents/imgproc"
)

// RectifyLine e o nivel N2 completo: ajusta uma curva pela baseline de uma
// linha de texto e delega a amostragem para geom.RemapCurve.
//
// aboveBaseline e belowBaseline sao a distancia, em pixels da imagem de
// origem, que a linha ocupa acima e abaixo da propria baseline --
// tipicamente a altura de ascendentes e descendentes do texto. Quem chama
// mede isso a partir do poligono original da linha detectada (o topo e a
// base do poligono, nao so a baseline); esta funcao nao tem como adivinhar.
//
// Com menos de CurveDegree+1 pontos na baseline (o caso de um quadrilatero
// simples, so 2 pontos), cai para grau 1: ainda endireita a inclinacao,
// so sem curvatura para ajustar -- coerente com o que MeasureLine ja faz
// no mesmo caso.
func RectifyLine(src *imgproc.Gray, baseline []geom.Point, outW, outH int, aboveBaseline, belowBaseline float64) (*imgproc.Gray, error) {
	if len(baseline) < 2 {
		return nil, fmt.Errorf("dewarp: %d ponto(s) na baseline nao bastam para retificar (preciso de pelo menos 2)", len(baseline))
	}

	grau := CurveDegree
	if len(baseline) < grau+1 {
		grau = 1
	}
	curva, err := geom.PolyFit(baseline, grau)
	if err != nil {
		return nil, fmt.Errorf("dewarp: ajuste de curva para retificar: %w", err)
	}

	xMin, xMax := baseline[0].X, baseline[0].X
	for _, p := range baseline {
		if p.X < xMin {
			xMin = p.X
		}
		if p.X > xMax {
			xMax = p.X
		}
	}

	out, err := geom.RemapCurve(src, curva, xMin, xMax, outW, outH, aboveBaseline, belowBaseline)
	if err != nil {
		return nil, fmt.Errorf("dewarp: retificar linha: %w", err)
	}
	return out, nil
}
