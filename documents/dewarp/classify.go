package dewarp

import "math"

// Thresholds separa N0/N1/N2/N3. Sao valores de partida, nao uma
// calibracao medida -- diferente de dist.Calibrate no maps, este motor
// ainda nao tem imagem real de linha fotografada para calibrar contra.
// Ajustar aqui quando houver dado real para medir contra.
type Thresholds struct {
	RetoPx    float64 // LinearBow ate aqui conta como reto: N0/N1, nao N2
	CurvoPx   float64 // CurveBow acima disso e irregular demais para N2: N3
	AnguloRad float64 // |Angle| acima disso conta como torto: N1, nao N0
}

// DefaultThresholds devolve limiares de partida: meio a um pixel e meio de
// folga para ruido de deteccao, um grau de tolerancia de inclinacao antes
// de considerar a pagina torta.
func DefaultThresholds() Thresholds {
	return Thresholds{
		RetoPx:    1.5,
		CurvoPx:   1.5,
		AnguloRad: math.Pi / 180, // 1 grau
	}
}

// Classify decide o nivel de uma unica linha a partir da sua forma medida.
//
// A ordem das checagens importa: CurveBow costuma ser <= LinearBow, porque
// PolyFit de grau 2 minimiza o erro quadratico sobre uma familia que contem
// a reta como caso particular -- mas isso vale para a soma dos quadrados,
// nao para o desvio maximo que estas duas metricas usam, entao nao e
// garantido ponto a ponto. Checar CurveBow primeiro trata o caso raro em
// que ele supera LinearBow do lado seguro: como N3, o nivel mais alto.
func Classify(s LineShape, th Thresholds) Level {
	switch {
	case s.CurveBow > th.CurvoPx:
		return N3
	case s.LinearBow > th.RetoPx:
		return N2
	case math.Abs(s.Angle) > th.AnguloRad:
		return N1
	default:
		return N0
	}
}

// PageLevel agrega a forma de todas as linhas da pagina numa decisao
// unica: o pior caso entre as linhas manda, porque retificar so parte da
// pagina nao ajuda quem le a pagina inteira.
//
// Pagina sem nenhuma linha (nada detectado ainda) devolve N0 -- nao ha
// deformacao para medir, nao e o mesmo que dizer que a pagina esta boa.
func PageLevel(shapes []LineShape, th Thresholds) Level {
	pior := N0
	for _, s := range shapes {
		if nivel := Classify(s, th); nivel > pior {
			pior = nivel
		}
	}
	return pior
}
