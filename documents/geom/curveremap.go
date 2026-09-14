package geom

import (
	"fmt"
	"math"

	"github.com/iafrotamacedo-cloud/era/documents/imgproc"
)

// derivPoly devolve os coeficientes da derivada de um polinomio, na mesma
// convencao de PolyFit/EvalPoly (ordem crescente de potencia).
func derivPoly(coeffs []float64) []float64 {
	if len(coeffs) <= 1 {
		return []float64{0}
	}
	d := make([]float64, len(coeffs)-1)
	for i := 1; i < len(coeffs); i++ {
		d[i-1] = coeffs[i] * float64(i)
	}
	return d
}

// arcLenTable e uma tabela de comprimento de arco acumulado ao longo de uma
// curva, amostrada em x igualmente espaçado e integrada pela regra do
// trapezio. Serve para inverter "comprimento de arco -> x", que nao tem
// formula fechada para um polinomio generico.
type arcLenTable struct {
	xs, ss []float64 // ss[i] = comprimento de arco entre xs[0] e xs[i]
}

func buildArcLenTable(curve []float64, xMin, xMax float64, n int) arcLenTable {
	deriv := derivPoly(curve)
	xs := make([]float64, n)
	ss := make([]float64, n)

	prevX := xMin
	prevSpeed := math.Hypot(1, EvalPoly(deriv, xMin))
	xs[0] = xMin
	for i := 1; i < n; i++ {
		x := xMin + (xMax-xMin)*float64(i)/float64(n-1)
		speed := math.Hypot(1, EvalPoly(deriv, x)) // |(dx,dy)/dx| = sqrt(1+f'(x)^2)
		ss[i] = ss[i-1] + (prevSpeed+speed)/2*(x-prevX)
		xs[i] = x
		prevX, prevSpeed = x, speed
	}
	return arcLenTable{xs: xs, ss: ss}
}

func (t arcLenTable) total() float64 { return t.ss[len(t.ss)-1] }

// xAtLen inverte a tabela: acha x tal que o comprimento de arco desde xMin
// seja aproximadamente s, por busca binaria na tabela e interpolacao linear
// entre os dois pontos vizinhos.
func (t arcLenTable) xAtLen(s float64) float64 {
	n := len(t.ss)
	if s <= t.ss[0] {
		return t.xs[0]
	}
	if s >= t.ss[n-1] {
		return t.xs[n-1]
	}
	lo, hi := 0, n-1
	for lo < hi {
		mid := (lo + hi) / 2
		if t.ss[mid] < s {
			lo = mid + 1
		} else {
			hi = mid
		}
	}
	frac := (s - t.ss[lo-1]) / (t.ss[lo] - t.ss[lo-1])
	return t.xs[lo-1] + frac*(t.xs[lo]-t.xs[lo-1])
}

// amostrasArco e a resolucao da tabela de comprimento de arco. Curvas de
// linha de texto sao suaves (grau baixo, poucas centenas de pixels de
// largura); esta resolucao deixa o erro de integracao bem abaixo de um
// pixel sem custar tempo perceptivel.
const amostrasArco = 512

// RemapCurve endireita a faixa da imagem que segue uma curva -- o nivel N2
// de retificacao: uma linha de texto cuja baseline e curva, mas suave.
//
// curve e o polinomio y=f(x) (convencao de PolyFit/EvalPoly) que aproxima a
// baseline. xMin/xMax delimitam o trecho a retificar -- tipicamente o
// intervalo em x da propria linha detectada.
//
// A saida tem outW colunas amostradas em espaçamento igual de COMPRIMENTO
// DE ARCO ao longo da curva, nao de x -- do contrario um trecho mais
// inclinado da curva sairia com o texto comprimido em relacao a um trecho
// mais plano, ja que percorre mais distancia real por unidade de x. Cada
// coluna de saida e uma reta perpendicular a curva naquele ponto (a normal
// local), amostrada de aboveCurve pixels acima ate belowCurve pixels
// abaixo, dividida em outH linhas.
//
// So funciona para curva que e, de fato, uma funcao de x -- sem dobra
// vertical. E a mesma suposicao que geom.PolyFit e dewarp.MeasureLine ja
// fazem: vale para linha de texto proxima da horizontal, nao para texto
// vertical ou uma curva que vira sobre si mesma.
func RemapCurve(src *imgproc.Gray, curve []float64, xMin, xMax float64, outW, outH int, aboveCurve, belowCurve float64) (*imgproc.Gray, error) {
	if xMax <= xMin {
		return nil, fmt.Errorf("geom: xMax (%v) tem que ser maior que xMin (%v)", xMax, xMin)
	}
	if outW <= 0 || outH <= 0 {
		return nil, fmt.Errorf("geom: dimensoes de saida invalidas %dx%d", outW, outH)
	}

	deriv := derivPoly(curve)
	tabela := buildArcLenTable(curve, xMin, xMax, amostrasArco)
	total := tabela.total()

	dst := imgproc.NewGray(outW, outH)
	for j := 0; j < outW; j++ {
		s := (float64(j) + 0.5) / float64(outW) * total
		x := tabela.xAtLen(s)
		y := EvalPoly(curve, x)
		m := EvalPoly(deriv, x)
		norma := math.Hypot(1, m)
		// normal unitaria a tangente (1,m): (-m,1)/norma. O componente em
		// y e sempre 1/norma > 0, entao "abaixo da curva" e sempre offset
		// positivo, independente do sinal da inclinacao local -- sem essa
		// normalizacao, o lado de "acima" e "abaixo" se inverteria onde a
		// curva descesse em vez de subir.
		nx, ny := -m/norma, 1/norma

		for i := 0; i < outH; i++ {
			offset := -aboveCurve
			if outH > 1 {
				offset = -aboveCurve + (aboveCurve+belowCurve)*float64(i)/float64(outH-1)
			}
			dst.Set(j, i, imgproc.SampleBilinear(src, x+nx*offset, y+ny*offset))
		}
	}
	return dst, nil
}
