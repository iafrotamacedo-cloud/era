package geom

import (
	"fmt"
	"math"
)

// Homography e uma transformacao projetiva 2D, guardada como matriz 3x3
// row-major: h[0..2] e a primeira linha, h[3..5] a segunda, h[6..8] a
// terceira.
//
// E o que resolve o nivel N1 de retificacao: quatro cantos de um papel
// fotografado em perspectiva mapeiam para os quatro cantos de um retangulo
// -- ao contrario de uma transformacao afim, a homografia representa
// perspectiva de verdade (linhas paralelas no papel que convergem na foto).
type Homography [9]float64

// Apply transforma p pela homografia, com a divisao de perspectiva.
func (h Homography) Apply(p Point) Point {
	w := h[6]*p.X + h[7]*p.Y + h[8]
	return Point{
		X: (h[0]*p.X + h[1]*p.Y + h[2]) / w,
		Y: (h[3]*p.X + h[4]*p.Y + h[5]) / w,
	}
}

// Invert devolve a homografia inversa, pela formula fechada da matriz
// adjunta de uma 3x3 -- nao ha razao para eliminacao gaussiana numa matriz
// deste tamanho fixo.
func (h Homography) Invert() (Homography, error) {
	a, b, c := h[0], h[1], h[2]
	d, e, f := h[3], h[4], h[5]
	g, k, l := h[6], h[7], h[8] // k e l porque h e i ja estao em uso (h e o receiver, i colide com convencao de indice)

	det := a*(e*l-f*k) - b*(d*l-f*g) + c*(d*k-e*g)
	if math.Abs(det) < 1e-12 {
		return Homography{}, ErrSistemaSingular
	}

	return Homography{
		(e*l - f*k) / det, (c*k - b*l) / det, (b*f - c*e) / det,
		(f*g - d*l) / det, (a*l - c*g) / det, (c*d - a*f) / det,
		(d*k - e*g) / det, (b*g - a*k) / det, (a*e - b*d) / det,
	}, nil
}

// SolveHomography4 encontra a homografia que leva cada src[i] a dst[i], via
// DLT (direct linear transform): fixa h[8]=1 e resolve o sistema linear
// 8x8 resultante para as outras oito entradas.
//
// Fixar h[8]=1 assume que a homografia verdadeira nao tem h[8]=0 -- o caso
// em que o ponto no infinito do plano de origem mapeia para z=0 no destino,
// o que nao acontece em nenhuma foto de papel real (exigiria um plano
// paralelo ao plano de projecao no infinito). Se os quatro pontos forem
// colineares o sistema fica singular e o erro remonta ate aqui.
func SolveHomography4(src, dst [4]Point) (Homography, error) {
	a := make([][]float64, 8)
	b := make([]float64, 8)
	for i := 0; i < 4; i++ {
		x, y := src[i].X, src[i].Y
		u, v := dst[i].X, dst[i].Y
		a[2*i] = []float64{x, y, 1, 0, 0, 0, -u * x, -u * y}
		b[2*i] = u
		a[2*i+1] = []float64{0, 0, 0, x, y, 1, -v * x, -v * y}
		b[2*i+1] = v
	}

	sol, err := solveLinear(a, b)
	if err != nil {
		return Homography{}, fmt.Errorf("geom: homografia degenerada (cantos colineares?): %w", err)
	}
	return Homography{sol[0], sol[1], sol[2], sol[3], sol[4], sol[5], sol[6], sol[7], 1}, nil
}
