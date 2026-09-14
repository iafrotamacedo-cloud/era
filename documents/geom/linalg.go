package geom

import (
	"errors"
	"math"
)

// ErrSistemaSingular indica que um sistema linear nao tem solucao unica --
// tipicamente quatro cantos colineares (SolveHomography4) ou pontos
// coincidentes/insuficientes em x (PolyFit).
var ErrSistemaSingular = errors.New("geom: sistema linear singular (entrada degenerada)")

// solveLinear resolve A x = b por eliminacao gaussiana com pivoteamento
// parcial. a e b nao sao alterados -- a funcao trabalha em copias.
//
// E um solver generico de proposito pequeno: os sistemas deste pacote tem
// no maximo 8 incognitas (SolveHomography4) ou o grau do polinomio mais um
// (PolyFit), entao eliminacao gaussiana simples e suficiente. Nao ha
// necessidade de decomposicao mais sofisticada nessa escala.
func solveLinear(a [][]float64, b []float64) ([]float64, error) {
	n := len(b)
	m := make([][]float64, n)
	for i := range a {
		m[i] = append([]float64(nil), a[i]...)
	}
	x := append([]float64(nil), b...)

	for col := 0; col < n; col++ {
		piv := col
		maxAbs := math.Abs(m[col][col])
		for r := col + 1; r < n; r++ {
			if v := math.Abs(m[r][col]); v > maxAbs {
				maxAbs, piv = v, r
			}
		}
		if maxAbs < 1e-12 {
			return nil, ErrSistemaSingular
		}
		m[col], m[piv] = m[piv], m[col]
		x[col], x[piv] = x[piv], x[col]

		pivotVal := m[col][col]
		for r := col + 1; r < n; r++ {
			factor := m[r][col] / pivotVal
			if factor == 0 {
				continue
			}
			for c := col; c < n; c++ {
				m[r][c] -= factor * m[col][c]
			}
			x[r] -= factor * x[col]
		}
	}

	resultado := make([]float64, n)
	for i := n - 1; i >= 0; i-- {
		soma := x[i]
		for j := i + 1; j < n; j++ {
			soma -= m[i][j] * resultado[j]
		}
		resultado[i] = soma / m[i][i]
	}
	return resultado, nil
}
