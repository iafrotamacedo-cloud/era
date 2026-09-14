package geom

import "fmt"

// PolyFit ajusta o polinomio de grau degree que minimiza o erro quadratico
// aos pontos dados, resolvendo as equacoes normais dos minimos quadrados.
// O resultado e devolvido em ordem crescente de potencia:
// coeffs[0] + coeffs[1]*x + coeffs[2]*x^2 + ...
//
// E o coracao do nivel N2: ajustar uma curva pela baseline de uma linha de
// texto detectada, para amostrar a linha reta na retificacao.
//
// Equacoes normais sao a via mais direta, mas mal condicionadas para grau
// alto ou coordenadas grandes -- o quadrado da matriz de Vandermonde
// amplifica o condicionamento numerico. Para o uso pretendido (grau 2 ou 3,
// coordenadas em pixels de uma unica linha de texto) isso nao chega a
// incomodar; grau maior ou dominio muito grande pediria decomposicao QR
// direta sobre a matriz de Vandermonde, nao normal equations.
func PolyFit(pts []Point, degree int) ([]float64, error) {
	if degree < 0 {
		return nil, fmt.Errorf("geom: grau do polinomio nao pode ser negativo: %d", degree)
	}
	n := degree + 1
	if len(pts) < n {
		return nil, fmt.Errorf("geom: %d pontos nao bastam para ajustar grau %d (preciso de pelo menos %d)", len(pts), degree, n)
	}

	// powSums[k] = soma de x_i^k, k de 0 a 2*degree -- as potencias que
	// aparecem na matriz simetrica das equacoes normais.
	powSums := make([]float64, 2*degree+1)
	ySums := make([]float64, n) // ySums[k] = soma de y_i * x_i^k
	for _, p := range pts {
		xp := 1.0
		for k := 0; k <= 2*degree; k++ {
			powSums[k] += xp
			xp *= p.X
		}
		xp = 1.0
		for k := 0; k < n; k++ {
			ySums[k] += p.Y * xp
			xp *= p.X
		}
	}

	a := make([][]float64, n)
	for k := 0; k < n; k++ {
		a[k] = make([]float64, n)
		for j := 0; j < n; j++ {
			a[k][j] = powSums[j+k]
		}
	}

	coeffs, err := solveLinear(a, ySums)
	if err != nil {
		return nil, fmt.Errorf("geom: ajuste de curva degenerado (pontos insuficientes em x distintos?): %w", err)
	}
	return coeffs, nil
}

// EvalPoly avalia o polinomio de coeficientes coeffs (ordem crescente de
// potencia, como devolvido por PolyFit) em x.
func EvalPoly(coeffs []float64, x float64) float64 {
	var y, xp float64 = 0, 1
	for _, c := range coeffs {
		y += c * xp
		xp *= x
	}
	return y
}
