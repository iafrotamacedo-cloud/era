package geo

import (
	"errors"
	"math"
)

// Este arquivo guarda as implementacoes de referencia -- lentas, obvias ou
// mais exatas do que o resto do pacote precisa ser. Elas existem para servir
// de verdade nos testes.
//
// Nunca otimize nada aqui. O valor destas funcoes esta em serem simples o
// bastante para dar para ler e afirmar que estao certas, ou exatas o
// bastante para servirem de padrao contra o qual o erro das outras e medido.

// HaversineRef e a formula de haversine escrita da forma mais direta,
// sem nenhum cuidado com desempenho ou com o caso limite de h > 1.
//
// Serve de conferencia algebrica para Haversine: as duas devem concordar
// ate o ultimo bit util em qualquer par de pontos nao antipodal.
func HaversineRef(a, b Point) float64 {
	lat1 := a.Lat * math.Pi / 180
	lon1 := a.Lon * math.Pi / 180
	lat2 := b.Lat * math.Pi / 180
	lon2 := b.Lon * math.Pi / 180

	h := math.Pow(math.Sin((lat2-lat1)/2), 2) +
		math.Cos(lat1)*math.Cos(lat2)*math.Pow(math.Sin((lon2-lon1)/2), 2)

	return 2 * EarthRadius * math.Atan2(math.Sqrt(h), math.Sqrt(1-h))
}

// LawOfCosinesRef mede a mesma distancia esferica pela lei dos cossenos.
//
// E algebricamente equivalente a haversine e numericamente pior: para
// pontos proximos, cos(sigma) fica tao perto de 1 que a subtracao dentro do
// Acos perde quase todos os digitos significativos. Esta aqui como
// contraexemplo -- o teste usa esta funcao para mostrar, com numeros, por
// que Haversine nao foi escrita assim.
//
// Nao use em codigo de producao.
func LawOfCosinesRef(a, b Point) float64 {
	lat1 := a.Lat * math.Pi / 180
	lat2 := b.Lat * math.Pi / 180
	dLon := (b.Lon - a.Lon) * math.Pi / 180

	c := math.Sin(lat1)*math.Sin(lat2) + math.Cos(lat1)*math.Cos(lat2)*math.Cos(dLon)
	if c > 1 {
		c = 1
	} else if c < -1 {
		c = -1
	}
	return EarthRadius * math.Acos(c)
}

// Parametros do elipsoide WGS-84, o mesmo que o GPS e o OpenStreetMap usam.
const (
	wgs84A = 6378137.0             // semieixo maior, em metros (raio no equador)
	wgs84F = 1 / 298.257223563     // achatamento
	wgs84B = (1 - wgs84F) * wgs84A // semieixo menor (raio nos polos)
)

// ErrVincentyNaoConvergiu indica que o metodo iterativo nao fechou.
//
// Acontece com pontos quase antipodais -- extremos opostos do planeta --,
// onde a geodesica e mal condicionada e existem infinitos caminhos de mesmo
// comprimento. Nao acontece com nenhum par de pontos dentro do Brasil.
var ErrVincentyNaoConvergiu = errors.New("geo: Vincenty nao convergiu (pontos quase antipodais)")

// Vincenty devolve a distancia geodesica sobre o elipsoide WGS-84, em
// metros, com precisao submilimetrica.
//
// E a verdade contra a qual o erro do modelo esferico e medido. A Terra e
// achatada nos polos: um grau de latitude perto do equador e mais curto do
// que perto dos polos, e nenhuma esfera reproduz isso. Haversine erra por
// isso, e o teste usa esta funcao para dizer exatamente quanto.
//
// Nao e o caminho de producao: e iterativa, faz varias chamadas
// trigonometricas por iteracao e custa uma ordem de grandeza a mais que
// Haversine. Para logistica o erro do modelo esferico e absorvido pelo
// fator de desvio da rota real muito antes de importar.
func Vincenty(p1, p2 Point) (float64, error) {
	const (
		tolerancia   = 1e-12
		maxIteracoes = 200
	)

	// Latitude reduzida: a projecao da latitude geografica sobre a esfera
	// auxiliar. E o truque que transforma o problema elipsoidal num
	// problema esferico corrigido por uma serie.
	u1 := math.Atan((1 - wgs84F) * math.Tan(p1.Lat*grausParaRad))
	u2 := math.Atan((1 - wgs84F) * math.Tan(p2.Lat*grausParaRad))
	l := lonDelta(p1.Lon, p2.Lon) * grausParaRad

	sinU1, cosU1 := math.Sincos(u1)
	sinU2, cosU2 := math.Sincos(u2)

	lambda := l
	var sinSigma, cosSigma, sigma, cos2Alpha, cos2SigmaM float64

	convergiu := false
	for i := 0; i < maxIteracoes; i++ {
		sinLambda, cosLambda := math.Sincos(lambda)

		a := cosU2 * sinLambda
		b := cosU1*sinU2 - sinU1*cosU2*cosLambda
		sinSigma = math.Hypot(a, b)
		if sinSigma == 0 {
			return 0, nil // pontos coincidentes
		}
		cosSigma = sinU1*sinU2 + cosU1*cosU2*cosLambda
		sigma = math.Atan2(sinSigma, cosSigma)

		sinAlpha := cosU1 * cosU2 * sinLambda / sinSigma
		cos2Alpha = 1 - sinAlpha*sinAlpha

		if cos2Alpha == 0 {
			cos2SigmaM = 0 // linha equatorial: nao ha ponto medio definido
		} else {
			cos2SigmaM = cosSigma - 2*sinU1*sinU2/cos2Alpha
		}

		c := wgs84F / 16 * cos2Alpha * (4 + wgs84F*(4-3*cos2Alpha))
		lambdaAnterior := lambda
		lambda = l + (1-c)*wgs84F*sinAlpha*
			(sigma+c*sinSigma*(cos2SigmaM+c*cosSigma*(-1+2*cos2SigmaM*cos2SigmaM)))

		if math.Abs(lambda-lambdaAnterior) < tolerancia {
			convergiu = true
			break
		}
	}
	if !convergiu {
		return 0, ErrVincentyNaoConvergiu
	}

	uSq := cos2Alpha * (wgs84A*wgs84A - wgs84B*wgs84B) / (wgs84B * wgs84B)
	bigA := 1 + uSq/16384*(4096+uSq*(-768+uSq*(320-175*uSq)))
	bigB := uSq / 1024 * (256 + uSq*(-128+uSq*(74-47*uSq)))

	deltaSigma := bigB * sinSigma * (cos2SigmaM +
		bigB/4*(cosSigma*(-1+2*cos2SigmaM*cos2SigmaM)-
			bigB/6*cos2SigmaM*(-3+4*sinSigma*sinSigma)*(-3+4*cos2SigmaM*cos2SigmaM)))

	return wgs84B * bigA * (sigma - deltaSigma), nil
}
