package geo

import "math"

// Haversine devolve a distancia em metros entre dois pontos sobre a esfera,
// medida ao longo da superficie -- o menor arco que liga os dois.
//
// O nome vem da funcao haversin(x) = sin^2(x/2), que aparece na formula. A
// versao ingenua desse calculo usa a lei dos cossenos esfericos e sofre de
// cancelamento catastrofico em distancias curtas: cos(sigma) fica tao perto
// de 1 que a diferenca se perde nos bits do float64, e distancias de poucos
// metros saem com erro grosseiro ou zeradas. A formula de haversine e
// algebricamente equivalente mas trabalha com senos de angulos pequenos, que
// o ponto flutuante representa bem. Numa operacao logistica, onde entregas
// na mesma rua sao a regra, isso deixa de ser detalhe academico.
//
// LawOfCosinesRef, em reference.go, existe justamente para o teste
// demonstrar a diferenca em vez de a afirmar.
func Haversine(a, b Point) float64 {
	lat1 := a.Lat * grausParaRad
	lat2 := b.Lat * grausParaRad
	dLat := (b.Lat - a.Lat) * grausParaRad
	dLon := (b.Lon - a.Lon) * grausParaRad

	sinLat := math.Sin(dLat / 2)
	sinLon := math.Sin(dLon / 2)

	h := sinLat*sinLat + math.Cos(lat1)*math.Cos(lat2)*sinLon*sinLon

	// h pode passar de 1 por arredondamento em pontos quase antipodais, e
	// Asin(>1) devolve NaN.
	if h > 1 {
		h = 1
	}
	return 2 * EarthRadius * math.Asin(math.Sqrt(h))
}

// Bearing devolve o azimute inicial de a para b, em graus de 0 a 360, com 0
// no norte e crescendo no sentido horario.
//
// E o azimute *inicial*: sobre uma esfera, o rumo de um grande circulo muda
// ao longo do caminho. Sair de Fortaleza rumo a Lisboa comeca apontando para
// um lado e termina apontando para outro.
func Bearing(a, b Point) float64 {
	lat1 := a.Lat * grausParaRad
	lat2 := b.Lat * grausParaRad
	dLon := (b.Lon - a.Lon) * grausParaRad

	y := math.Sin(dLon) * math.Cos(lat2)
	x := math.Cos(lat1)*math.Sin(lat2) - math.Sin(lat1)*math.Cos(lat2)*math.Cos(dLon)

	deg := math.Atan2(y, x) * radParaGraus
	if deg < 0 {
		deg += 360
	}
	return deg
}

// Destination devolve o ponto alcancado saindo de p no azimute dado e
// andando distM metros sobre a esfera.
//
// E a inversa de Haversine e Bearing juntas, e serve principalmente aos
// testes: gerar um ponto a distancia conhecida e conferir se a medicao a
// devolve fecha o ciclo sem precisar de tabela de valores decorados.
func Destination(p Point, bearingDeg, distM float64) Point {
	lat1 := p.Lat * grausParaRad
	lon1 := p.Lon * grausParaRad
	brg := bearingDeg * grausParaRad
	d := distM / EarthRadius // distancia angular

	sinLat1, cosLat1 := math.Sincos(lat1)
	sinD, cosD := math.Sincos(d)

	sinLat2 := sinLat1*cosD + cosLat1*sinD*math.Cos(brg)
	lat2 := math.Asin(sinLat2)
	lon2 := lon1 + math.Atan2(
		math.Sin(brg)*sinD*cosLat1,
		cosD-sinLat1*sinLat2,
	)

	return Point{Lat: lat2 * radParaGraus, Lon: NormalizeLon(lon2 * radParaGraus)}
}

// Ruler mede distancias por aproximacao plana em torno de uma latitude fixa.
//
// A ideia: dentro de uma regiao pequena, a Terra e praticamente um plano, e
// basta saber quantos metros vale um grau de latitude e um grau de longitude
// naquela faixa. Esses dois numeros sao calculados uma unica vez, e cada
// medicao vira duas multiplicacoes e uma raiz -- sem nenhuma chamada
// trigonometrica.
//
// Serve para o laco interno: comparar um ponto contra milhares de candidatos,
// ordenar por proximidade, montar agrupamentos. Um estado brasileiro cabe
// folgado na faixa util. Nao serve para distancia entre capitais distantes,
// e nao serve para nada que va virar numero em nota fiscal.
//
// O erro cresce com o afastamento da latitude de referencia; RulerErro, nos
// testes, mede o limite real em vez de confiar nesta frase.
type Ruler struct {
	kLat float64 // metros por grau de latitude
	kLon float64 // metros por grau de longitude na latitude de referencia
}

// NewRuler prepara um Ruler calibrado para a latitude de ref.
//
// Use o centro da regiao de interesse -- o deposito, o centroide da carteira
// de clientes, o meio do estado.
func NewRuler(ref Point) Ruler {
	return Ruler{
		kLat: metrosPorGrauLat,
		kLon: metrosPorGrauLat * math.Cos(ref.Lat*grausParaRad),
	}
}

// Distance devolve a distancia aproximada em metros entre dois pontos.
func (r Ruler) Distance(a, b Point) float64 {
	dLat := (b.Lat - a.Lat) * r.kLat
	dLon := lonDelta(a.Lon, b.Lon) * r.kLon
	return math.Hypot(dLat, dLon)
}

// SquaredDistance devolve o quadrado da distancia, em metros ao quadrado.
//
// Existe porque ordenar por distancia e comparar contra um raio nao precisam
// da raiz quadrada: se d1^2 < d2^2 entao d1 < d2. Numa ordenacao de mil
// candidatos sao mil raizes economizadas, e nenhuma perda de precisao.
func (r Ruler) SquaredDistance(a, b Point) float64 {
	dLat := (b.Lat - a.Lat) * r.kLat
	dLon := lonDelta(a.Lon, b.Lon) * r.kLon
	return dLat*dLat + dLon*dLon
}

// lonDelta devolve a diferenca de longitude pelo caminho mais curto, em
// graus, sempre em [-180, 180]. Sem isso, ir de 179 a -179 -- um passo de
// dois graus pelo Pacifico -- seria contado como 358.
func lonDelta(lon1, lon2 float64) float64 {
	d := lon2 - lon1
	switch {
	case d > 180:
		d -= 360
	case d < -180:
		d += 360
	}
	return d
}
