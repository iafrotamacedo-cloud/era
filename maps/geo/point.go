// Package geo fornece o vocabulario geografico da ERA MAPAS: um ponto na
// superficie da Terra, o retangulo que envolve uma regiao e um indice
// espacial em grade.
//
// Nada aqui conhece estradas. Distancia em geo e distancia sobre a esfera:
// o piso teorico de qualquer rota, e o filtro barato que decide quais pares
// de pontos merecem o custo de um calculo de rota de verdade. Numa operacao
// com mil clientes ha um milhao de pares, e quase todos podem ser
// descartados por geometria antes de tocar no grafo rodoviario.
package geo

import (
	"fmt"
	"math"
)

// EarthRadius e o raio medio da Terra em metros, conforme a IUGG.
//
// A Terra nao e uma esfera -- e um elipsoide achatado nos polos, cerca de
// 21 km mais larga no equador do que alta nos polos. Tratar como esfera
// introduz erro de ate ~0,5% na distancia. Para filtrar e agrupar isso e
// irrelevante; para cotar frete, nao e -- e para isso existe a rota real.
// Vincenty, em reference.go, mede exatamente esse erro.
const EarthRadius = 6371008.8

// grausParaRad e metrosPorGrau aparecem em todo calculo deste pacote.
const (
	grausParaRad = math.Pi / 180
	radParaGraus = 180 / math.Pi

	// metrosPorGrauLat e o comprimento de um grau de latitude sobre a
	// esfera: constante em qualquer lugar do planeta. Um grau de
	// longitude, ao contrario, encolhe com o cosseno da latitude.
	metrosPorGrauLat = EarthRadius * grausParaRad

	// folgaGraus e a margem que BoxAround acrescenta de cada lado para
	// absorver arredondamento de ponto flutuante. Vale cerca de 0,1 mm no
	// terreno -- fisicamente nada, e varias ordens de grandeza acima do
	// erro do float64 em qualquer latitude.
	folgaGraus = 1e-9
)

// Point e uma coordenada geografica em graus decimais.
//
// Latitude cresce para o norte, longitude para o leste. E a ordem lat/lon,
// nao lon/lat -- a mesma do GPS e do OpenStreetMap, e o inverso da que
// alguns formatos GIS usam. Trocar as duas e o erro mais comum da area, e
// silencioso: Fortaleza vira um ponto no oceano Indico.
type Point struct {
	Lat float64 // graus, -90 (sul) a +90 (norte)
	Lon float64 // graus, -180 (oeste) a +180 (leste)
}

// Valid informa se o ponto esta dentro da faixa de coordenadas possiveis.
//
// Vale checar na fronteira do sistema -- ao ler um cadastro, um CSV ou uma
// resposta de geocodificacao. NaN e considerado invalido: as comparacoes
// abaixo sao falsas para NaN, entao ele reprova naturalmente.
func (p Point) Valid() bool {
	return p.Lat >= -90 && p.Lat <= 90 && p.Lon >= -180 && p.Lon <= 180
}

// String formata o ponto com 6 casas decimais -- cerca de 11 cm no equador,
// precisao muito acima da que qualquer fonte de endereco brasileira entrega.
func (p Point) String() string {
	return fmt.Sprintf("(%.6f, %.6f)", p.Lat, p.Lon)
}

// NormalizeLon traz uma longitude qualquer para a faixa [-180, 180).
//
// Existe porque somar deslocamentos a uma longitude pode estourar a faixa
// (179 + 3 = 182, que e o mesmo lugar que -178). No Brasil isso nunca
// acontece; o pacote trata mesmo assim porque um caso especial nao tratado
// vira bug em producao no dia em que alguem usar a biblioteca fora daqui.
func NormalizeLon(lon float64) float64 {
	if lon >= -180 && lon < 180 {
		return lon
	}
	lon = math.Mod(lon+180, 360)
	if lon < 0 {
		lon += 360
	}
	return lon - 180
}

// Box e um retangulo em latitude e longitude.
//
// Quando MinLon > MaxLon o retangulo cruza o antimeridiano (a linha de data,
// no Pacifico) e representa a uniao de [MinLon, 180) com [-180, MaxLon].
// Contains e Intersects sabem disso.
type Box struct {
	MinLat, MinLon float64
	MaxLat, MaxLon float64
}

// CrossesAntimeridian informa se o retangulo da a volta pelo Pacifico.
func (b Box) CrossesAntimeridian() bool { return b.MinLon > b.MaxLon }

// Contains informa se o ponto esta dentro do retangulo, bordas incluidas.
func (b Box) Contains(p Point) bool {
	if p.Lat < b.MinLat || p.Lat > b.MaxLat {
		return false
	}
	if b.CrossesAntimeridian() {
		return p.Lon >= b.MinLon || p.Lon <= b.MaxLon
	}
	return p.Lon >= b.MinLon && p.Lon <= b.MaxLon
}

// Intersects informa se os dois retangulos tem alguma area em comum.
func (b Box) Intersects(o Box) bool {
	if b.MaxLat < o.MinLat || b.MinLat > o.MaxLat {
		return false
	}
	bc, oc := b.CrossesAntimeridian(), o.CrossesAntimeridian()
	switch {
	case bc && oc:
		// Ambos cobrem o antimeridiano, entao ambos contem +-180.
		return true
	case bc:
		return o.MaxLon >= b.MinLon || o.MinLon <= b.MaxLon
	case oc:
		return b.MaxLon >= o.MinLon || b.MinLon <= o.MaxLon
	default:
		return b.MaxLon >= o.MinLon && b.MinLon <= o.MaxLon
	}
}

// Center devolve o ponto no meio do retangulo.
func (b Box) Center() Point {
	lat := (b.MinLat + b.MaxLat) / 2
	if b.CrossesAntimeridian() {
		return Point{Lat: lat, Lon: NormalizeLon((b.MinLon + b.MaxLon + 360) / 2)}
	}
	return Point{Lat: lat, Lon: (b.MinLon + b.MaxLon) / 2}
}

// BoxAround devolve o menor retangulo que contem o circulo de raio radiusM
// centrado em p.
//
// E o primeiro passo de toda busca por proximidade: o retangulo diz quais
// celulas da grade olhar, e so os pontos dessas celulas pagam o custo do
// calculo exato de distancia. O retangulo e sempre maior que o circulo --
// nos cantos sobra area -- entao ele filtra, mas nao decide.
//
// Perto dos polos um raio pequeno ja cobre todas as longitudes; nesse caso
// a faixa de longitude vira a volta inteira.
//
// O retangulo sai com uma folga de folgaGraus em cada lado. Sem ela, um
// ponto exatamente sobre a borda do circulo pode cair de fora: o retangulo
// chega na borda por uma divisao e o ponto chega la por trigonometria, e as
// duas contas discordam no ultimo bit do float64. Aqui a unica falha
// inaceitavel e perder um ponto -- sobrar area nao custa nada, porque quem
// decide de verdade e o calculo exato de distancia depois --, entao o
// desempate e sempre para dentro.
func BoxAround(p Point, radiusM float64) Box {
	dLat := radiusM/metrosPorGrauLat + folgaGraus

	minLat := p.Lat - dLat
	maxLat := p.Lat + dLat

	// Um grau de longitude vale menos quanto mais longe do equador. Usa-se
	// a latitude mais proxima do polo dentro do retangulo, que e onde os
	// graus de longitude sao mais curtos e portanto onde e preciso abrir
	// mais graus para cobrir o mesmo raio em metros.
	latExtrema := math.Max(math.Abs(minLat), math.Abs(maxLat))
	if minLat <= -90 || maxLat >= 90 || latExtrema >= 90 {
		return Box{
			MinLat: math.Max(minLat, -90), MinLon: -180,
			MaxLat: math.Min(maxLat, 90), MaxLon: 180,
		}
	}

	cos := math.Cos(latExtrema * grausParaRad)
	dLon := radiusM/(metrosPorGrauLat*cos) + folgaGraus
	if dLon >= 180 {
		return Box{MinLat: minLat, MinLon: -180, MaxLat: maxLat, MaxLon: 180}
	}

	return Box{
		MinLat: minLat, MinLon: NormalizeLon(p.Lon - dLon),
		MaxLat: maxLat, MaxLon: NormalizeLon(p.Lon + dLon),
	}
}
