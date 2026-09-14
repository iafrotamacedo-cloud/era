package geom

import "math"

// Polygon e uma sequencia de vertices, no sentido dado -- o ultimo vertice
// fecha implicitamente no primeiro.
//
// E o formato que um detector de texto devolve por caixa: 4 pontos para uma
// palavra reta, ou mais para uma curva -- ao contrario de um retangulo
// axis-aligned, um Polygon representa texto que a perspectiva ou a
// curvatura do papel deixou torto.
type Polygon []Point

// Area devolve a area assinada pela formula do shoelace. O sinal indica a
// orientacao: positivo se os vertices andam no sentido anti-horario da tela
// (lembrando que Y cresce para baixo). Poligonos com menos de 3 vertices
// tem area zero.
func (p Polygon) Area() float64 {
	n := len(p)
	if n < 3 {
		return 0
	}
	var soma float64
	for i := 0; i < n; i++ {
		j := (i + 1) % n
		soma += p[i].X*p[j].Y - p[j].X*p[i].Y
	}
	return soma / 2
}

// Perimeter soma a distancia entre vertices consecutivos, incluindo o
// segmento que fecha do ultimo ao primeiro.
func (p Polygon) Perimeter() float64 {
	n := len(p)
	if n < 2 {
		return 0
	}
	var soma float64
	for i := 0; i < n; i++ {
		j := (i + 1) % n
		soma += p[i].Dist(p[j])
	}
	return soma
}

// Centroid devolve o centro de massa da area do poligono (nao a media
// simples dos vertices -- as duas coincidem so em poligonos regulares).
//
// Poligonos degenerados (area proxima de zero: colineares, ou os mesmos
// pontos repetidos) caem para a media simples dos vertices, que continua
// bem definida quando a formula de area nao e.
func (p Polygon) Centroid() Point {
	n := len(p)
	switch {
	case n == 0:
		return Point{}
	case n < 3:
		var sx, sy float64
		for _, pt := range p {
			sx += pt.X
			sy += pt.Y
		}
		return Point{sx / float64(n), sy / float64(n)}
	}

	var cx, cy, areaAcc float64
	for i := 0; i < n; i++ {
		j := (i + 1) % n
		cross := p[i].X*p[j].Y - p[j].X*p[i].Y
		areaAcc += cross
		cx += (p[i].X + p[j].X) * cross
		cy += (p[i].Y + p[j].Y) * cross
	}
	a := areaAcc / 2
	if math.Abs(a) < 1e-9 {
		var sx, sy float64
		for _, pt := range p {
			sx += pt.X
			sy += pt.Y
		}
		return Point{sx / float64(n), sy / float64(n)}
	}
	return Point{cx / (6 * a), cy / (6 * a)}
}

// Bounds devolve o retangulo axis-aligned que envolve todos os vertices.
func (p Polygon) Bounds() (min, max Point) {
	if len(p) == 0 {
		return Point{}, Point{}
	}
	min, max = p[0], p[0]
	for _, pt := range p[1:] {
		min.X = math.Min(min.X, pt.X)
		min.Y = math.Min(min.Y, pt.Y)
		max.X = math.Max(max.X, pt.X)
		max.Y = math.Max(max.Y, pt.Y)
	}
	return min, max
}

// Contains informa se pt esta dentro do poligono, por ray casting (conta
// quantas arestas um raio horizontal partindo de pt cruza; impar e dentro).
// Comportamento na borda exata nao e garantido -- e a imprecisao classica
// do metodo, e nao importa aqui: nada neste motor decide algo no fio da
// borda de um poligono de texto.
func (p Polygon) Contains(pt Point) bool {
	n := len(p)
	dentro := false
	for i, j := 0, n-1; i < n; j, i = i, i+1 {
		pi, pj := p[i], p[j]
		if (pi.Y > pt.Y) != (pj.Y > pt.Y) {
			xCruza := (pj.X-pi.X)*(pt.Y-pi.Y)/(pj.Y-pi.Y) + pi.X
			if pt.X < xCruza {
				dentro = !dentro
			}
		}
	}
	return dentro
}
