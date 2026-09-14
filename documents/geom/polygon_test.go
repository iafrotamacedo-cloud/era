package geom

import (
	"math"
	"testing"
)

func quadrado(lado float64) Polygon {
	return Polygon{{0, 0}, {lado, 0}, {lado, lado}, {0, lado}}
}

func TestPolygonAreaQuadrado(t *testing.T) {
	p := quadrado(3)
	if got := p.Area(); math.Abs(got-9) > 1e-9 {
		t.Errorf("Area = %v, quero 9", got)
	}
	// vertices na ordem inversa devem dar a mesma area com sinal trocado --
	// e a definicao de area assinada, nao um detalhe de implementacao.
	invertido := Polygon{{0, 0}, {0, 3}, {3, 3}, {3, 0}}
	if got := invertido.Area(); math.Abs(got+9) > 1e-9 {
		t.Errorf("Area invertida = %v, quero -9", got)
	}
}

func TestPolygonAreaTriangulo(t *testing.T) {
	// triangulo retangulo de base 4 e altura 3: area = base*altura/2 = 6
	p := Polygon{{0, 0}, {4, 0}, {0, 3}}
	if got := math.Abs(p.Area()); math.Abs(got-6) > 1e-9 {
		t.Errorf("Area do triangulo = %v, quero 6", got)
	}
}

func TestPolygonAreaMenosDeTresVertices(t *testing.T) {
	if got := (Polygon{{0, 0}, {1, 1}}).Area(); got != 0 {
		t.Errorf("Area com 2 vertices = %v, quero 0", got)
	}
}

func TestPolygonPerimeterQuadrado(t *testing.T) {
	p := quadrado(5)
	if got := p.Perimeter(); math.Abs(got-20) > 1e-9 {
		t.Errorf("Perimeter = %v, quero 20", got)
	}
}

func TestPolygonCentroidQuadrado(t *testing.T) {
	p := quadrado(4)
	c := p.Centroid()
	if math.Abs(c.X-2) > 1e-9 || math.Abs(c.Y-2) > 1e-9 {
		t.Errorf("Centroid = %v, quero (2,2)", c)
	}
}

// TestPolygonCentroidNaoEMediaDeVertices confere que Centroid pondera por
// area, nao so tira a media dos vertices -- as duas coincidem num quadrado
// (por simetria) mas nao num poligono assimetrico, entao o quadrado sozinho
// nao provaria nada.
func TestPolygonCentroidNaoEMediaDeVertices(t *testing.T) {
	// Quadrado 4x4 com uma pequena mordida 1x1 no canto superior direito:
	// a mordida e pequena o bastante para nao arrastar o centroide de area
	// para fora do poligono (ao contrario de uma mordida grande, onde o
	// centroide pode legitimamente cair no recorte -- isso nao seria bug,
	// so nao serviria para este teste).
	p := Polygon{
		{0, 0}, {4, 0}, {4, 3}, {3, 3}, {3, 4}, {0, 4},
	}
	c := p.Centroid()

	var mx, my float64
	for _, v := range p {
		mx += v.X
		my += v.Y
	}
	mx /= float64(len(p))
	my /= float64(len(p))

	if math.Abs(c.X-mx) < 1e-6 && math.Abs(c.Y-my) < 1e-6 {
		t.Fatalf("Centroid = %v igual a media dos vertices %v, esperava diferente", c, Point{mx, my})
	}
	// e ainda precisa estar dentro do poligono, que e a propriedade que
	// importa de verdade.
	if !p.Contains(c) {
		t.Errorf("Centroid %v deveria estar dentro do poligono em L", c)
	}
}

func TestPolygonCentroidDegenerado(t *testing.T) {
	// tres pontos colineares: area e zero, Centroid cai para a media.
	p := Polygon{{0, 0}, {1, 0}, {2, 0}}
	c := p.Centroid()
	if math.Abs(c.X-1) > 1e-9 || math.Abs(c.Y) > 1e-9 {
		t.Errorf("Centroid degenerado = %v, quero (1,0) (media dos vertices)", c)
	}
}

func TestPolygonBounds(t *testing.T) {
	p := Polygon{{2, 5}, {-1, 3}, {4, -2}, {0, 0}}
	min, max := p.Bounds()
	if min != (Point{-1, -2}) {
		t.Errorf("min = %v, quero {-1,-2}", min)
	}
	if max != (Point{4, 5}) {
		t.Errorf("max = %v, quero {4,5}", max)
	}
}

func TestPolygonContains(t *testing.T) {
	p := quadrado(10)
	casos := []struct {
		nome string
		pt   Point
		want bool
	}{
		{"centro", Point{5, 5}, true},
		{"longe fora", Point{50, 50}, false},
		{"fora a esquerda", Point{-1, 5}, false},
		{"fora acima", Point{5, -1}, false},
		{"quase no canto, dentro", Point{9.9, 9.9}, true},
	}
	for _, c := range casos {
		t.Run(c.nome, func(t *testing.T) {
			if got := p.Contains(c.pt); got != c.want {
				t.Errorf("Contains(%v) = %v, quero %v", c.pt, got, c.want)
			}
		})
	}
}
