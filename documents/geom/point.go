package geom

import "math"

// Point e uma coordenada 2D em pixels de imagem.
//
// Diferente de geo.Point (latitude/longitude em graus, na superficie da
// Terra), este Point vive no plano da imagem: X cresce para a direita, Y
// para baixo -- a convencao de imagem, nao a de eixos cartesianos.
type Point struct {
	X, Y float64
}

// Add soma dois pontos como vetores.
func (p Point) Add(q Point) Point { return Point{p.X + q.X, p.Y + q.Y} }

// Sub subtrai q de p.
func (p Point) Sub(q Point) Point { return Point{p.X - q.X, p.Y - q.Y} }

// Scale multiplica p por um escalar.
func (p Point) Scale(s float64) Point { return Point{p.X * s, p.Y * s} }

// Dot e o produto escalar de p e q.
func (p Point) Dot(q Point) float64 { return p.X*q.X + p.Y*q.Y }

// Cross e o componente Z do produto vetorial 3D de p e q tratados como
// vetores no plano (0 em Z) -- um escalar, nao um vetor. O sinal diz de que
// lado q esta em relacao a p: positivo se a volta de p para q e
// anti-horaria (lembrando que Y cresce para baixo, entao "anti-horaria" na
// tela parece horaria em papel). E o que Polygon.Area usa para dar sinal a
// orientacao do poligono.
func (p Point) Cross(q Point) float64 { return p.X*q.Y - p.Y*q.X }

// Len e a distancia de p ate a origem.
func (p Point) Len() float64 { return math.Hypot(p.X, p.Y) }

// Dist e a distancia entre p e q.
func (p Point) Dist(q Point) float64 { return p.Sub(q).Len() }
