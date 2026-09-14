package geom

import (
	"math"
	"testing"
)

func TestPointOperacoesBasicas(t *testing.T) {
	p := Point{X: 3, Y: 4}
	q := Point{X: 1, Y: 2}

	if got := p.Add(q); got != (Point{4, 6}) {
		t.Errorf("Add = %v, quero {4,6}", got)
	}
	if got := p.Sub(q); got != (Point{2, 2}) {
		t.Errorf("Sub = %v, quero {2,2}", got)
	}
	if got := p.Scale(2); got != (Point{6, 8}) {
		t.Errorf("Scale = %v, quero {6,8}", got)
	}
	if got := p.Dot(q); got != 11 {
		t.Errorf("Dot = %v, quero 11", got)
	}
	if got := p.Len(); math.Abs(got-5) > 1e-9 {
		t.Errorf("Len = %v, quero 5 (triangulo 3-4-5)", got)
	}
}

func TestPointCrossSinalDaOrientacao(t *testing.T) {
	// De (1,0) para (0,1): girando na direcao que a formula do shoelace
	// chama de anti-horaria nesta convencao (Y para baixo). O sinal do
	// cross e o que Polygon.Area usa para decidir a orientacao do
	// poligono -- se o sinal aqui inverter sem querer, toda area sai com o
	// sinal trocado.
	a := Point{X: 1, Y: 0}
	b := Point{X: 0, Y: 1}
	if got := a.Cross(b); got <= 0 {
		t.Errorf("Cross((1,0),(0,1)) = %v, quero positivo", got)
	}
	if got := b.Cross(a); got >= 0 {
		t.Errorf("Cross((0,1),(1,0)) = %v, quero negativo (anti-simetrico)", got)
	}
}

func TestPointDist(t *testing.T) {
	a := Point{X: 0, Y: 0}
	b := Point{X: 3, Y: 4}
	if got := a.Dist(b); math.Abs(got-5) > 1e-9 {
		t.Errorf("Dist = %v, quero 5", got)
	}
	if got := a.Dist(a); got != 0 {
		t.Errorf("Dist ao proprio ponto = %v, quero 0", got)
	}
}
