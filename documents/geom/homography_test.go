package geom

import (
	"math"
	"math/rand"
	"testing"
)

func TestSolveHomography4CasoAfim(t *testing.T) {
	// Um caso sem perspectiva (escala 2x mais translacao) e o mais facil de
	// conferir a mao: a homografia resolvida deve reproduzir exatamente o
	// mapeamento afim que gerou os pontos.
	src := [4]Point{{0, 0}, {10, 0}, {10, 10}, {0, 10}}
	dst := [4]Point{{5, 5}, {25, 5}, {25, 25}, {5, 25}} // escala 2, desloca +5

	hom, err := SolveHomography4(src, dst)
	if err != nil {
		t.Fatalf("SolveHomography4: %v", err)
	}
	for i, s := range src {
		got := hom.Apply(s)
		if math.Abs(got.X-dst[i].X) > 1e-6 || math.Abs(got.Y-dst[i].Y) > 1e-6 {
			t.Errorf("Apply(%v) = %v, quero %v", s, got, dst[i])
		}
	}
}

func TestSolveHomography4CasoPerspectiva(t *testing.T) {
	// Um trapezio (papel fotografado em angulo) mapeando para um
	// retangulo: aqui a divisao de perspectiva de fato importa, diferente
	// do caso afim acima.
	src := [4]Point{{2, 0}, {8, 0}, {10, 10}, {0, 10}} // trapezio
	dst := [4]Point{{0, 0}, {10, 0}, {10, 10}, {0, 10}}

	hom, err := SolveHomography4(src, dst)
	if err != nil {
		t.Fatalf("SolveHomography4: %v", err)
	}
	for i, s := range src {
		got := hom.Apply(s)
		if math.Abs(got.X-dst[i].X) > 1e-6 || math.Abs(got.Y-dst[i].Y) > 1e-6 {
			t.Errorf("Apply(%v) = %v, quero %v", s, got, dst[i])
		}
	}
}

func TestSolveHomography4Colineares(t *testing.T) {
	src := [4]Point{{0, 0}, {1, 0}, {2, 0}, {3, 0}} // todos na mesma reta
	dst := [4]Point{{0, 0}, {1, 1}, {2, 2}, {3, 3}}
	if _, err := SolveHomography4(src, dst); err == nil {
		t.Error("SolveHomography4 com cantos colineares deveria falhar")
	}
}

// TestHomographyInvertEIdaEVolta varre varias homografias e varios pontos:
// aplicar e depois inverter (ou o contrario) tem que devolver o ponto
// original. E o teste que realmente confere Invert -- comparar com uma
// formula escrita a mao seria so reescrever o mesmo calculo duas vezes.
func TestHomographyInvertEIdaEVolta(t *testing.T) {
	rnd := rand.New(rand.NewSource(7))

	for tentativa := 0; tentativa < 30; tentativa++ {
		// quatro cantos de um quadrilatero convexo simples, perturbados o
		// bastante para gerar perspectiva real mas sem colapsar em reta.
		src := [4]Point{{0, 0}, {100, 0}, {100, 100}, {0, 100}}
		dst := [4]Point{
			{10 + rnd.Float64()*10, 5 + rnd.Float64()*10},
			{90 + rnd.Float64()*15, 15 + rnd.Float64()*10},
			{95 + rnd.Float64()*10, 90 + rnd.Float64()*10},
			{5 + rnd.Float64()*10, 95 + rnd.Float64()*10},
		}
		hom, err := SolveHomography4(src, dst)
		if err != nil {
			t.Fatalf("tentativa %d: SolveHomography4: %v", tentativa, err)
		}
		inv, err := hom.Invert()
		if err != nil {
			t.Fatalf("tentativa %d: Invert: %v", tentativa, err)
		}

		for i := 0; i < 10; i++ {
			p := Point{X: rnd.Float64() * 100, Y: rnd.Float64() * 100}
			roundTrip := inv.Apply(hom.Apply(p))
			if p.Dist(roundTrip) > 1e-6 {
				t.Fatalf("tentativa %d: ida e volta de %v deu %v", tentativa, p, roundTrip)
			}
		}
	}
}

func TestHomographyInvertSingular(t *testing.T) {
	var h Homography // matriz zero, determinante zero
	if _, err := h.Invert(); err == nil {
		t.Error("Invert de matriz singular deveria falhar")
	}
}
