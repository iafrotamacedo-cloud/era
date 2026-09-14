package imgproc

import (
	"math/rand"
	"testing"
)

func TestSampleBilinearNosVertices(t *testing.T) {
	g := NewGray(4, 4)
	for y := 0; y < 4; y++ {
		for x := 0; x < 4; x++ {
			g.Set(x, y, float32(y*4+x))
		}
	}
	// Em coordenada inteira exata, bilinear deve devolver o pixel exato --
	// os quatro pesos colapsam em um so.
	for y := 0; y < 4; y++ {
		for x := 0; x < 4; x++ {
			got := SampleBilinear(g, float64(x), float64(y))
			want := g.At(x, y)
			if got != want {
				t.Errorf("SampleBilinear(%d,%d) = %v, quero %v", x, y, got, want)
			}
		}
	}
}

func TestSampleBilinearMeio(t *testing.T) {
	g := NewGray(2, 1)
	g.Set(0, 0, 0)
	g.Set(1, 0, 10)

	got := SampleBilinear(g, 0.5, 0)
	if abs32(got-5) > 1e-5 {
		t.Errorf("meio de 0 e 10 = %v, quero 5", got)
	}
}

// TestSampleBilinearClamp confere que amostrar fora da imagem repete a
// borda, em vez de extrapolar ou devolver zero -- e o comportamento que o
// pacote geom vai depender para remap perto das margens.
func TestSampleBilinearClamp(t *testing.T) {
	g := NewGray(3, 3)
	for y := 0; y < 3; y++ {
		for x := 0; x < 3; x++ {
			g.Set(x, y, float32(10+x+y))
		}
	}
	casos := []struct {
		nome string
		x, y float64
	}{
		{"muito a esquerda", -50, 1},
		{"muito a direita", 50, 1},
		{"muito acima", 1, -50},
		{"muito abaixo", 1, 50},
		{"canto fora", -10, -10},
	}
	for _, c := range casos {
		t.Run(c.nome, func(t *testing.T) {
			got := SampleBilinear(g, c.x, c.y)
			cx := clampInt(int(c.x), 0, 2)
			cy := clampInt(int(c.y), 0, 2)
			want := g.At(cx, cy)
			if got != want {
				t.Errorf("clamp(%v,%v) = %v, quero %v (pixel de borda)", c.x, c.y, got, want)
			}
		})
	}
}

// TestSampleBilinearExataEmRampaLinear usa a propriedade de que bilinear
// reproduz exatamente qualquer funcao linear (nao so nos vertices): para uma
// imagem cujo valor e uma rampa em x e y, amostrar em qualquer ponto interno
// tem de bater com a formula fechada da rampa, nao so aproximar.
func TestSampleBilinearExataEmRampaLinear(t *testing.T) {
	const w, h = 10, 8
	a, b, c := float32(1.5), float32(0.3), float32(-0.7) // f(x,y) = a + b*x + c*y
	g := NewGray(w, h)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			g.Set(x, y, a+b*float32(x)+c*float32(y))
		}
	}

	rnd := rand.New(rand.NewSource(1))
	for i := 0; i < 200; i++ {
		x := rnd.Float64() * (w - 1)
		y := rnd.Float64() * (h - 1)
		got := SampleBilinear(g, x, y)
		want := a + b*float32(x) + c*float32(y)
		if abs32(got-want) > 1e-4 {
			t.Fatalf("amostra(%.3f,%.3f) = %v, quero %v (rampa linear)", x, y, got, want)
		}
	}
}
