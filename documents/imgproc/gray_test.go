package imgproc

import (
	"image"
	"image/color"
	"testing"
)

func TestNewGrayZerado(t *testing.T) {
	g := NewGray(4, 3)
	if g.W != 4 || g.H != 3 || g.Stride != 4 {
		t.Fatalf("dimensoes = %dx%d stride %d, quero 4x3 stride 4", g.W, g.H, g.Stride)
	}
	for i, v := range g.Pix {
		if v != 0 {
			t.Fatalf("NewGray deveria zerar, Pix[%d] = %v", i, v)
		}
	}
}

func TestAtSetGray(t *testing.T) {
	g := NewGray(5, 5)
	g.Set(2, 3, 0.75)
	if got := g.At(2, 3); got != 0.75 {
		t.Errorf("At(2,3) = %v, quero 0.75", got)
	}
	// posicao plana esperada: 3*5 + 2 = 17
	if g.Pix[17] != 0.75 {
		t.Errorf("Pix[17] = %v, stride errado?", g.Pix[17])
	}
}

func TestInBounds(t *testing.T) {
	g := NewGray(3, 2)
	casos := []struct {
		x, y int
		want bool
	}{
		{0, 0, true}, {2, 1, true},
		{-1, 0, false}, {3, 0, false},
		{0, -1, false}, {0, 2, false},
	}
	for _, c := range casos {
		if got := g.In(c.x, c.y); got != c.want {
			t.Errorf("In(%d,%d) = %v, quero %v", c.x, c.y, got, c.want)
		}
	}
}

func TestFromImagePretoBranco(t *testing.T) {
	src := image.NewRGBA(image.Rect(0, 0, 2, 1))
	src.Set(0, 0, color.RGBA{0, 0, 0, 255})
	src.Set(1, 0, color.RGBA{255, 255, 255, 255})

	g := FromImage(src)
	if g.W != 2 || g.H != 1 {
		t.Fatalf("dimensoes = %dx%d, quero 2x1", g.W, g.H)
	}
	if got := g.At(0, 0); got != 0 {
		t.Errorf("preto = %v, quero 0", got)
	}
	if got := g.At(1, 0); abs32(got-1) > 1e-6 {
		t.Errorf("branco = %v, quero ~1", got)
	}
}

// TestFromImageLuminancia confere os pesos BT.601 contra vermelho, verde e
// azul puros -- o caso onde um erro de peso (por exemplo trocar por BT.709)
// aparece na primeira casa decimal, nao no arredondamento.
func TestFromImageLuminancia(t *testing.T) {
	casos := []struct {
		nome    string
		c       color.RGBA
		wantLum float32
	}{
		{"vermelho puro", color.RGBA{255, 0, 0, 255}, 0.299},
		{"verde puro", color.RGBA{0, 255, 0, 255}, 0.587},
		{"azul puro", color.RGBA{0, 0, 255, 255}, 0.114},
		{"cinza medio", color.RGBA{128, 128, 128, 255}, 128.0 / 255.0},
	}
	for _, c := range casos {
		t.Run(c.nome, func(t *testing.T) {
			src := image.NewRGBA(image.Rect(0, 0, 1, 1))
			src.Set(0, 0, c.c)
			g := FromImage(src)
			if got := g.At(0, 0); abs32(got-c.wantLum) > 0.01 {
				t.Errorf("luminancia = %v, quero ~%v", got, c.wantLum)
			}
		})
	}
}

func abs32(v float32) float32 {
	if v < 0 {
		return -v
	}
	return v
}
