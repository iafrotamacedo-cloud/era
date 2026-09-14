package imgproc

import (
	"math/rand"
	"testing"
)

// TestResizeMesmoTamanhoEIdentidade e o teste mais importante deste
// arquivo: redimensionar para o proprio tamanho tem de devolver a mesma
// imagem, pixel a pixel. Se o alinhamento de grade em Resize estiver
// errado (por exemplo, mapear cantos em vez de centros de pixel), este
// teste falha mesmo que "pareça razoavel" visualmente.
func TestResizeMesmoTamanhoEIdentidade(t *testing.T) {
	rnd := rand.New(rand.NewSource(2))
	g := NewGray(7, 5)
	for i := range g.Pix {
		g.Pix[i] = rnd.Float32()
	}

	out := Resize(g, 7, 5)
	for i := range g.Pix {
		if abs32(out.Pix[i]-g.Pix[i]) > 1e-5 {
			t.Fatalf("Pix[%d] = %v, quero %v (resize p/ mesmo tamanho nao e identidade)", i, out.Pix[i], g.Pix[i])
		}
	}
}

func TestResizeImagemConstante(t *testing.T) {
	g := NewGray(6, 6)
	for i := range g.Pix {
		g.Pix[i] = 0.42
	}
	for _, dim := range [][2]int{{3, 3}, {12, 12}, {1, 1}, {20, 5}} {
		out := Resize(g, dim[0], dim[1])
		for i, v := range out.Pix {
			if abs32(v-0.42) > 1e-5 {
				t.Errorf("resize p/ %dx%d, Pix[%d] = %v, quero 0.42 (imagem constante)", dim[0], dim[1], i, v)
			}
		}
	}
}

func TestResize1x1EMediaAproximada(t *testing.T) {
	g := NewGray(2, 2)
	g.Set(0, 0, 0)
	g.Set(1, 0, 1)
	g.Set(0, 1, 0)
	g.Set(1, 1, 1)

	out := Resize(g, 1, 1)
	if out.W != 1 || out.H != 1 {
		t.Fatalf("dimensoes = %dx%d, quero 1x1", out.W, out.H)
	}
	// bilinear no centro de uma grade 2x2 com metade 0 e metade 1 da 0.5.
	if got := out.At(0, 0); abs32(got-0.5) > 1e-5 {
		t.Errorf("Resize 1x1 = %v, quero ~0.5", got)
	}
}
