package imgproc

import (
	"math/rand"
	"testing"
)

// TestBoxBlurContraReferencia varre varios tamanhos de imagem e de raio e
// confere que a versao com imagem integral (O(1) por pixel) bate com a
// versao ingenua (O(raio^2) por pixel) ate o arredondamento de float32.
// Sem essa varredura, um erro de off-by-one na integral passaria batido:
// e o tipo de bug que so aparece em bordas ou em raios especificos.
func TestBoxBlurContraReferencia(t *testing.T) {
	rnd := rand.New(rand.NewSource(3))
	tamanhos := [][2]int{{1, 1}, {5, 1}, {1, 5}, {8, 8}, {13, 9}, {31, 17}}
	raios := []int{0, 1, 2, 3, 7, 20}

	for _, dim := range tamanhos {
		g := NewGray(dim[0], dim[1])
		for i := range g.Pix {
			g.Pix[i] = rnd.Float32()
		}
		for _, r := range raios {
			got := BoxBlur(g, r)
			want := BoxBlurRef(g, r)
			for i := range got.Pix {
				if abs32(got.Pix[i]-want.Pix[i]) > 1e-4 {
					t.Fatalf("dim %v raio %d: Pix[%d] = %v, quero %v (referencia)",
						dim, r, i, got.Pix[i], want.Pix[i])
				}
			}
		}
	}
}

func TestBoxBlurRaioZeroECopia(t *testing.T) {
	g := NewGray(4, 4)
	for i := range g.Pix {
		g.Pix[i] = float32(i)
	}
	out := BoxBlur(g, 0)
	for i := range g.Pix {
		if out.Pix[i] != g.Pix[i] {
			t.Fatalf("raio 0, Pix[%d] = %v, quero %v (copia identica)", i, out.Pix[i], g.Pix[i])
		}
	}
}

func TestBoxBlurImagemConstante(t *testing.T) {
	g := NewGray(10, 10)
	for i := range g.Pix {
		g.Pix[i] = 0.6
	}
	out := BoxBlur(g, 4)
	for i, v := range out.Pix {
		if abs32(v-0.6) > 1e-5 {
			t.Errorf("Pix[%d] = %v, quero 0.6 (media de constante e a propria constante)", i, v)
		}
	}
}

// TestNormalizeIlluminationAchataGradiente simula uma pagina com sombra: um
// gradiente suave de iluminacao (baixa frequencia) mais um "traco de texto"
// escuro pontual (alta frequencia). Depois de normalizar, o fundo longe do
// traco deve ficar perto de 1 em toda a imagem -- o teste que prova que o
// gradiente foi removido, nao so escurecido ou clareado por igual.
func TestNormalizeIlluminationAchataGradiente(t *testing.T) {
	const w, h = 100, 40
	g := NewGray(w, h)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			// gradiente de 0.3 a 0.9 da esquerda para a direita
			fundo := 0.3 + 0.6*float32(x)/float32(w-1)
			g.Set(x, y, fundo)
		}
	}
	// um "traco" escuro vertical no meio, estreito o bastante para nao
	// contaminar o blur de fundo (radius bem maior que sua largura).
	for y := 0; y < h; y++ {
		g.Set(w/2, y, 0.05)
	}

	out := NormalizeIllumination(g, 15, 1e-3)

	// Pontos longe do traco (que fica em w/2 = 50) E longe das bordas: o
	// blur de fundo, perto da borda, tem janela assimetrica (clampada de
	// um lado so) e a media sai puxada para o lado que sobrou -- nao e bug
	// de NormalizeIllumination, e a mesma ressalva que BoxBlur ja documenta
	// para janela perto da borda. Em janela simetrica sobre um gradiente
	// linear, a media bate exatamente com o valor do centro, entao a razao
	// aqui deveria ficar bem perto de 1, nao so "perto".
	for _, x := range []int{20, 30, 70, 80} {
		got := out.At(x, h/2)
		if abs32(got-1) > 0.01 {
			t.Errorf("x=%d: normalizado = %v, quero ~1 (gradiente removido)", x, got)
		}
	}
	// o traco continua bem mais escuro que o fundo ao redor, dividido pelo
	// mesmo fundo local -- a informacao de texto sobrevive a normalizacao.
	if got := out.At(w/2, h/2); got > 0.5 {
		t.Errorf("traco normalizado = %v, deveria continuar escuro (< 0.5)", got)
	}
}
