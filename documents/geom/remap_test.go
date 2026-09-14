package geom

import (
	"math"
	"math/rand"
	"testing"

	"github.com/iafrotamacedo-cloud/era/documents/imgproc"
)

func TestRemapIdentidade(t *testing.T) {
	rnd := rand.New(rand.NewSource(4))
	src := imgproc.NewGray(8, 6)
	for i := range src.Pix {
		src.Pix[i] = rnd.Float32()
	}

	out := Remap(src, 8, 6, func(dx, dy int) (float64, float64) {
		return float64(dx), float64(dy)
	})

	for i := range src.Pix {
		if math.Abs(float64(out.Pix[i]-src.Pix[i])) > 1e-6 {
			t.Fatalf("Pix[%d] = %v, quero %v (mapeamento identidade)", i, out.Pix[i], src.Pix[i])
		}
	}
}

// TestRemapHomographyEndireitaRetangulo fotografa (simuladamente) um
// retangulo perfeito sob uma homografia conhecida, produzindo um
// quadrilatero -- depois usa RemapHomography com os cantos desse
// quadrilatero para tentar recuperar o retangulo original. Se a
// composicao dos dois passos (gerar a "foto torta" e depois endireitar)
// bater com a fonte original, a homografia inversa foi montada certa.
func TestRemapHomographyEndireitaRetangulo(t *testing.T) {
	const w, h = 40, 30
	original := imgproc.NewGray(w, h)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			// padrao xadrez: fácil de notar se o remap sair deslocado.
			if (x/5+y/5)%2 == 0 {
				original.Set(x, y, 1)
			}
		}
	}

	// Homografia sintetica com perspectiva leve, mapeando o retangulo
	// original para um quadrilatero "fotografado".
	srcCorners := [4]Point{{0, 0}, {w - 1, 0}, {w - 1, h - 1}, {0, h - 1}}
	dstCorners := [4]Point{{3, 2}, {w + 4, 1}, {w - 2, h + 3}, {2, h - 4}}
	fwd, err := SolveHomography4(srcCorners, dstCorners)
	if err != nil {
		t.Fatalf("SolveHomography4: %v", err)
	}

	// "Fotografa": para cada pixel de uma tela maior (que cobre o
	// quadrilatero deslocado), busca o pixel correspondente no original.
	inv, err := fwd.Invert()
	if err != nil {
		t.Fatalf("Invert: %v", err)
	}
	const margem = 6
	foto := Remap(original, w+2*margem, h+2*margem, func(dx, dy int) (float64, float64) {
		p := inv.Apply(Point{X: float64(dx - margem), Y: float64(dy - margem)})
		return p.X, p.Y
	})

	// Agora endireita de volta, usando os cantos deslocados (+margem, ja
	// que "foto" tem essa borda extra).
	cantosNaFoto := [4]Point{
		{dstCorners[0].X + margem, dstCorners[0].Y + margem},
		{dstCorners[1].X + margem, dstCorners[1].Y + margem},
		{dstCorners[2].X + margem, dstCorners[2].Y + margem},
		{dstCorners[3].X + margem, dstCorners[3].Y + margem},
	}
	recuperado, err := RemapHomography(foto, cantosNaFoto, w, h)
	if err != nil {
		t.Fatalf("RemapHomography: %v", err)
	}

	// Compara longe das bordas -- amostragem bilinear repetida (foto,
	// depois recuperado) borra levemente as bordas dos quadrados do
	// xadrez, entao a comparacao pixel a pixel so e limpa no interior de
	// cada quadrado.
	var maxErro float32
	for y := 2; y < h-2; y++ {
		for x := 2; x < w-2; x++ {
			if x%5 < 1 || x%5 > 3 || y%5 < 1 || y%5 > 3 {
				continue // perto da borda de um quadrado do xadrez, pula
			}
			erro := recuperado.At(x, y) - original.At(x, y)
			if erro < 0 {
				erro = -erro
			}
			if erro > maxErro {
				maxErro = erro
			}
		}
	}
	if maxErro > 0.05 {
		t.Errorf("erro maximo apos ida e volta = %v, quero < 0.05", maxErro)
	}
}
