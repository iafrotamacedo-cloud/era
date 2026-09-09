package align

import (
	"image"
	"image/color"
	"math"
	"math/rand"
	"testing"
)

func perto(a, b, tol float64) bool { return math.Abs(a-b) <= tol }

func pontosPertos(t *testing.T, got, want Landmarks, tol float64) {
	t.Helper()
	for i := range got {
		if !perto(got[i].X, want[i].X, tol) || !perto(got[i].Y, want[i].Y, tol) {
			t.Errorf("ponto %d = (%.4f, %.4f), quero (%.4f, %.4f)",
				i, got[i].X, got[i].Y, want[i].X, want[i].Y)
		}
	}
}

// aplicar leva todos os pontos por uma transformacao.
func aplicar(t Transform, l Landmarks) Landmarks {
	var out Landmarks
	for i, p := range l {
		out[i] = t.Apply(p)
	}
	return out
}

func TestTemplate(t *testing.T) {
	g := Template(112)
	if g != gabarito112 {
		t.Error("Template(112) deveria devolver o gabarito original sem mexer")
	}

	// Escala proporcional: dobrar o lado dobra as coordenadas.
	g224 := Template(224)
	for i := range g224 {
		if !perto(g224[i].X, gabarito112[i].X*2, 1e-9) ||
			!perto(g224[i].Y, gabarito112[i].Y*2, 1e-9) {
			t.Errorf("Template(224) ponto %d = %v, quero o dobro de %v", i, g224[i], gabarito112[i])
		}
	}

	// Sanidade do gabarito: os olhos na mesma altura aproximada, o nariz
	// abaixo deles, a boca abaixo do nariz.
	if !perto(g[OlhoDireito].Y, g[OlhoEsquerdo].Y, 1) {
		t.Error("os dois olhos deveriam estar quase na mesma altura")
	}
	if !(g[OlhoDireito].Y < g[Nariz].Y && g[Nariz].Y < g[BocaDireita].Y) {
		t.Error("a ordem vertical deveria ser olhos, nariz, boca")
	}
	if !(g[OlhoDireito].X < g[OlhoEsquerdo].X) {
		t.Error("o olho de indice 0 deveria estar a esquerda na imagem")
	}
}

// TestSimilarityRecuperaTransformacaoExata: quando os pontos de destino sao
// de fato uma similaridade dos de origem, o ajuste tem que recuperar a
// transformacao exatamente -- residuo zero.
func TestSimilarityRecuperaTransformacaoExata(t *testing.T) {
	base := Template(112)

	casos := []struct {
		nome string
		tr   Transform
	}{
		{"identidade", Identidade()},
		{"translacao", Transform{A: 1, B: 0, Tx: 37, Ty: -19}},
		{"escala 2x", Transform{A: 2, B: 0}},
		{"escala 0.5x", Transform{A: 0.5, B: 0}},
		{"rotacao 90 graus", Transform{A: 0, B: 1}},
		{"rotacao 30 graus", Transform{A: math.Cos(math.Pi / 6), B: math.Sin(math.Pi / 6)}},
		{"rotacao -45 graus", Transform{A: math.Cos(-math.Pi / 4), B: math.Sin(-math.Pi / 4)}},
		{"tudo junto", Transform{A: 1.3 * math.Cos(0.4), B: 1.3 * math.Sin(0.4), Tx: 100, Ty: -50}},
	}

	for _, c := range casos {
		t.Run(c.nome, func(t *testing.T) {
			destino := aplicar(c.tr, base)

			got, err := Similarity(base, destino)
			if err != nil {
				t.Fatalf("Similarity: %v", err)
			}

			for _, par := range [][2]float64{
				{got.A, c.tr.A}, {got.B, c.tr.B}, {got.Tx, c.tr.Tx}, {got.Ty, c.tr.Ty},
			} {
				if !perto(par[0], par[1], 1e-9) {
					t.Errorf("recuperou %+v, quero %+v", got, c.tr)
					break
				}
			}

			// E os pontos precisam cair exatamente em cima.
			pontosPertos(t, aplicar(got, base), destino, 1e-9)
		})
	}
}

func TestEscalaERotacao(t *testing.T) {
	casos := []struct {
		escala, angulo float64
	}{
		{1, 0}, {2, 0}, {0.5, math.Pi / 4}, {3, -math.Pi / 3}, {1, math.Pi / 2},
	}

	for _, c := range casos {
		tr := Transform{A: c.escala * math.Cos(c.angulo), B: c.escala * math.Sin(c.angulo)}
		if !perto(tr.Escala(), c.escala, 1e-12) {
			t.Errorf("Escala() = %v, quero %v", tr.Escala(), c.escala)
		}
		if !perto(tr.Rotacao(), c.angulo, 1e-12) {
			t.Errorf("Rotacao() = %v, quero %v", tr.Rotacao(), c.angulo)
		}
	}
}

// TestSimilarityNaoProduzReflexao cobre a propriedade que o uso de numeros
// complexos garante de graca. Um rosto espelhado seria uma solucao valida
// para o minimo quadrados, e uma catastrofe para o reconhecimento.
func TestSimilarityNaoProduzReflexao(t *testing.T) {
	base := Template(112)

	// Destino espelhado horizontalmente -- uma reflexao pura.
	var espelhado Landmarks
	for i, p := range base {
		espelhado[i] = Point{X: 112 - p.X, Y: p.Y}
	}

	tr, err := Similarity(base, espelhado)
	if err != nil {
		t.Fatal(err)
	}

	// O determinante de [A -B; B A] e A^2 + B^2, sempre positivo ou zero.
	// A forma da matriz simplesmente nao consegue representar reflexao, e o
	// ajuste devolve a melhor rotacao possivel em vez disso.
	det := tr.A*tr.A + tr.B*tr.B
	if det < 0 {
		t.Errorf("determinante %v: a transformacao refletiu", det)
	}
}

// TestSimilarityEMinimoQuadrados: com pontos que NAO sao uma similaridade
// exata, o ajuste precisa ser o melhor possivel. Conferimos comparando com
// perturbacoes da solucao encontrada.
func TestSimilarityEMinimoQuadrados(t *testing.T) {
	r := rand.New(rand.NewSource(42))
	base := Template(112)

	// Destino: uma similaridade conhecida mais ruido.
	verdadeira := Transform{A: 1.2 * math.Cos(0.3), B: 1.2 * math.Sin(0.3), Tx: 20, Ty: 10}
	destino := aplicar(verdadeira, base)
	for i := range destino {
		destino[i].X += r.NormFloat64() * 2
		destino[i].Y += r.NormFloat64() * 2
	}

	melhor, err := Similarity(base, destino)
	if err != nil {
		t.Fatal(err)
	}

	residuo := func(tr Transform) float64 {
		var s float64
		for i, p := range aplicar(tr, base) {
			dx := p.X - destino[i].X
			dy := p.Y - destino[i].Y
			s += dx*dx + dy*dy
		}
		return s
	}

	base_ := residuo(melhor)
	for i := 0; i < 200; i++ {
		perturbada := melhor
		perturbada.A += r.NormFloat64() * 0.02
		perturbada.B += r.NormFloat64() * 0.02
		perturbada.Tx += r.NormFloat64() * 0.5
		perturbada.Ty += r.NormFloat64() * 0.5

		if residuo(perturbada) < base_-1e-9 {
			t.Fatalf("achei uma transformacao melhor que a devolvida (%.6f < %.6f)",
				residuo(perturbada), base_)
		}
	}
}

func TestInverse(t *testing.T) {
	casos := []Transform{
		Identidade(),
		{A: 2, B: 0, Tx: 10, Ty: -5},
		{A: math.Cos(0.7), B: math.Sin(0.7), Tx: -30, Ty: 40},
		{A: 0.3 * math.Cos(-1.1), B: 0.3 * math.Sin(-1.1), Tx: 5, Ty: 5},
	}

	for _, tr := range casos {
		inv, err := tr.Inverse()
		if err != nil {
			t.Fatalf("Inverse de %v: %v", tr, err)
		}

		for _, p := range []Point{{0, 0}, {1, 0}, {0, 1}, {37.5, -12.25}, {112, 112}} {
			volta := inv.Apply(tr.Apply(p))
			if !perto(volta.X, p.X, 1e-9) || !perto(volta.Y, p.Y, 1e-9) {
				t.Errorf("%v: ida e volta de %v deu %v", tr, p, volta)
			}
		}
	}
}

func TestInverseDegeneradaDaErro(t *testing.T) {
	if _, err := (Transform{A: 0, B: 0, Tx: 5}).Inverse(); err == nil {
		t.Error("transformacao de escala zero nao tem inversa; deveria dar erro")
	}
}

func TestSimilarityPontosCoincidentesDaErro(t *testing.T) {
	var todosIguais Landmarks
	for i := range todosIguais {
		todosIguais[i] = Point{X: 50, Y: 50}
	}
	if _, err := Similarity(todosIguais, Template(112)); err == nil {
		t.Error("pontos coincidentes deveriam dar erro em vez de divisao por zero")
	}
}

// TestParaLevaAoGabarito e a propriedade que o alinhamento inteiro existe
// para garantir: seja qual for a posicao, o tamanho ou a inclinacao do rosto
// na foto, os 5 pontos vao parar nas posicoes canonicas.
func TestParaLevaAoGabarito(t *testing.T) {
	base := Template(112)

	casos := []Transform{
		{A: 3 * math.Cos(0.5), B: 3 * math.Sin(0.5), Tx: 400, Ty: 250},    // grande e torto
		{A: 0.4 * math.Cos(-0.2), B: 0.4 * math.Sin(-0.2), Tx: 10, Ty: 5}, // pequeno
		{A: math.Cos(1.2), B: math.Sin(1.2), Tx: 0, Ty: 0},                // bem rotacionado
	}

	for _, c := range casos {
		naFoto := aplicar(c, base)

		tr, err := Para(naFoto, 112)
		if err != nil {
			t.Fatal(err)
		}
		pontosPertos(t, aplicar(tr, naFoto), base, 1e-6)
	}
}

// ---------- recorte ----------

// imagemGradiente cria uma imagem em que a cor de cada pixel codifica a
// propria posicao. Assim da para conferir de onde cada pixel do recorte veio.
func imagemGradiente(w, h int) *image.NRGBA {
	img := image.NewNRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			img.Set(x, y, color.NRGBA{
				R: uint8(x % 256),
				G: uint8(y % 256),
				B: 128,
				A: 255,
			})
		}
	}
	return img
}

// TestCropIdentidade: com os pontos ja nas posicoes do gabarito, a
// transformacao e a identidade e o recorte tem que reproduzir a imagem.
func TestCropIdentidade(t *testing.T) {
	img := imagemGradiente(112, 112)

	out, err := Crop(img, Template(112), OpcoesPadrao())
	if err != nil {
		t.Fatalf("Crop: %v", err)
	}

	if want := []int{1, 3, 112, 112}; len(out.Shape) != 4 {
		t.Fatalf("forma = %v, quero %v", out.Shape, want)
	}

	plano := 112 * 112
	for _, p := range []struct{ x, y int }{{0, 0}, {5, 7}, {50, 60}, {111, 111}} {
		pos := p.y*112 + p.x
		r := out.Data[0*plano+pos]
		g := out.Data[1*plano+pos]
		b := out.Data[2*plano+pos]

		if !perto(float64(r), float64(p.x), 0.01) {
			t.Errorf("(%d,%d) canal R = %v, quero %d", p.x, p.y, r, p.x)
		}
		if !perto(float64(g), float64(p.y), 0.01) {
			t.Errorf("(%d,%d) canal G = %v, quero %d", p.x, p.y, g, p.y)
		}
		if !perto(float64(b), 128, 0.01) {
			t.Errorf("(%d,%d) canal B = %v, quero 128", p.x, p.y, b)
		}
	}
}

// TestCropDesfazTransformacao: o rosto e colocado na foto por uma
// transformacao conhecida, e o recorte tem que desfazer exatamente ela.
//
// A conta da expectativa merece cuidado, e errar nela custou uma falha. O
// centro do pixel (x,y) da saida esta em (x+0.5, y+0.5); levado a foto pela
// transformacao inversa, vira (s*(x+0.5)+Tx, s*(y+0.5)+Ty); e amostrar nessa
// coordenada continua significa interpolar em torno do INDICE
// (coordenada - 0.5). Com escala 1 isso cai exatamente em cima de um pixel;
// com escala 2, meio pixel na saida vira um pixel inteiro na entrada, e a
// amostragem cai no meio de dois.
//
// O gradiente tambem precisa caber em 8 bits sem dar a volta -- por isso a
// foto tem menos de 256 pixels de lado.
func TestCropDesfazTransformacao(t *testing.T) {
	plano := 112 * 112

	t.Run("escala 1: cai em cima do pixel", func(t *testing.T) {
		foto := imagemGradiente(250, 250)
		naFoto := aplicar(Transform{A: 1, B: 0, Tx: 60, Ty: 40}, Template(112))

		out, err := Crop(foto, naFoto, OpcoesPadrao())
		if err != nil {
			t.Fatalf("Crop: %v", err)
		}

		for _, p := range []struct{ x, y int }{{0, 0}, {10, 10}, {56, 56}, {100, 90}} {
			pos := p.y*112 + p.x
			wantR := float64(p.x + 60)
			wantG := float64(p.y + 40)

			if !perto(float64(out.Data[pos]), wantR, 0.01) {
				t.Errorf("(%d,%d) R = %v, quero %v", p.x, p.y, out.Data[pos], wantR)
			}
			if !perto(float64(out.Data[plano+pos]), wantG, 0.01) {
				t.Errorf("(%d,%d) G = %v, quero %v", p.x, p.y, out.Data[plano+pos], wantG)
			}
		}
	})

	t.Run("escala 2: interpola entre dois pixels", func(t *testing.T) {
		foto := imagemGradiente(250, 250)
		naFoto := aplicar(Transform{A: 2, B: 0, Tx: 20, Ty: 10}, Template(112))

		out, err := Crop(foto, naFoto, OpcoesPadrao())
		if err != nil {
			t.Fatalf("Crop: %v", err)
		}

		// indice amostrado = 2*(x+0.5) + Tx - 0.5 = 2x + Tx + 0.5
		for _, p := range []struct{ x, y int }{{0, 0}, {10, 10}, {56, 56}, {100, 90}} {
			pos := p.y*112 + p.x
			wantR := float64(2*p.x+20) + 0.5
			wantG := float64(2*p.y+10) + 0.5

			if !perto(float64(out.Data[pos]), wantR, 0.01) {
				t.Errorf("(%d,%d) R = %v, quero %v", p.x, p.y, out.Data[pos], wantR)
			}
			if !perto(float64(out.Data[plano+pos]), wantG, 0.01) {
				t.Errorf("(%d,%d) G = %v, quero %v", p.x, p.y, out.Data[plano+pos], wantG)
			}
		}
	})
}

func TestCropBGR(t *testing.T) {
	img := image.NewNRGBA(image.Rect(0, 0, 112, 112))
	for y := 0; y < 112; y++ {
		for x := 0; x < 112; x++ {
			img.Set(x, y, color.NRGBA{R: 10, G: 20, B: 30, A: 255})
		}
	}

	rgb, err := Crop(img, Template(112), OpcoesPadrao())
	if err != nil {
		t.Fatal(err)
	}
	bgr, err := Crop(img, Template(112), Options{Size: 112, Scale: [3]float32{1, 1, 1}, BGR: true})
	if err != nil {
		t.Fatal(err)
	}

	plano := 112 * 112
	if rgb.Data[0] != 10 || rgb.Data[plano] != 20 || rgb.Data[2*plano] != 30 {
		t.Errorf("RGB = %v %v %v, quero 10 20 30", rgb.Data[0], rgb.Data[plano], rgb.Data[2*plano])
	}
	if bgr.Data[0] != 30 || bgr.Data[plano] != 20 || bgr.Data[2*plano] != 10 {
		t.Errorf("BGR = %v %v %v, quero 30 20 10", bgr.Data[0], bgr.Data[plano], bgr.Data[2*plano])
	}
}

func TestCropMediaEEscala(t *testing.T) {
	img := image.NewNRGBA(image.Rect(0, 0, 112, 112))
	for y := 0; y < 112; y++ {
		for x := 0; x < 112; x++ {
			img.Set(x, y, color.NRGBA{R: 200, G: 100, B: 50, A: 255})
		}
	}

	// (v - media) * escala, por canal.
	out, err := Crop(img, Template(112), Options{
		Size:  112,
		Mean:  [3]float32{100, 100, 100},
		Scale: [3]float32{0.5, 2, 1},
	})
	if err != nil {
		t.Fatal(err)
	}

	plano := 112 * 112
	quero := []float32{(200 - 100) * 0.5, (100 - 100) * 2, (50 - 100) * 1}
	for c, w := range quero {
		if got := out.Data[c*plano]; got != w {
			t.Errorf("canal %d = %v, quero %v", c, got, w)
		}
	}
}

// TestCropReplicaBorda: rosto encostado no limite da foto e comum em foto de
// documento. Devolver preto criaria uma faixa escura que a rede nunca viu.
func TestCropReplicaBorda(t *testing.T) {
	img := image.NewNRGBA(image.Rect(0, 0, 112, 112))
	for y := 0; y < 112; y++ {
		for x := 0; x < 112; x++ {
			img.Set(x, y, color.NRGBA{R: 200, G: 200, B: 200, A: 255})
		}
	}

	// Puxa o rosto para fora da imagem: metade do recorte cai fora.
	fora := aplicar(Transform{A: 1, Tx: -60, Ty: -60}, Template(112))

	out, err := Crop(img, fora, OpcoesPadrao())
	if err != nil {
		t.Fatal(err)
	}

	// Nenhum pixel pode ter virado preto: a borda e replicada.
	for i, v := range out.Data {
		if v < 190 {
			t.Fatalf("posicao %d = %v; a borda deveria ter sido replicada, nao zerada", i, v)
		}
	}
}

func TestCropAceitaVariosFormatos(t *testing.T) {
	const w, h = 112, 112

	rgba := image.NewRGBA(image.Rect(0, 0, w, h))
	nrgba := image.NewNRGBA(image.Rect(0, 0, w, h))
	cinza := image.NewGray(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			rgba.Set(x, y, color.RGBA{R: 90, G: 90, B: 90, A: 255})
			nrgba.Set(x, y, color.NRGBA{R: 90, G: 90, B: 90, A: 255})
			cinza.Set(x, y, color.Gray{Y: 90})
		}
	}

	// image.YCbCr e o que a decodificacao de JPEG produz.
	ycbcr := image.NewYCbCr(image.Rect(0, 0, w, h), image.YCbCrSubsampleRatio444)
	for i := range ycbcr.Y {
		ycbcr.Y[i] = 90
	}
	for i := range ycbcr.Cb {
		ycbcr.Cb[i] = 128
		ycbcr.Cr[i] = 128
	}

	casos := []struct {
		nome string
		img  image.Image
	}{
		{"RGBA", rgba},
		{"NRGBA", nrgba},
		{"Gray", cinza},
		{"YCbCr", ycbcr},
		{"generica", umaImagemQualquer{w, h}},
	}

	for _, c := range casos {
		t.Run(c.nome, func(t *testing.T) {
			out, err := Crop(c.img, Template(112), OpcoesPadrao())
			if err != nil {
				t.Fatalf("Crop: %v", err)
			}
			// Cinza 90 nos tres canais, com folga para o arredondamento da
			// conversao de YCbCr.
			for c := 0; c < 3; c++ {
				got := out.Data[c*112*112+50*112+50]
				if !perto(float64(got), 90, 2) {
					t.Errorf("canal %d = %v, quero ~90", c, got)
				}
			}
		})
	}
}

// umaImagemQualquer implementa image.Image sem ser nenhum dos tipos com
// caminho rapido, exercitando o caminho generico via At().
type umaImagemQualquer struct{ w, h int }

func (i umaImagemQualquer) ColorModel() color.Model { return color.RGBAModel }
func (i umaImagemQualquer) Bounds() image.Rectangle { return image.Rect(0, 0, i.w, i.h) }
func (i umaImagemQualquer) At(x, y int) color.Color {
	return color.RGBA{R: 90, G: 90, B: 90, A: 255}
}

func TestCropRecusaEntradaInvalida(t *testing.T) {
	if _, err := Crop(nil, Template(112), OpcoesPadrao()); err == nil {
		t.Error("imagem nula deveria dar erro")
	}
	vazia := image.NewNRGBA(image.Rect(0, 0, 0, 0))
	if _, err := Crop(vazia, Template(112), OpcoesPadrao()); err == nil {
		t.Error("imagem vazia deveria dar erro")
	}

	var coincidentes Landmarks
	img := imagemGradiente(112, 112)
	if _, err := Crop(img, coincidentes, OpcoesPadrao()); err == nil {
		t.Error("pontos coincidentes deveriam dar erro")
	}
}

func TestCropImage(t *testing.T) {
	foto := imagemGradiente(200, 200)
	naFoto := aplicar(Transform{A: 1, Tx: 30, Ty: 20}, Template(112))

	out, err := CropImage(foto, naFoto, 112)
	if err != nil {
		t.Fatalf("CropImage: %v", err)
	}
	if b := out.Bounds(); b.Dx() != 112 || b.Dy() != 112 {
		t.Errorf("tamanho = %v, quero 112x112", b)
	}

	// O pixel (10,10) do recorte veio de (40,30) da foto.
	r, g, _, a := out.At(10, 10).RGBA()
	if got := int(r / 257); !perto(float64(got), 40, 1) {
		t.Errorf("R = %d, quero ~40", got)
	}
	if got := int(g / 257); !perto(float64(got), 30, 1) {
		t.Errorf("G = %d, quero ~30", got)
	}
	if a>>8 != 255 {
		t.Errorf("alfa = %d, quero 255", a>>8)
	}
}

func TestOpcoesPadraoENormalizacao(t *testing.T) {
	o := Options{}.normalizada()
	if o.Size != 112 {
		t.Errorf("Size = %d, quero 112", o.Size)
	}
	for i, s := range o.Scale {
		if s != 1 {
			t.Errorf("Scale[%d] = %v, quero 1", i, s)
		}
	}

	if s := OpcoesSFace(); !s.BGR || s.Size != 112 {
		t.Errorf("OpcoesSFace = %+v, quero BGR em 112", s)
	}
}

func BenchmarkCrop(b *testing.B) {
	foto := imagemGradiente(1280, 720)
	pontos := aplicar(Transform{A: 2, B: 0.3, Tx: 500, Ty: 300}, Template(112))
	opt := OpcoesPadrao()

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := Crop(foto, pontos, opt); err != nil {
			b.Fatal(err)
		}
	}
}
