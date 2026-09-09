package align

import (
	"fmt"
	"image"
	"math"

	"github.com/iafrotamacedo-cloud/era/faces/tensor"
)

// Options configura como o recorte vira tensor.
//
// Os valores de Mean, Scale e BGR dependem do MODELO, nao da biblioteca:
// cada rede foi treinada esperando os pixels numa faixa e numa ordem de
// canais. Usar os valores errados nao produz erro -- produz um vetor de
// identidade sutilmente errado, que e bem pior.
type Options struct {
	// Size e o lado do recorte quadrado. Zero vale 112, que e o que
	// praticamente todo modelo de reconhecimento facial espera.
	Size int

	// Mean e subtraido de cada canal, antes da escala.
	Mean [3]float32

	// Scale multiplica cada canal, depois da subtracao. Zero vale 1.
	Scale [3]float32

	// BGR inverte a ordem dos canais. Modelos vindos do mundo OpenCV
	// costumam esperar BGR; os do mundo PyTorch, RGB.
	BGR bool
}

// OpcoesPadrao devolve pixels crus de 0 a 255, em RGB.
func OpcoesPadrao() Options {
	return Options{Size: 112, Scale: [3]float32{1, 1, 1}}
}

// OpcoesSFace devolve o pre-processamento que o SFace do OpenCV Zoo espera:
// BGR, valores de 0 a 255, sem normalizacao.
//
// ATENCAO: estes valores vieram da leitura de como o OpenCV alimenta o
// modelo, e ainda NAO foram conferidos contra uma imagem real passando pelos
// dois caminhos. Pre-processamento errado nao da erro -- da um vetor
// sutilmente errado. Confira antes de usar em producao.
func OpcoesSFace() Options {
	return Options{Size: 112, Scale: [3]float32{1, 1, 1}, BGR: true}
}

// normalizada preenche os campos zerados com os padroes.
func (o Options) normalizada() Options {
	if o.Size == 0 {
		o.Size = 112
	}
	for i := range o.Scale {
		if o.Scale[i] == 0 {
			o.Scale[i] = 1
		}
	}
	return o
}

// Crop recorta e endireita o rosto, devolvendo o tensor [1,3,S,S] pronto
// para a rede.
func Crop(img image.Image, pontos Landmarks, opt Options) (*tensor.Tensor, error) {
	opt = opt.normalizada()
	if opt.Size <= 0 {
		return nil, fmt.Errorf("align: tamanho invalido (%d)", opt.Size)
	}

	t, err := Para(pontos, opt.Size)
	if err != nil {
		return nil, err
	}
	inv, err := t.Inverse()
	if err != nil {
		return nil, err
	}

	am, err := novoAmostrador(img)
	if err != nil {
		return nil, err
	}

	s := opt.Size
	out := tensor.New(1, 3, s, s)
	plano := s * s

	// Ordem dos canais na saida.
	iR, iG, iB := 0, 1, 2
	if opt.BGR {
		iR, iB = 2, 0
	}

	for y := 0; y < s; y++ {
		for x := 0; x < s; x++ {
			// O centro do pixel de saida, levado de volta a imagem original.
			// O meio-pixel importa: sem ele o recorte sai deslocado meio
			// pixel, e o erro aparece na terceira casa do vetor.
			p := inv.Apply(Point{X: float64(x) + 0.5, Y: float64(y) + 0.5})

			r, g, b := am.bilinear(p.X-0.5, p.Y-0.5)
			pos := y*s + x

			out.Data[iR*plano+pos] = (float32(r) - opt.Mean[0]) * opt.Scale[0]
			out.Data[iG*plano+pos] = (float32(g) - opt.Mean[1]) * opt.Scale[1]
			out.Data[iB*plano+pos] = (float32(b) - opt.Mean[2]) * opt.Scale[2]
		}
	}

	return out, nil
}

// CropImage e como Crop, mas devolve uma imagem em vez de um tensor.
//
// Serve para conferir o alinhamento com os proprios olhos, que e o jeito
// mais rapido de descobrir que os pontos vieram trocados ou que o gabarito
// esta errado.
func CropImage(img image.Image, pontos Landmarks, tamanho int) (*image.RGBA, error) {
	if tamanho <= 0 {
		tamanho = 112
	}

	t, err := Para(pontos, tamanho)
	if err != nil {
		return nil, err
	}
	inv, err := t.Inverse()
	if err != nil {
		return nil, err
	}

	am, err := novoAmostrador(img)
	if err != nil {
		return nil, err
	}

	out := image.NewRGBA(image.Rect(0, 0, tamanho, tamanho))
	for y := 0; y < tamanho; y++ {
		for x := 0; x < tamanho; x++ {
			p := inv.Apply(Point{X: float64(x) + 0.5, Y: float64(y) + 0.5})
			r, g, b := am.bilinear(p.X-0.5, p.Y-0.5)

			i := out.PixOffset(x, y)
			out.Pix[i+0] = clampByte(r)
			out.Pix[i+1] = clampByte(g)
			out.Pix[i+2] = clampByte(b)
			out.Pix[i+3] = 255
		}
	}
	return out, nil
}

func clampByte(v float64) uint8 {
	switch {
	case v <= 0:
		return 0
	case v >= 255:
		return 255
	}
	return uint8(v + 0.5)
}

// amostrador le pixels de uma image.Image com caminho rapido para os
// formatos mais comuns.
//
// A interface image.Image so oferece At(), que devolve uma cor por chamada
// de interface e valores de 16 bits pre-multiplicados. Para um recorte de
// 112x112 sao 50 mil dessas chamadas, e o caminho rapido evita quase todas.
type amostrador struct {
	limites  image.Rectangle
	generica image.Image
	rgba     *image.RGBA
	nrgba    *image.NRGBA
	cinza    *image.Gray
	ycbcr    *image.YCbCr
}

func novoAmostrador(img image.Image) (*amostrador, error) {
	if img == nil {
		return nil, fmt.Errorf("align: imagem nula")
	}
	b := img.Bounds()
	if b.Dx() <= 0 || b.Dy() <= 0 {
		return nil, fmt.Errorf("align: imagem vazia (%v)", b)
	}

	a := &amostrador{limites: b}
	switch v := img.(type) {
	case *image.RGBA:
		a.rgba = v
	case *image.NRGBA:
		a.nrgba = v
	case *image.Gray:
		a.cinza = v
	case *image.YCbCr: // o que a decodificacao de JPEG produz
		a.ycbcr = v
	default:
		a.generica = img
	}
	return a, nil
}

// pixel le um pixel, replicando a borda para coordenadas fora da imagem.
//
// Replicar a borda -- em vez de devolver preto -- evita uma faixa escura no
// recorte quando o rosto encosta no limite da foto, que e comum em foto de
// documento e em quadro de camera.
func (a *amostrador) pixel(x, y int) (float64, float64, float64) {
	if x < a.limites.Min.X {
		x = a.limites.Min.X
	}
	if x >= a.limites.Max.X {
		x = a.limites.Max.X - 1
	}
	if y < a.limites.Min.Y {
		y = a.limites.Min.Y
	}
	if y >= a.limites.Max.Y {
		y = a.limites.Max.Y - 1
	}

	switch {
	case a.rgba != nil:
		i := a.rgba.PixOffset(x, y)
		p := a.rgba.Pix[i : i+4 : i+4]
		// RGBA guarda os canais pre-multiplicados pelo alfa; desfazer isso
		// e o que mantem a cor certa em imagem com transparencia.
		if p[3] == 0 {
			return 0, 0, 0
		}
		if p[3] == 255 {
			return float64(p[0]), float64(p[1]), float64(p[2])
		}
		k := 255 / float64(p[3])
		return float64(p[0]) * k, float64(p[1]) * k, float64(p[2]) * k

	case a.nrgba != nil:
		i := a.nrgba.PixOffset(x, y)
		p := a.nrgba.Pix[i : i+3 : i+3]
		return float64(p[0]), float64(p[1]), float64(p[2])

	case a.cinza != nil:
		v := float64(a.cinza.Pix[a.cinza.PixOffset(x, y)])
		return v, v, v

	case a.ycbcr != nil:
		r, g, b := ycbcrParaRGB(a.ycbcr, x, y)
		return float64(r), float64(g), float64(b)
	}

	r, g, b, _ := a.generica.At(x, y).RGBA()
	// RGBA() devolve 16 bits pre-multiplicados; 257 = 65535/255.
	return float64(r) / 257, float64(g) / 257, float64(b) / 257
}

// ycbcrParaRGB converte um pixel do formato que o decodificador de JPEG
// produz, sem passar pela interface color.Color.
func ycbcrParaRGB(img *image.YCbCr, x, y int) (uint8, uint8, uint8) {
	yi := img.YOffset(x, y)
	ci := img.COffset(x, y)

	yy := int32(img.Y[yi]) * 0x10101
	cb := int32(img.Cb[ci]) - 128
	cr := int32(img.Cr[ci]) - 128

	r := yy + 91881*cr
	g := yy - 22554*cb - 46802*cr
	b := yy + 116130*cb

	return sat16(r), sat16(g), sat16(b)
}

func sat16(v int32) uint8 {
	if uint32(v)&0xff000000 == 0 {
		return uint8(v >> 16)
	}
	if v < 0 {
		return 0
	}
	return 255
}

// bilinear amostra em coordenada fracionaria, interpolando os 4 pixels
// vizinhos.
//
// Sem interpolacao, o recorte de um rosto rotacionado fica serrilhado, e o
// serrilhado e ruido que a rede nunca viu no treino.
func (a *amostrador) bilinear(x, y float64) (float64, float64, float64) {
	x0 := math.Floor(x)
	y0 := math.Floor(y)
	fx := x - x0
	fy := y - y0

	ix, iy := int(x0), int(y0)

	r00, g00, b00 := a.pixel(ix, iy)
	r10, g10, b10 := a.pixel(ix+1, iy)
	r01, g01, b01 := a.pixel(ix, iy+1)
	r11, g11, b11 := a.pixel(ix+1, iy+1)

	// Pesos dos quatro cantos.
	w00 := (1 - fx) * (1 - fy)
	w10 := fx * (1 - fy)
	w01 := (1 - fx) * fy
	w11 := fx * fy

	return r00*w00 + r10*w10 + r01*w01 + r11*w11,
		g00*w00 + g10*w10 + g01*w01 + g11*w11,
		b00*w00 + b10*w10 + b01*w01 + b11*w11
}
