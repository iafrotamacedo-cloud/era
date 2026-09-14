// Package imgproc fornece o pre-processamento de imagem do ERA READ:
// decodificar, converter para cinza, normalizar iluminacao e amostrar.
//
// Tudo aqui trabalha em escala de cinza, em float32, com valores em [0,1].
// Cor nao ajuda a achar texto -- o contraste entre tinta e papel e uma
// questao de luminancia, nao de matiz -- e float32 em [0,1] e o formato que
// o resto da ERA ja usa (faces/tensor), entao a imagem chega pronta para
// virar tensor quando a fase de deteccao existir.
//
// Nada neste pacote conhece documento, papel ou texto. E processamento de
// imagem generico; o significado entra em pacotes posteriores.
package imgproc

import (
	"fmt"
	"image"
)

// Gray e uma imagem em escala de cinza, float32 em [0,1], linha a linha.
//
// O pixel (x, y) mora em Pix[y*Stride+x]. Guardar Stride em vez de assumir
// Stride == W permite que uma Gray seja uma janela de outra maior sem copiar
// dado -- o mesmo motivo pelo qual tensor.Tensor guarda Strides em vez de
// arrays aninhados.
type Gray struct {
	Pix    []float32
	W, H   int
	Stride int
}

// NewGray cria uma imagem cinza zerada (preta) de w por h.
func NewGray(w, h int) *Gray {
	if w <= 0 || h <= 0 {
		panic(fmt.Sprintf("imgproc: dimensoes invalidas %dx%d", w, h))
	}
	return &Gray{
		Pix:    make([]float32, w*h),
		W:      w,
		H:      h,
		Stride: w,
	}
}

// At devolve o pixel em (x, y). Fora dos limites e erro de quem chama --
// nao ha checagem, pelo mesmo motivo que Tensor.At nao checa: este e o
// caminho quente do pacote, chamado por pixel.
func (g *Gray) At(x, y int) float32 {
	return g.Pix[y*g.Stride+x]
}

// Set atribui o pixel em (x, y).
func (g *Gray) Set(x, y int, v float32) {
	g.Pix[y*g.Stride+x] = v
}

// In informa se (x, y) cai dentro da imagem.
func (g *Gray) In(x, y int) bool {
	return x >= 0 && x < g.W && y >= 0 && y < g.H
}

// pesosLuminancia sao os pesos ITU-R BT.601 para RGB -> luminancia.
//
// BT.601 (em vez do BT.709, mais novo) porque e o que o pacote image da
// stdlib usa em image/color.GrayModel -- manter o mesmo padrao evita que a
// mesma foto produza cinza ligeiramente diferente dependendo de qual
// caminho de conversao foi usado.
const (
	pesoR = 299.0 / 1000.0
	pesoG = 587.0 / 1000.0
	pesoB = 114.0 / 1000.0
)

// FromImage converte uma image.Image (o que image/png e image/jpeg
// decodificam) para Gray, em [0,1].
//
// RGBA() da stdlib devolve componentes em [0,65535] ja com o alpha
// pre-multiplicado. Documento fotografado nao tem transparencia real, mas
// alguns PNGs trazem canal alfa mesmo assim (opaco, alfa=65535) -- nesse
// caso o pre-multiplicado nao muda nada. Se algum dia entrar uma imagem com
// alfa parcial, o pixel sai mais escuro do que devia; nao e o caso de uso
// deste motor.
func FromImage(src image.Image) *Gray {
	b := src.Bounds()
	dst := NewGray(b.Dx(), b.Dy())

	const escala = 1.0 / 65535.0
	for y := b.Min.Y; y < b.Max.Y; y++ {
		for x := b.Min.X; x < b.Max.X; x++ {
			r, g, bl, _ := src.At(x, y).RGBA()
			lum := pesoR*float64(r) + pesoG*float64(g) + pesoB*float64(bl)
			dst.Set(x-b.Min.X, y-b.Min.Y, float32(lum*escala))
		}
	}
	return dst
}
