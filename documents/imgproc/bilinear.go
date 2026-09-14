package imgproc

import "math"

// SampleBilinear amostra g na posicao continua (x, y), interpolando entre
// os quatro pixels vizinhos.
//
// (x, y) fora da imagem e grampeado (clamp) na borda em vez de dar erro --
// e o comportamento certo para remap geometrico: uma homografia ou uma
// retificacao de linha rotineiramente pede um pixel a fracoes de distancia
// da borda, e clamp e a convencao que nao introduz uma borda preta artificial
// bem onde o dewarp mais precisa ser preciso (perto das margens do texto).
func SampleBilinear(g *Gray, x, y float64) float32 {
	// clamp para dentro de [0, W-1] x [0, H-1] antes de qualquer conta:
	// assim x0/x1 e y0/y1 abaixo ja saem validos, sem checagem por canto.
	if x < 0 {
		x = 0
	} else if x > float64(g.W-1) {
		x = float64(g.W - 1)
	}
	if y < 0 {
		y = 0
	} else if y > float64(g.H-1) {
		y = float64(g.H - 1)
	}

	x0 := int(math.Floor(x))
	y0 := int(math.Floor(y))
	x1 := x0 + 1
	y1 := y0 + 1
	if x1 > g.W-1 {
		x1 = g.W - 1
	}
	if y1 > g.H-1 {
		y1 = g.H - 1
	}

	fx := float32(x - float64(x0))
	fy := float32(y - float64(y0))

	top := g.At(x0, y0)*(1-fx) + g.At(x1, y0)*fx
	bot := g.At(x0, y1)*(1-fx) + g.At(x1, y1)*fx
	return top*(1-fy) + bot*fy
}
