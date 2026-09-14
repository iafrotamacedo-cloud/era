package imgproc

// integral calcula a imagem integral (summed-area table) de g, com uma
// borda de zeros a esquerda e em cima -- a convencao classica que evita
// checar limite em cada consulta: sum.at(x,y) e a soma de todo pixel com
// coordenada estritamente menor que (x,y).
//
// Com ela, a soma de qualquer retangulo sai em 4 leituras e 3 somas,
// nao importa o tamanho do retangulo. E o que faz BoxBlur custar O(1) por
// pixel em vez de O(raio^2): ver reference.go para a versao ingenua com a
// qual este arquivo e conferido.
func integral(g *Gray) []float64 {
	w, h := g.W, g.H
	stride := w + 1
	sum := make([]float64, stride*(h+1))

	for y := 0; y < h; y++ {
		var linha float64
		for x := 0; x < w; x++ {
			linha += float64(g.At(x, y))
			sum[(y+1)*stride+(x+1)] = sum[y*stride+(x+1)] + linha
		}
	}
	return sum
}

// boxSum devolve a soma dos pixels no retangulo [x0,x1) x [y0,y1), com os
// quatro limites ja grampeados para dentro de [0,w] x [0,h] por quem chama.
func boxSum(sum []float64, stride, x0, y0, x1, y1 int) float64 {
	return sum[y1*stride+x1] - sum[y0*stride+x1] - sum[y1*stride+x0] + sum[y0*stride+x0]
}

// BoxBlur borra g com uma media movel de janela (2*radius+1) por
// (2*radius+1), grampeando na borda (o pixel fora da imagem repete o mais
// proximo, em vez de contar como zero -- do contrario a borda escureceria
// artificialmente).
//
// radius <= 0 devolve uma copia sem alteracao.
func BoxBlur(g *Gray, radius int) *Gray {
	dst := NewGray(g.W, g.H)
	if radius <= 0 {
		copy(dst.Pix, g.Pix)
		return dst
	}

	stride := g.W + 1
	sum := integral(g)

	for y := 0; y < g.H; y++ {
		y0 := clampInt(y-radius, 0, g.H)
		y1 := clampInt(y+radius+1, 0, g.H)
		for x := 0; x < g.W; x++ {
			x0 := clampInt(x-radius, 0, g.W)
			x1 := clampInt(x+radius+1, 0, g.W)
			area := float64((x1 - x0) * (y1 - y0))
			dst.Set(x, y, float32(boxSum(sum, stride, x0, y0, x1, y1)/area))
		}
	}
	return dst
}

func clampInt(v, lo, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

// NormalizeIllumination corrige sombra e gradiente de luz desigual.
//
// Papel curvo ou fotografado com luz lateral tem iluminacao que varia
// suavemente ao longo da pagina -- exatamente a frequencia espacial que um
// blur de raio grande captura e o texto (frequencia alta, traços finos) nao.
// Estimar o fundo com BoxBlur e dividir por ele remove o gradiente e deixa
// so o que muda rapido: a tinta.
//
// radius deve ser bem maior que a espessura de um traco de texto e menor
// que a escala da propria sombra -- tipicamente uma fracao da altura da
// imagem. epsilon evita divisao por fundo perto de zero (imagem quase toda
// preta).
//
// A saida nao fica limitada a [0,1]: tinta mais escura que o fundo local
// (o caso normal) desce, mas ruido acima do fundo pode passar de 1. Quem
// consumir o resultado como probabilidade ou pixel de exibicao precisa
// grampear; quem so compara contraste (o caso de deteccao de texto) nao.
func NormalizeIllumination(g *Gray, radius int, epsilon float32) *Gray {
	if epsilon <= 0 {
		epsilon = 1e-3
	}
	fundo := BoxBlur(g, radius)
	dst := NewGray(g.W, g.H)
	for i, v := range g.Pix {
		dst.Pix[i] = v / (fundo.Pix[i] + epsilon)
	}
	return dst
}
