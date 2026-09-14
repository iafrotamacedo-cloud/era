package imgproc

// Este arquivo guarda as implementacoes de referencia -- lentas e obvias,
// para servir de verdade nos testes. Nunca otimize nada aqui.

// BoxBlurRef calcula a mesma media movel que BoxBlur, mas somando cada
// janela do zero, pixel por pixel: O(raio^2) por pixel de saida em vez de
// O(1). Existe so para o teste confirmar que a imagem integral em BoxBlur
// nao introduziu erro.
func BoxBlurRef(g *Gray, radius int) *Gray {
	dst := NewGray(g.W, g.H)
	if radius <= 0 {
		copy(dst.Pix, g.Pix)
		return dst
	}

	for y := 0; y < g.H; y++ {
		y0 := clampInt(y-radius, 0, g.H-1)
		y1 := clampInt(y+radius, 0, g.H-1)
		for x := 0; x < g.W; x++ {
			x0 := clampInt(x-radius, 0, g.W-1)
			x1 := clampInt(x+radius, 0, g.W-1)

			var soma float64
			var n int
			for yy := y0; yy <= y1; yy++ {
				for xx := x0; xx <= x1; xx++ {
					soma += float64(g.At(xx, yy))
					n++
				}
			}
			dst.Set(x, y, float32(soma/float64(n)))
		}
	}
	return dst
}
