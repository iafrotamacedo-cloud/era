package imgproc

// Resize devolve uma copia de g redimensionada para newW por newH, por
// amostragem bilinear.
//
// Sem filtro de pre-suavizacao ao encolher: para reduzir mais que ~2x, o
// certo seria uma media de area (ou um blur antes do resize) para evitar
// aliasing. Este motor redimensiona sobretudo para levar um recorte de
// linha ate a altura fixa que o reconhecedor espera -- uma reducao leve,
// nao um thumbnail -- entao o custo de aliasing nao compensa a complexidade
// ainda. Revisitar se aparecer perda de qualidade em reducoes fortes.
func Resize(g *Gray, newW, newH int) *Gray {
	dst := NewGray(newW, newH)
	if newW == 1 && newH == 1 {
		dst.Set(0, 0, SampleBilinear(g, float64(g.W-1)/2, float64(g.H-1)/2))
		return dst
	}

	// Mapeia o centro de cada pixel de destino para a coordenada continua
	// correspondente na origem, alinhando os cantos das duas grades (nao
	// so as origens) -- e o que faz redimensionar para o mesmo tamanho ser
	// a identidade.
	scaleX := float64(g.W) / float64(newW)
	scaleY := float64(g.H) / float64(newH)

	for dy := 0; dy < newH; dy++ {
		sy := (float64(dy) + 0.5) * scaleY
		for dx := 0; dx < newW; dx++ {
			sx := (float64(dx) + 0.5) * scaleX
			dst.Set(dx, dy, SampleBilinear(g, sx-0.5, sy-0.5))
		}
	}
	return dst
}
