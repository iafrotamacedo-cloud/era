package geom

import "github.com/iafrotamacedo-cloud/era/documents/imgproc"

// Remap gera uma imagem dstW x dstH amostrando src atraves de um
// mapeamento inverso: invMap(dstX, dstY) devolve a coordenada de origem
// (em indice de pixel, na mesma convencao de imgproc.SampleBilinear --
// inteiro cai exatamente no centro de um pixel) correspondente a cada
// pixel de destino.
//
// O mapeamento e inverso -- destino para origem, nao origem para destino
// -- porque e assim que se evita buraco no resultado: percorrer a origem e
// "jogar" cada pixel no destino deixaria posicoes de destino sem nenhum
// pixel caindo nelas (sobretudo ao ampliar). Percorrer o destino e "puxar"
// da origem garante que todo pixel de saida seja preenchido, e a
// amostragem bilinear cuida do que cai entre pixels da origem.
func Remap(src *imgproc.Gray, dstW, dstH int, invMap func(dstX, dstY int) (srcX, srcY float64)) *imgproc.Gray {
	dst := imgproc.NewGray(dstW, dstH)
	for dy := 0; dy < dstH; dy++ {
		for dx := 0; dx < dstW; dx++ {
			sx, sy := invMap(dx, dy)
			dst.Set(dx, dy, imgproc.SampleBilinear(src, sx, sy))
		}
	}
	return dst
}

// RemapHomography e o nivel N1 completo: endireita o quadrilatero definido
// por corners (fotografado em perspectiva, cantos na ordem
// superior-esquerdo, superior-direito, inferior-direito, inferior-esquerdo)
// para um retangulo dstW x dstH.
//
// A homografia resolvida mapeia diretamente destino -> origem (dst para
// corners), que e exatamente o mapeamento inverso que Remap precisa -- por
// isso SolveHomography4 e chamado com os cantos do retangulo de saida como
// "origem" da formula e os cantos fotografados como "destino": o nome dos
// parametros de SolveHomography4 e generico, o sentido de uso e que muda.
func RemapHomography(src *imgproc.Gray, corners [4]Point, dstW, dstH int) (*imgproc.Gray, error) {
	retangulo := [4]Point{
		{X: 0, Y: 0},
		{X: float64(dstW - 1), Y: 0},
		{X: float64(dstW - 1), Y: float64(dstH - 1)},
		{X: 0, Y: float64(dstH - 1)},
	}
	hom, err := SolveHomography4(retangulo, corners)
	if err != nil {
		return nil, err
	}
	return Remap(src, dstW, dstH, func(dx, dy int) (float64, float64) {
		p := hom.Apply(Point{X: float64(dx), Y: float64(dy)})
		return p.X, p.Y
	}), nil
}
