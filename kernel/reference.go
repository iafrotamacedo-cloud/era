package kernel

// Este arquivo guarda as implementacoes ingenuas -- lentas, obvias e
// dificeis de errar. Elas existem para servir de verdade: as versoes
// otimizadas sao conferidas contra elas nos testes.
//
// Nunca otimize nada aqui. O valor destas funcoes esta em serem tao simples
// que da para ler e afirmar que estao certas.

// MatMulRef calcula C = A x B da forma mais direta possivel.
//
//	A: m x k    B: k x n    C: m x n
func MatMulRef(a, b, c []float32, m, k, n int) {
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			var sum float32
			for kk := 0; kk < k; kk++ {
				sum += a[i*k+kk] * b[kk*n+j]
			}
			c[i*n+j] = sum
		}
	}
}

// Conv2DRef executa a convolucao 2D pela definicao, deslizando o kernel
// posicao a posicao. Sem im2col, sem matmul, sem bloqueio.
//
//	src:     [C, H, W]
//	weights: [outC, C/Groups, KH, KW]
//	bias:    [outC] ou nil
//	dst:     [outC, OutH, OutW]
func Conv2DRef(src, weights, bias []float32, outC int, p ConvParams, dst []float32) {
	p = p.norm()
	oh, ow := p.OutH(), p.OutW()
	inPerGroup := p.C / p.Groups
	outPerGroup := outC / p.Groups

	for oc := 0; oc < outC; oc++ {
		g := oc / outPerGroup // a qual grupo este canal de saida pertence

		for y := 0; y < oh; y++ {
			for x := 0; x < ow; x++ {
				var sum float32

				for ic := 0; ic < inPerGroup; ic++ {
					srcC := g*inPerGroup + ic

					for i := 0; i < p.KH; i++ {
						iy := y*p.StrideH - p.PadH + i*p.DilH
						if iy < 0 || iy >= p.H {
							continue // fora da imagem: o padding vale zero
						}

						for j := 0; j < p.KW; j++ {
							ix := x*p.StrideW - p.PadW + j*p.DilW
							if ix < 0 || ix >= p.W {
								continue
							}

							w := weights[((oc*inPerGroup+ic)*p.KH+i)*p.KW+j]
							sum += w * src[(srcC*p.H+iy)*p.W+ix]
						}
					}
				}

				if bias != nil {
					sum += bias[oc]
				}
				dst[(oc*oh+y)*ow+x] = sum
			}
		}
	}
}
