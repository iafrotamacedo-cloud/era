package kernel

import "fmt"

// Convolucao depthwise: um filtro por canal, sem misturar canais.
//
// Passar por Im2Col + MatMul aqui e um erro de desenho, ainda que produza o
// resultado certo. Com Groups == C, cada grupo vira uma matmul de UMA linha:
// o paralelismo por linhas nao tem o que dividir, e o rearranjo de memoria
// do im2col e pago sem que a matmul tenha volume para amortiza-lo. Medido:
// 0,86 GFLOPS, contra ~12 das demais camadas.
//
// A saida e percorrer o kernel direto sobre cada canal, paralelizando por
// canal em vez de por linha. E o que ncnn, MNN e TFLite fazem -- depthwise
// e sempre caso especial.
//
// Isso importa porque MobileFaceNet, o modelo que a ERA tem como alvo, e
// feito majoritariamente de depthwise.

// DepthwiseConv2D executa a convolucao depthwise: cada canal de entrada e
// convoluido com o seu proprio filtro, e nada se mistura.
//
//	src:     [C, H, W]
//	weights: [C, 1, KH, KW]  (mesmo layout de Conv2D com Groups == C)
//	bias:    [C] ou nil
//	dst:     [C, OutH, OutW]
//
// Exige p.Groups == p.C. Conv2D chama esta funcao sozinha quando reconhece
// o caso, entao normalmente nao ha motivo para chamar diretamente.
func DepthwiseConv2D(src, weights, bias []float32, p ConvParams, dst []float32) error {
	if err := p.Validate(); err != nil {
		return err
	}
	p = p.norm()

	if p.Groups != p.C {
		return fmt.Errorf("kernel: depthwise exige Groups == C, recebi Groups=%d C=%d", p.Groups, p.C)
	}

	oh, ow := p.OutH(), p.OutW()
	spatial := oh * ow
	kSize := p.KH * p.KW

	if len(src) < p.C*p.H*p.W {
		return fmt.Errorf("kernel: src tem %d elementos, precisa de %d", len(src), p.C*p.H*p.W)
	}
	if len(weights) < p.C*kSize {
		return fmt.Errorf("kernel: weights tem %d elementos, precisa de %d", len(weights), p.C*kSize)
	}
	if bias != nil && len(bias) < p.C {
		return fmt.Errorf("kernel: bias tem %d elementos, precisa de %d", len(bias), p.C)
	}
	if len(dst) < p.C*spatial {
		return fmt.Errorf("kernel: dst tem %d elementos, precisa de %d", len(dst), p.C*spatial)
	}

	// Um canal nunca toca o outro, entao a divisao por canais e naturalmente
	// livre de corrida -- e sem o gargalo de m=1 do caminho por matmul.
	parallelFor(p.C, func(lo, hi int) {
		for ch := lo; ch < hi; ch++ {
			var b float32
			if bias != nil {
				b = bias[ch]
			}
			depthwiseChannel(
				src[ch*p.H*p.W:(ch+1)*p.H*p.W],
				weights[ch*kSize:(ch+1)*kSize],
				b, p, oh, ow,
				dst[ch*spatial:(ch+1)*spatial],
			)
		}
	})

	return nil
}

// depthwiseChannel resolve um unico canal.
//
// A ordem dos lacos e kernel por fora, imagem por dentro. Cada posicao do
// kernel percorre a saida inteira acumulando a sua contribuicao, o que faz
// o laco mais interno andar de forma contigua na memoria -- e com stride 1
// ele vira um axpy, exatamente como o miolo do matmul.
//
// O truque que elimina o custo do padding: em vez de testar posicao por
// posicao se a janela caiu fora da imagem, calcula-se de antemao a faixa de
// colunas em que ela cai dentro. O laco interno fica sem nenhum desvio.
func depthwiseChannel(in, w []float32, bias float32, p ConvParams, oh, ow int, out []float32) {
	if bias == 0 {
		clear(out)
	} else {
		for i := range out {
			out[i] = bias
		}
	}

	for i := 0; i < p.KH; i++ {
		for j := 0; j < p.KW; j++ {
			wv := w[i*p.KW+j]
			if wv == 0 {
				continue
			}

			// Colunas de saida em que esta posicao do kernel le um pixel
			// real: 0 <= x*StrideW - PadW + j*DilW < W.
			offset := j*p.DilW - p.PadW
			xlo := max(0, ceilDiv(-offset, p.StrideW))
			xhi := min(ow, ceilDiv(p.W-offset, p.StrideW))
			if xlo >= xhi {
				continue // esta posicao do kernel esta sempre no padding
			}

			for y := 0; y < oh; y++ {
				iy := y*p.StrideH - p.PadH + i*p.DilH
				if iy < 0 || iy >= p.H {
					continue // a linha inteira caiu fora da imagem
				}

				base := iy*p.W + offset
				orow := out[y*ow : (y+1)*ow]

				if p.StrideW == 1 {
					// Caminho rapido: entrada e saida contiguas lado a lado.
					os := orow[xlo:xhi]
					is := in[base+xlo : base+xhi]
					for k, iv := range is {
						os[k] += wv * iv
					}
					continue
				}

				for x := xlo; x < xhi; x++ {
					orow[x] += wv * in[base+x*p.StrideW]
				}
			}
		}
	}
}
