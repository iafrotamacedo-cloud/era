package kernel

import "fmt"

// ConvTransposeParams descreve a geometria de uma convolucao transposta 2D
// (deconvolucao). Os mesmos campos de ConvParams, com um significado
// diferente: aqui C, H, W sao a entrada da rede (o lado pequeno), e a
// saida e MAIOR, nao menor.
//
// Campos zerados em Stride, Dil e Groups sao lidos como 1.
type ConvTransposeParams struct {
	C, H, W          int // canais, altura e largura da entrada
	KH, KW           int // altura e largura do kernel
	StrideH, StrideW int // passo
	PadH, PadW       int // preenchimento (subtraido da saida, ao contrario de Conv)
	DilH, DilW       int // dilatacao
	Groups           int // convolucao agrupada
}

func (p ConvTransposeParams) norm() ConvTransposeParams {
	if p.StrideH == 0 {
		p.StrideH = 1
	}
	if p.StrideW == 0 {
		p.StrideW = 1
	}
	if p.DilH == 0 {
		p.DilH = 1
	}
	if p.DilW == 0 {
		p.DilW = 1
	}
	if p.Groups == 0 {
		p.Groups = 1
	}
	return p
}

// OutH devolve a altura da saida.
//
// Formula inversa de ConvParams.OutH: onde a convolucao direta subtrai o
// kernel do tamanho da entrada e divide pelo passo, a transposta multiplica
// pelo passo e soma o kernel. E a mesma relacao que faz uma ser a adjunta
// da outra.
func (p ConvTransposeParams) OutH() int {
	p = p.norm()
	return (p.H-1)*p.StrideH - 2*p.PadH + p.DilH*(p.KH-1) + 1
}

// OutW devolve a largura da saida.
func (p ConvTransposeParams) OutW() int {
	p = p.norm()
	return (p.W-1)*p.StrideW - 2*p.PadW + p.DilW*(p.KW-1) + 1
}

// Validate confere se a geometria faz sentido.
func (p ConvTransposeParams) Validate() error {
	q := p.norm()
	switch {
	case q.C <= 0 || q.H <= 0 || q.W <= 0:
		return fmt.Errorf("kernel: entrada invalida C=%d H=%d W=%d", q.C, q.H, q.W)
	case q.KH <= 0 || q.KW <= 0:
		return fmt.Errorf("kernel: kernel invalido %dx%d", q.KH, q.KW)
	case q.StrideH <= 0 || q.StrideW <= 0:
		return fmt.Errorf("kernel: stride invalido %dx%d", q.StrideH, q.StrideW)
	case q.PadH < 0 || q.PadW < 0:
		return fmt.Errorf("kernel: padding negativo %dx%d", q.PadH, q.PadW)
	case q.DilH <= 0 || q.DilW <= 0:
		return fmt.Errorf("kernel: dilatacao invalida %dx%d", q.DilH, q.DilW)
	case q.Groups <= 0 || q.C%q.Groups != 0:
		return fmt.Errorf("kernel: %d canais nao dividem em %d grupos", q.C, q.Groups)
	case q.OutH() <= 0 || q.OutW() <= 0:
		return fmt.Errorf("kernel: geometria produz saida vazia %dx%d", q.OutH(), q.OutW())
	}
	return nil
}

// ConvTranspose2D executa a convolucao transposta (deconvolucao) reunindo,
// para cada posicao de saida, as contribuicoes de entrada cuja janela do
// kernel a alcancaria numa convolucao direta -- o mesmo "gather" de
// Conv2DRef, com os papeis de entrada e saida invertidos pela relacao
// inversa da convolucao. E um algoritmo diferente do que
// ConvTranspose2DRef usa (que distribui, em vez de reunir); a conferencia
// entre os dois nos testes e uma verificacao de verdade, nao a mesma conta
// duas vezes.
//
//	src:     [C, H, W]
//	weights: [C, outC/Groups, KH, KW]  -- ATENCAO: eixo diferente de
//	         Conv2D. No ONNX, ConvTranspose guarda o peso pelo canal de
//	         ENTRADA primeiro, nao pelo de saida.
//	bias:    [outC] ou nil
//	dst:     [outC, OutH, OutW]
//
// Paraleliza por canal de saida: canais de saida diferentes nunca escrevem
// na mesma posicao de dst, entao a divisao e livre de corrida sem lock
// nenhum -- a mesma razao de Conv2D e DepthwiseConv2D.
func ConvTranspose2D(src, weights, bias []float32, outC int, p ConvTransposeParams, dst []float32) error {
	if err := p.Validate(); err != nil {
		return err
	}
	p = p.norm()

	oh, ow := p.OutH(), p.OutW()
	inPerGroup := p.C / p.Groups
	outPerGroup := outC / p.Groups
	kSize := p.KH * p.KW

	if len(src) < p.C*p.H*p.W {
		return fmt.Errorf("kernel: src tem %d elementos, precisa de %d", len(src), p.C*p.H*p.W)
	}
	if len(weights) < p.C*outPerGroup*kSize {
		return fmt.Errorf("kernel: weights tem %d elementos, precisa de %d", len(weights), p.C*outPerGroup*kSize)
	}
	if bias != nil && len(bias) < outC {
		return fmt.Errorf("kernel: bias tem %d elementos, precisa de %d", len(bias), outC)
	}
	if len(dst) < outC*oh*ow {
		return fmt.Errorf("kernel: dst tem %d elementos, precisa de %d", len(dst), outC*oh*ow)
	}

	parallelFor(outC, func(lo, hi int) {
		for oc := lo; oc < hi; oc++ {
			g := oc / outPerGroup
			ocLocal := oc - g*outPerGroup

			for oy := 0; oy < oh; oy++ {
				for ox := 0; ox < ow; ox++ {
					var sum float32

					for icLocal := 0; icLocal < inPerGroup; icLocal++ {
						ic := g*inPerGroup + icLocal

						for kh := 0; kh < p.KH; kh++ {
							t := oy + p.PadH - kh*p.DilH
							if t%p.StrideH != 0 {
								continue
							}
							iy := t / p.StrideH
							if iy < 0 || iy >= p.H {
								continue
							}

							for kw := 0; kw < p.KW; kw++ {
								s := ox + p.PadW - kw*p.DilW
								if s%p.StrideW != 0 {
									continue
								}
								ix := s / p.StrideW
								if ix < 0 || ix >= p.W {
									continue
								}

								w := weights[((ic*outPerGroup+ocLocal)*p.KH+kh)*p.KW+kw]
								sum += w * src[(ic*p.H+iy)*p.W+ix]
							}
						}
					}

					if bias != nil {
						sum += bias[oc]
					}
					dst[(oc*oh+oy)*ow+ox] = sum
				}
			}
		}
	})

	return nil
}
