package kernel

import "fmt"

// ConvParams descreve a geometria de uma convolucao 2D.
//
// Campos zerados em Stride, Dil e Groups sao lidos como 1, que e o padrao
// da esmagadora maioria das camadas.
type ConvParams struct {
	C, H, W          int // canais, altura e largura da entrada
	KH, KW           int // altura e largura do kernel
	StrideH, StrideW int // passo do deslizamento
	PadH, PadW       int // preenchimento com zeros nas bordas
	DilH, DilW       int // dilatacao (espacamento entre os pesos do kernel)
	Groups           int // convolucao agrupada; Groups == C e a depthwise
}

// norm devolve uma copia com os padroes aplicados.
func (p ConvParams) norm() ConvParams {
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
func (p ConvParams) OutH() int {
	p = p.norm()
	return (p.H+2*p.PadH-(p.DilH*(p.KH-1)+1))/p.StrideH + 1
}

// OutW devolve a largura da saida.
func (p ConvParams) OutW() int {
	p = p.norm()
	return (p.W+2*p.PadW-(p.DilW*(p.KW-1)+1))/p.StrideW + 1
}

// ColRows e o numero de linhas da matriz de janelas: um elemento do kernel
// por linha, para cada canal de entrada do grupo.
func (p ConvParams) ColRows() int {
	p = p.norm()
	return (p.C / p.Groups) * p.KH * p.KW
}

// ColCols e o numero de colunas da matriz de janelas: uma posicao da saida
// por coluna.
func (p ConvParams) ColCols() int { return p.OutH() * p.OutW() }

// ColSize e quantos float32 o buffer de Im2Col precisa ter.
func (p ConvParams) ColSize() int { return p.ColRows() * p.ColCols() }

// Validate confere se a geometria faz sentido.
func (p ConvParams) Validate() error {
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

// Im2Col rearranja a entrada de forma que a convolucao vire uma unica
// multiplicacao de matrizes.
//
// A ideia: cada janela que o kernel visitaria vira uma coluna da matriz de
// saida. Depois disso,
//
//	saida = pesos x colunas
//
// resolve a convolucao inteira de uma vez. Custa memoria -- cada pixel
// aparece repetido em ate KH*KW colunas -- e devolve em troca a operacao
// mais otimizavel que existe.
//
//	src: [C, H, W] contiguo
//	dst: [C*KH*KW, OutH*OutW], precisa ter ao menos ColSize() elementos
//
// Para convolucao agrupada, chame uma vez por grupo passando a fatia de
// canais daquele grupo e um ConvParams com C ja dividido.
func Im2Col(src []float32, p ConvParams, dst []float32) {
	p = p.norm()
	oh, ow := p.OutH(), p.OutW()

	if len(src) < p.C*p.H*p.W {
		panic(fmt.Sprintf("kernel: src tem %d elementos, precisa de %d", len(src), p.C*p.H*p.W))
	}
	if len(dst) < p.ColRows()*oh*ow {
		panic(fmt.Sprintf("kernel: dst tem %d elementos, precisa de %d", len(dst), p.ColRows()*oh*ow))
	}

	// A ordem dos lacos -- canal, depois posicao dentro do kernel, e so
	// entao a varredura espacial -- faz cada linha de dst ser preenchida do
	// inicio ao fim de uma vez, lendo src quase sequencialmente.
	for ch := 0; ch < p.C; ch++ {
		for i := 0; i < p.KH; i++ {
			for j := 0; j < p.KW; j++ {
				row := (ch*p.KH+i)*p.KW + j
				out := dst[row*oh*ow : (row+1)*oh*ow]

				for y := 0; y < oh; y++ {
					seg := out[y*ow : (y+1)*ow]
					iy := y*p.StrideH - p.PadH + i*p.DilH

					if iy < 0 || iy >= p.H {
						clear(seg) // a janela inteira caiu no padding
						continue
					}

					base := (ch*p.H + iy) * p.W
					for x := 0; x < ow; x++ {
						ix := x*p.StrideW - p.PadW + j*p.DilW
						if ix < 0 || ix >= p.W {
							seg[x] = 0
							continue
						}
						seg[x] = src[base+ix]
					}
				}
			}
		}
	}
}

// Conv2D executa uma convolucao 2D via Im2Col seguido de MatMul.
//
//	src:     [C, H, W]
//	weights: [outC, C/Groups, KH, KW]
//	bias:    [outC] ou nil
//	dst:     [outC, OutH, OutW]
//
// Esta e a funcao que prova o encadeamento: se ela bate com a convolucao
// ingenua de referencia, entao Tensor, Im2Col e MatMul estao todos certos.
//
// Aloca o buffer intermediario do im2col a cada chamada. Numa rede com
// dezenas de camadas isso vira pressao de coletor de lixo; para esse caso
// use Conv2DScratch com um buffer reaproveitado.
func Conv2D(src, weights, bias []float32, outC int, p ConvParams, dst []float32) error {
	return Conv2DScratch(src, weights, bias, outC, p, dst, nil)
}

// Conv2DScratch e a Conv2D recebendo o buffer intermediario de fora.
//
// scratch precisa ter ao menos p.ColSize() elementos; passe nil para que a
// funcao aloque um. O conteudo anterior e irrelevante -- o im2col sobrescreve
// tudo que usa.
//
// Existe para quem processa muitas camadas em sequencia e quer alocar uma
// vez so, em vez de uma vez por camada por imagem.
func Conv2DScratch(src, weights, bias []float32, outC int, p ConvParams, dst, scratch []float32) error {
	if err := p.Validate(); err != nil {
		return err
	}
	p = p.norm()

	if outC <= 0 || outC%p.Groups != 0 {
		return fmt.Errorf("kernel: %d canais de saida nao dividem em %d grupos", outC, p.Groups)
	}

	// Depthwise -- um filtro por canal -- tem kernel proprio. Por im2col ela
	// viraria C matmuls de uma linha so, o que desliga o paralelismo e paga
	// o rearranjo de memoria sem ter volume para amortiza-lo. Ver depthwise.go.
	if p.Groups == p.C && outC == p.C {
		return DepthwiseConv2D(src, weights, bias, p, dst)
	}

	oh, ow := p.OutH(), p.OutW()
	inPerGroup := p.C / p.Groups
	outPerGroup := outC / p.Groups
	colRows := p.ColRows() // ja considera o grupo
	spatial := oh * ow

	if want := outC * inPerGroup * p.KH * p.KW; len(weights) < want {
		return fmt.Errorf("kernel: weights tem %d elementos, precisa de %d", len(weights), want)
	}
	if bias != nil && len(bias) < outC {
		return fmt.Errorf("kernel: bias tem %d elementos, precisa de %d", len(bias), outC)
	}
	if len(dst) < outC*spatial {
		return fmt.Errorf("kernel: dst tem %d elementos, precisa de %d", len(dst), outC*spatial)
	}

	// Geometria de um unico grupo: mesma coisa, com menos canais.
	gp := p
	gp.C = inPerGroup
	gp.Groups = 1

	need := colRows * spatial
	var cols []float32
	if len(scratch) >= need {
		cols = scratch[:need]
	} else {
		cols = make([]float32, need)
	}

	for g := 0; g < p.Groups; g++ {
		srcGroup := src[g*inPerGroup*p.H*p.W : (g+1)*inPerGroup*p.H*p.W]
		Im2Col(srcGroup, gp, cols)

		// pesos do grupo: [outPerGroup, colRows]
		wGroup := weights[g*outPerGroup*colRows : (g+1)*outPerGroup*colRows]
		dstGroup := dst[g*outPerGroup*spatial : (g+1)*outPerGroup*spatial]

		MatMul(wGroup, cols, dstGroup, outPerGroup, colRows, spatial)
	}

	if bias != nil {
		for oc := 0; oc < outC; oc++ {
			bv := bias[oc]
			if bv == 0 {
				continue
			}
			row := dst[oc*spatial : (oc+1)*spatial]
			for i := range row {
				row[i] += bv
			}
		}
	}

	return nil
}
