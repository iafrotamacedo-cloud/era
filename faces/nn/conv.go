package nn

import (
	"fmt"

	"github.com/iafrotamacedo-cloud/era/faces/kernel"
	"github.com/iafrotamacedo-cloud/era/faces/tensor"
)

// Conv2D e a camada de convolucao 2D.
//
// Guarda os pesos e a geometria; as contas ficam no pacote kernel, que
// escolhe sozinho entre im2col+matmul e o kernel dedicado de depthwise.
//
// Layout dos pesos, igual ao do ONNX:
//
//	Weights  [OutC, InC/Groups, KH, KW]
//	Bias     [OutC] ou nil
type Conv2D struct {
	name string

	InC, OutC        int
	KH, KW           int
	StrideH, StrideW int
	PadH, PadW       int
	DilH, DilW       int
	Groups           int

	Weights []float32
	Bias    []float32
}

// Conv2DConfig descreve a geometria de uma convolucao.
//
// Campos zerados em Stride, Dil e Groups valem 1.
type Conv2DConfig struct {
	InC, OutC        int
	KH, KW           int
	StrideH, StrideW int
	PadH, PadW       int
	DilH, DilW       int
	Groups           int
}

// NewConv2D monta uma camada de convolucao.
//
// weights precisa ter OutC*(InC/Groups)*KH*KW elementos; bias, OutC ou nil.
// Os slices passam a ser propriedade da camada -- FuseBatchNorm escreve
// neles.
func NewConv2D(name string, cfg Conv2DConfig, weights, bias []float32) (*Conv2D, error) {
	c := &Conv2D{
		name:    name,
		InC:     cfg.InC,
		OutC:    cfg.OutC,
		KH:      cfg.KH,
		KW:      cfg.KW,
		StrideH: cfg.StrideH,
		StrideW: cfg.StrideW,
		PadH:    cfg.PadH,
		PadW:    cfg.PadW,
		DilH:    cfg.DilH,
		DilW:    cfg.DilW,
		Groups:  cfg.Groups,
		Weights: weights,
		Bias:    bias,
	}

	if c.StrideH == 0 {
		c.StrideH = 1
	}
	if c.StrideW == 0 {
		c.StrideW = 1
	}
	if c.DilH == 0 {
		c.DilH = 1
	}
	if c.DilW == 0 {
		c.DilW = 1
	}
	if c.Groups == 0 {
		c.Groups = 1
	}

	switch {
	case c.InC <= 0 || c.OutC <= 0:
		return nil, fmt.Errorf("nn: %s: canais invalidos InC=%d OutC=%d", name, c.InC, c.OutC)
	case c.KH <= 0 || c.KW <= 0:
		return nil, fmt.Errorf("nn: %s: kernel invalido %dx%d", name, c.KH, c.KW)
	case c.InC%c.Groups != 0 || c.OutC%c.Groups != 0:
		return nil, fmt.Errorf("nn: %s: %d/%d canais nao dividem em %d grupos", name, c.InC, c.OutC, c.Groups)
	}

	if want := c.OutC * (c.InC / c.Groups) * c.KH * c.KW; len(weights) != want {
		return nil, fmt.Errorf("nn: %s: weights tem %d elementos, precisa de %d", name, len(weights), want)
	}
	if bias != nil && len(bias) != c.OutC {
		return nil, fmt.Errorf("nn: %s: bias tem %d elementos, precisa de %d", name, len(bias), c.OutC)
	}

	return c, nil
}

// Name identifica a camada.
func (c *Conv2D) Name() string {
	if c.name == "" {
		return "Conv2D"
	}
	return c.name
}

// IsDepthwise informa se esta e uma convolucao depthwise -- um filtro por
// canal, sem mistura. E o caso que tem kernel proprio.
func (c *Conv2D) IsDepthwise() bool { return c.Groups == c.InC && c.OutC == c.InC }

// params monta a geometria que o kernel espera, para uma entrada HxW.
func (c *Conv2D) params(h, w int) kernel.ConvParams {
	return kernel.ConvParams{
		C: c.InC, H: h, W: w,
		KH: c.KH, KW: c.KW,
		StrideH: c.StrideH, StrideW: c.StrideW,
		PadH: c.PadH, PadW: c.PadW,
		DilH: c.DilH, DilW: c.DilW,
		Groups: c.Groups,
	}
}

// OutputShape calcula a forma da saida sem executar a convolucao.
func (c *Conv2D) OutputShape(in []int) ([]int, error) {
	if len(in) != 4 {
		return nil, fmt.Errorf("nn: %s espera [N,C,H,W], recebeu %v", c.Name(), in)
	}
	if in[1] != c.InC {
		return nil, fmt.Errorf("nn: %s espera %d canais, recebeu %d", c.Name(), c.InC, in[1])
	}

	p := c.params(in[2], in[3])
	if err := p.Validate(); err != nil {
		return nil, fmt.Errorf("nn: %s: %w", c.Name(), err)
	}
	return []int{in[0], c.OutC, p.OutH(), p.OutW()}, nil
}

// Forward aplica a convolucao a cada imagem do lote.
func (c *Conv2D) Forward(ws *Workspace, x *tensor.Tensor) (*tensor.Tensor, error) {
	if err := checkRank(c.Name(), x, 4); err != nil {
		return nil, err
	}

	outShape, err := c.OutputShape(x.Shape)
	if err != nil {
		return nil, err
	}

	src := x.Flat()
	n := x.Shape[0]
	p := c.params(x.Shape[2], x.Shape[3])

	inPorImagem := c.InC * x.Shape[2] * x.Shape[3]
	outPorImagem := c.OutC * outShape[2] * outShape[3]

	out := ws.Tensor(outShape...)

	// Um buffer de im2col para o lote inteiro: as imagens sao processadas
	// em sequencia e cada uma sobrescreve o que a anterior deixou.
	var scratch []float32
	if !c.IsDepthwise() {
		scratch = ws.Alloc(p.ColSize())
	}

	for i := 0; i < n; i++ {
		err := kernel.Conv2DScratch(
			src[i*inPorImagem:(i+1)*inPorImagem],
			c.Weights, c.Bias, c.OutC, p,
			out.Data[i*outPorImagem:(i+1)*outPorImagem],
			scratch,
		)
		if err != nil {
			return nil, fmt.Errorf("nn: %s: %w", c.Name(), err)
		}
	}

	return out, nil
}

// FuseBatchNorm absorve um BatchNorm nos pesos desta convolucao.
//
//	conv:  y = W*x + b
//	bn:    z = y*escala + deslocamento
//	       z = (W*escala)*x + (b*escala + deslocamento)
//
// Como escala e deslocamento sao por canal de saida, e cada canal de saida
// tem o seu proprio bloco de pesos, a absorcao e exata -- nao ha
// aproximacao envolvida. Depois disso o BatchNorm pode ser removido da rede.
//
// Modifica Weights e Bias no lugar. Se a convolucao nao tinha bias, ganha um.
func (c *Conv2D) FuseBatchNorm(bn *BatchNorm) error {
	if bn.Channels != c.OutC {
		return fmt.Errorf("nn: %s tem %d canais de saida, %s espera %d",
			c.Name(), c.OutC, bn.Name(), bn.Channels)
	}

	if c.Bias == nil {
		c.Bias = make([]float32, c.OutC)
	}

	pesosPorCanal := len(c.Weights) / c.OutC
	for oc := 0; oc < c.OutC; oc++ {
		s := bn.Scale[oc]

		w := c.Weights[oc*pesosPorCanal : (oc+1)*pesosPorCanal]
		for i := range w {
			w[i] *= s
		}

		c.Bias[oc] = c.Bias[oc]*s + bn.Shift[oc]
	}

	return nil
}
