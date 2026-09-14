package nn

import (
	"fmt"

	"github.com/iafrotamacedo-cloud/era/faces/kernel"
	"github.com/iafrotamacedo-cloud/era/faces/tensor"
)

// ConvTranspose2D e a camada de convolucao transposta 2D (deconvolucao,
// upsample aprendido).
//
// Guarda os pesos e a geometria; as contas ficam no pacote kernel, igual a
// Conv2D. A diferenca que importa: o layout dos pesos segue o ONNX de
// ConvTranspose, que guarda o peso pelo canal de ENTRADA primeiro --
// oposto de Conv2D, que guarda pelo de saida:
//
//	Weights  [InC, OutC/Groups, KH, KW]
//	Bias     [OutC] ou nil
type ConvTranspose2D struct {
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

// ConvTranspose2DConfig descreve a geometria de uma convolucao transposta.
//
// Campos zerados em Stride, Dil e Groups valem 1.
type ConvTranspose2DConfig struct {
	InC, OutC        int
	KH, KW           int
	StrideH, StrideW int
	PadH, PadW       int
	DilH, DilW       int
	Groups           int
}

// NewConvTranspose2D monta uma camada de convolucao transposta.
//
// weights precisa ter InC*(OutC/Groups)*KH*KW elementos; bias, OutC ou nil.
func NewConvTranspose2D(name string, cfg ConvTranspose2DConfig, weights, bias []float32) (*ConvTranspose2D, error) {
	c := &ConvTranspose2D{
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

	if want := c.InC * (c.OutC / c.Groups) * c.KH * c.KW; len(weights) != want {
		return nil, fmt.Errorf("nn: %s: weights tem %d elementos, precisa de %d", name, len(weights), want)
	}
	if bias != nil && len(bias) != c.OutC {
		return nil, fmt.Errorf("nn: %s: bias tem %d elementos, precisa de %d", name, len(bias), c.OutC)
	}

	return c, nil
}

// Name identifica a camada.
func (c *ConvTranspose2D) Name() string {
	if c.name == "" {
		return "ConvTranspose2D"
	}
	return c.name
}

// params monta a geometria que o kernel espera, para uma entrada HxW.
func (c *ConvTranspose2D) params(h, w int) kernel.ConvTransposeParams {
	return kernel.ConvTransposeParams{
		C: c.InC, H: h, W: w,
		KH: c.KH, KW: c.KW,
		StrideH: c.StrideH, StrideW: c.StrideW,
		PadH: c.PadH, PadW: c.PadW,
		DilH: c.DilH, DilW: c.DilW,
		Groups: c.Groups,
	}
}

// OutputShape calcula a forma da saida sem executar a convolucao.
func (c *ConvTranspose2D) OutputShape(in []int) ([]int, error) {
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

// Forward aplica a convolucao transposta a cada imagem do lote.
func (c *ConvTranspose2D) Forward(ws *Workspace, x *tensor.Tensor) (*tensor.Tensor, error) {
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

	for i := 0; i < n; i++ {
		err := kernel.ConvTranspose2D(
			src[i*inPorImagem:(i+1)*inPorImagem],
			c.Weights, c.Bias, c.OutC, p,
			out.Data[i*outPorImagem:(i+1)*outPorImagem],
		)
		if err != nil {
			return nil, fmt.Errorf("nn: %s: %w", c.Name(), err)
		}
	}

	return out, nil
}
