package nn

import (
	"fmt"
	"math"

	"github.com/iafrotamacedo-cloud/era/faces/tensor"
)

// BatchNorm normaliza cada canal com estatisticas fixas, aprendidas no
// treino.
//
// No treino, a camada calcula media e variancia do lote e vai acumulando
// medias moveis. Na inferencia -- que e tudo o que a ERA faz -- essas
// estatisticas ja estao congeladas, e a camada inteira colapsa em uma
// multiplicacao e uma soma por canal:
//
//	y = gama * (x - media) / sqrt(variancia + eps) + beta
//	  = x * escala + deslocamento
//
//	escala       = gama / sqrt(variancia + eps)
//	deslocamento = beta - media * escala
//
// Escala e deslocamento sao calculados uma vez, na construcao. Melhor ainda:
// os dois cabem dentro dos pesos da convolucao anterior, e ai a camada some
// do tempo de execucao. Ver Conv2D.FuseBatchNorm e Sequential.Fuse.
type BatchNorm struct {
	name     string
	Channels int

	// Scale e Shift sao a forma reduzida, prontos para uso e para fusao.
	Scale []float32
	Shift []float32
}

// NewBatchNorm monta a camada a partir dos quatro vetores que o ONNX guarda,
// reduzindo-os a escala e deslocamento.
//
// gamma, beta, mean e variance precisam ter um elemento por canal.
func NewBatchNorm(name string, gamma, beta, mean, variance []float32, eps float32) (*BatchNorm, error) {
	c := len(gamma)
	if c == 0 {
		return nil, fmt.Errorf("nn: %s: gamma vazio", name)
	}
	if len(beta) != c || len(mean) != c || len(variance) != c {
		return nil, fmt.Errorf("nn: %s: gamma/beta/mean/variance com tamanhos diferentes (%d/%d/%d/%d)",
			name, len(gamma), len(beta), len(mean), len(variance))
	}
	if eps < 0 {
		return nil, fmt.Errorf("nn: %s: eps negativo (%v)", name, eps)
	}

	bn := &BatchNorm{
		name:     name,
		Channels: c,
		Scale:    make([]float32, c),
		Shift:    make([]float32, c),
	}

	for i := 0; i < c; i++ {
		denom := variance[i] + eps
		if denom <= 0 {
			// Variancia nao negativa mais eps deveria ser positiva. Se nao
			// for, os pesos vieram corrompidos -- melhor recusar do que
			// produzir NaN silenciosamente la na frente.
			return nil, fmt.Errorf("nn: %s: canal %d tem variancia+eps = %v, precisa ser positivo",
				name, i, denom)
		}
		s := float32(1 / math.Sqrt(float64(denom)))
		bn.Scale[i] = gamma[i] * s
		bn.Shift[i] = beta[i] - mean[i]*bn.Scale[i]
	}

	return bn, nil
}

// NewBatchNormFromScaleShift monta a camada direto na forma reduzida. Util
// para testes e para pesos ja fundidos.
func NewBatchNormFromScaleShift(name string, scale, shift []float32) (*BatchNorm, error) {
	if len(scale) == 0 || len(scale) != len(shift) {
		return nil, fmt.Errorf("nn: %s: scale e shift precisam ter o mesmo tamanho, nao vazio (%d/%d)",
			name, len(scale), len(shift))
	}
	return &BatchNorm{
		name:     name,
		Channels: len(scale),
		Scale:    append([]float32(nil), scale...),
		Shift:    append([]float32(nil), shift...),
	}, nil
}

// Name identifica a camada.
func (b *BatchNorm) Name() string {
	if b.name == "" {
		return "BatchNorm"
	}
	return b.name
}

// OutputShape: BatchNorm preserva a forma.
func (b *BatchNorm) OutputShape(in []int) ([]int, error) {
	if len(in) != 4 && len(in) != 2 {
		return nil, fmt.Errorf("nn: %s espera [N,C,H,W] ou [N,C], recebeu %v", b.Name(), in)
	}
	if in[1] != b.Channels {
		return nil, fmt.Errorf("nn: %s espera %d canais, recebeu %d", b.Name(), b.Channels, in[1])
	}
	return append([]int(nil), in...), nil
}

// Forward aplica escala e deslocamento por canal.
func (b *BatchNorm) Forward(ws *Workspace, x *tensor.Tensor) (*tensor.Tensor, error) {
	if _, err := b.OutputShape(x.Shape); err != nil {
		return nil, err
	}

	src := x.Flat()
	n, c := x.Shape[0], x.Shape[1]

	espacial := 1
	for _, d := range x.Shape[2:] {
		espacial *= d
	}

	out := ws.Tensor(x.Shape...)

	for i := 0; i < n; i++ {
		for ch := 0; ch < c; ch++ {
			s, sh := b.Scale[ch], b.Shift[ch]
			base := (i*c + ch) * espacial

			in := src[base : base+espacial]
			o := out.Data[base : base+espacial]
			for k, v := range in {
				o[k] = v*s + sh
			}
		}
	}

	return out, nil
}
