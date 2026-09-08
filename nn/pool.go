package nn

import (
	"fmt"
	"math"

	"github.com/iafrotamacedo-cloud/era/tensor"
)

// GlobalAvgPool reduz cada canal a um unico numero: a media de todas as
// posicoes espaciais.
//
//	[N, C, H, W]  ->  [N, C, 1, 1]
//
// E a transicao classica entre a parte convolucional e a camada final.
// MobileFaceNet usa outra coisa no lugar -- uma convolucao depthwise global,
// que aprende o peso de cada posicao em vez de tratar todas por igual -- mas
// muitas arquiteturas usam esta, e a ERA precisa carrega-las.
type GlobalAvgPool struct{}

// Name identifica a camada.
func (g *GlobalAvgPool) Name() string { return "GlobalAvgPool" }

// OutputShape colapsa as dimensoes espaciais para 1x1.
func (g *GlobalAvgPool) OutputShape(in []int) ([]int, error) {
	if len(in) != 4 {
		return nil, fmt.Errorf("nn: GlobalAvgPool espera [N,C,H,W], recebeu %v", in)
	}
	return []int{in[0], in[1], 1, 1}, nil
}

// Forward calcula a media por canal.
func (g *GlobalAvgPool) Forward(ws *Workspace, x *tensor.Tensor) (*tensor.Tensor, error) {
	outShape, err := g.OutputShape(x.Shape)
	if err != nil {
		return nil, err
	}

	n, c, h, w := x.Shape[0], x.Shape[1], x.Shape[2], x.Shape[3]
	espacial := h * w
	if espacial == 0 {
		return nil, fmt.Errorf("nn: GlobalAvgPool recebeu entrada vazia %v", x.Shape)
	}

	src := x.Flat()
	out := ws.Tensor(outShape...)
	inv := 1 / float32(espacial)

	for i := 0; i < n*c; i++ {
		var soma float32
		for _, v := range src[i*espacial : (i+1)*espacial] {
			soma += v
		}
		out.Data[i] = soma * inv
	}

	return out, nil
}

// MaxPool2D reduz a resolucao ficando com o maior valor de cada janela.
//
// Posicoes que caem no padding valem menos infinito, e por isso nunca
// vencem -- o que e diferente de valerem zero, que venceria de qualquer
// ativacao negativa e falsearia a borda da imagem.
type MaxPool2D struct {
	name             string
	KH, KW           int
	StrideH, StrideW int
	PadH, PadW       int
}

// NewMaxPool2D monta a camada. Stride zerado assume o proprio tamanho da
// janela, que e o comportamento padrao do ONNX.
func NewMaxPool2D(name string, kh, kw, strideH, strideW, padH, padW int) (*MaxPool2D, error) {
	if kh <= 0 || kw <= 0 {
		return nil, fmt.Errorf("nn: %s: janela invalida %dx%d", name, kh, kw)
	}
	if strideH == 0 {
		strideH = kh
	}
	if strideW == 0 {
		strideW = kw
	}
	if strideH < 0 || strideW < 0 || padH < 0 || padW < 0 {
		return nil, fmt.Errorf("nn: %s: stride ou padding negativo", name)
	}
	return &MaxPool2D{name: name, KH: kh, KW: kw, StrideH: strideH, StrideW: strideW, PadH: padH, PadW: padW}, nil
}

// Name identifica a camada.
func (m *MaxPool2D) Name() string {
	if m.name == "" {
		return "MaxPool2D"
	}
	return m.name
}

// OutputShape calcula a resolucao de saida.
func (m *MaxPool2D) OutputShape(in []int) ([]int, error) {
	if len(in) != 4 {
		return nil, fmt.Errorf("nn: %s espera [N,C,H,W], recebeu %v", m.Name(), in)
	}
	oh := (in[2]+2*m.PadH-m.KH)/m.StrideH + 1
	ow := (in[3]+2*m.PadW-m.KW)/m.StrideW + 1
	if oh <= 0 || ow <= 0 {
		return nil, fmt.Errorf("nn: %s: geometria produz saida vazia %dx%d", m.Name(), oh, ow)
	}
	return []int{in[0], in[1], oh, ow}, nil
}

// Forward aplica o pooling.
func (m *MaxPool2D) Forward(ws *Workspace, x *tensor.Tensor) (*tensor.Tensor, error) {
	outShape, err := m.OutputShape(x.Shape)
	if err != nil {
		return nil, err
	}

	n, c, h, w := x.Shape[0], x.Shape[1], x.Shape[2], x.Shape[3]
	oh, ow := outShape[2], outShape[3]

	src := x.Flat()
	out := ws.Tensor(outShape...)
	menosInf := float32(math.Inf(-1))

	for plano := 0; plano < n*c; plano++ {
		in := src[plano*h*w : (plano+1)*h*w]
		o := out.Data[plano*oh*ow : (plano+1)*oh*ow]

		for y := 0; y < oh; y++ {
			for x := 0; x < ow; x++ {
				melhor := menosInf

				for i := 0; i < m.KH; i++ {
					iy := y*m.StrideH - m.PadH + i
					if iy < 0 || iy >= h {
						continue
					}
					for j := 0; j < m.KW; j++ {
						ix := x*m.StrideW - m.PadW + j
						if ix < 0 || ix >= w {
							continue
						}
						if v := in[iy*w+ix]; v > melhor {
							melhor = v
						}
					}
				}

				o[y*ow+x] = melhor
			}
		}
	}

	return out, nil
}
