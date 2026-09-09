package nn

import (
	"fmt"

	"github.com/iafrotamacedo-cloud/era/faces/kernel"
	"github.com/iafrotamacedo-cloud/era/faces/tensor"
)

// Linear e a camada totalmente conectada: y = W*x + b.
//
// Na ERA ela e a ultima etapa -- a que produz o vetor que representa o rosto.
//
// O tamanho do vetor e do modelo, nao da biblioteca: o SFace produz 128
// dimensoes, o ArcFace produz 512. A camada le o que os pesos disserem.
//
// Layout dos pesos, igual ao do ONNX (Gemm com transB=1):
//
//	Weights  [OutFeatures, InFeatures]
//	Bias     [OutFeatures] ou nil
type Linear struct {
	name                    string
	InFeatures, OutFeatures int

	Weights []float32
	Bias    []float32
}

// NewLinear monta a camada.
func NewLinear(name string, inF, outF int, weights, bias []float32) (*Linear, error) {
	if inF <= 0 || outF <= 0 {
		return nil, fmt.Errorf("nn: %s: dimensoes invalidas in=%d out=%d", name, inF, outF)
	}
	if len(weights) != inF*outF {
		return nil, fmt.Errorf("nn: %s: weights tem %d elementos, precisa de %d (%dx%d)",
			name, len(weights), inF*outF, outF, inF)
	}
	if bias != nil && len(bias) != outF {
		return nil, fmt.Errorf("nn: %s: bias tem %d elementos, precisa de %d", name, len(bias), outF)
	}
	return &Linear{name: name, InFeatures: inF, OutFeatures: outF, Weights: weights, Bias: bias}, nil
}

// Name identifica a camada.
func (l *Linear) Name() string {
	if l.name == "" {
		return "Linear"
	}
	return l.name
}

// OutputShape troca a dimensao de atributos.
func (l *Linear) OutputShape(in []int) ([]int, error) {
	if len(in) != 2 {
		return nil, fmt.Errorf("nn: %s espera [N,F], recebeu %v (falta um Flatten?)", l.Name(), in)
	}
	if in[1] != l.InFeatures {
		return nil, fmt.Errorf("nn: %s espera %d atributos, recebeu %d", l.Name(), l.InFeatures, in[1])
	}
	return []int{in[0], l.OutFeatures}, nil
}

// Forward calcula y = W*x + b para cada linha do lote.
//
// A conta e feita como W x transposta(x), e nao como x x transposta(W). Os
// dois dao o mesmo resultado, mas o paralelismo do matmul divide pelas
// LINHAS da primeira matriz: no primeiro arranjo isso da OutFeatures linhas
// -- centenas -- e no segundo daria N, que normalmente e 1. Seria o mesmo
// erro que travava a depthwise em 0,86 GFLOPS.
//
// Com N == 1, as duas transposicoes sao gratuitas: [1,F] e [F,1] tem
// exatamente o mesmo layout na memoria.
func (l *Linear) Forward(ws *Workspace, x *tensor.Tensor) (*tensor.Tensor, error) {
	outShape, err := l.OutputShape(x.Shape)
	if err != nil {
		return nil, err
	}

	n := x.Shape[0]
	src := x.Flat()
	out := ws.Tensor(outShape...)

	if n == 1 {
		kernel.MatMul(l.Weights, src, out.Data, l.OutFeatures, l.InFeatures, 1)
	} else {
		xt := ws.Alloc(l.InFeatures * n)
		transpose(src, xt, n, l.InFeatures)

		ot := ws.Alloc(l.OutFeatures * n)
		kernel.MatMul(l.Weights, xt, ot, l.OutFeatures, l.InFeatures, n)

		transpose(ot, out.Data, l.OutFeatures, n)
	}

	if l.Bias != nil {
		for i := 0; i < n; i++ {
			row := out.Data[i*l.OutFeatures : (i+1)*l.OutFeatures]
			for j := range row {
				row[j] += l.Bias[j]
			}
		}
	}

	return out, nil
}

// transpose copia uma matriz rows x cols para cols x rows.
//
// Percorre a origem em ordem sequencial e escreve espacado no destino. O
// contrario -- ler espacado e escrever sequencial -- custaria o mesmo em
// numero de acessos, mas leitura desalinhada e mais barata que escrita
// desalinhada na maioria dos processadores.
func transpose(src, dst []float32, rows, cols int) {
	for i := 0; i < rows; i++ {
		row := src[i*cols : (i+1)*cols]
		for j, v := range row {
			dst[j*rows+i] = v
		}
	}
}
