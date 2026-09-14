package graph

import (
	"fmt"

	"github.com/iafrotamacedo-cloud/era/faces/nn"
	"github.com/iafrotamacedo-cloud/era/faces/onnx"
	"github.com/iafrotamacedo-cloud/era/faces/tensor"
)

// montaShape devolve a forma da entrada como vetor 1D de int64.
func montaShape(b *builder, n *onnx.Node) (*operation, error) {
	return novaOp(n, func(ws *nn.Workspace, ins []*tensor.Tensor) ([]*tensor.Tensor, error) {
		x := ins[0]
		out := ws.Tensor(x.Rank())
		for i, d := range x.Shape {
			out.Data[i] = float32(d)
		}
		return []*tensor.Tensor{out}, nil
	}), nil
}

// montaGather seleciona fatias ao longo de um eixo.
func montaGather(b *builder, n *onnx.Node) (*operation, error) {
	if len(n.Inputs) < 2 {
		return nil, fmt.Errorf("Gather espera pelo menos 2 entradas, recebeu %d", len(n.Inputs))
	}
	eixo := int(n.AttrInt("axis", 0))

	indicesConst := b.constOpcional(n, 1)

	return novaOp(n, func(ws *nn.Workspace, ins []*tensor.Tensor) ([]*tensor.Tensor, error) {
		dados := ins[0]
		var ind *tensor.Tensor
		if indicesConst != nil {
			ind = indicesConst
		} else if len(ins) >= 2 {
			ind = ins[1]
		} else {
			return nil, fmt.Errorf("Gather sem tensor de indices")
		}

		out, err := gather(ws, dados, ind, eixo)
		if err != nil {
			return nil, err
		}
		return []*tensor.Tensor{out}, nil
	}), nil
}

func gather(ws *nn.Workspace, dados *tensor.Tensor, ind *tensor.Tensor, eixo int) (*tensor.Tensor, error) {
	if eixo < 0 {
		eixo += dados.Rank()
	}
	if eixo < 0 || eixo >= dados.Rank() {
		return nil, fmt.Errorf("Gather: eixo %d fora da forma %v", eixo, dados.Shape)
	}

	prefixo := 1
	for _, d := range dados.Shape[:eixo] {
		prefixo *= d
	}
	sufixo := 1
	for _, d := range dados.Shape[eixo+1:] {
		sufixo *= d
	}
	tamEixo := dados.Shape[eixo]

	indices := ind.Flat()
	nInd := len(indices)
	if nInd == 0 {
		return nil, fmt.Errorf("Gather: indices vazio")
	}

	forma := make([]int, 0, dados.Rank()-1+ind.Rank())
	forma = append(forma, dados.Shape[:eixo]...)
	forma = append(forma, ind.Shape...)
	forma = append(forma, dados.Shape[eixo+1:]...)

	out := ws.Tensor(forma...)
	src := garanteContiguo(ws, dados).Flat()

	for o := 0; o < prefixo; o++ {
		for i, iv := range indices {
			idx := int(iv)
			if idx < 0 {
				idx += tamEixo
			}
			if idx < 0 || idx >= tamEixo {
				return nil, fmt.Errorf("Gather: indice %d fora de [0,%d)", idx, tamEixo)
			}
			dstOff := (o*nInd + i) * sufixo
			srcOff := (o*tamEixo + idx) * sufixo
			copy(out.Data[dstOff:dstOff+sufixo], src[srcOff:srcOff+sufixo])
		}
	}

	return out, nil
}
