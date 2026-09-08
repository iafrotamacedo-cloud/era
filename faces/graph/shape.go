package graph

import (
	"fmt"

	"github.com/iafrotamacedo-cloud/era/faces/nn"
	"github.com/iafrotamacedo-cloud/era/faces/onnx"
	"github.com/iafrotamacedo-cloud/era/faces/tensor"
)

// Operacoes que mexem na forma, nao nos numeros.
//
// Quase todas sao views: os dados ficam onde estao e so os metadados mudam.
// A excecao e Transpose e Concat, que precisam mover memoria de verdade.

func montaIdentity(b *builder, n *onnx.Node) (*operation, error) {
	// Dropout na inferencia tambem cai aqui: ele nao faz nada quando a rede
	// nao esta treinando. Mas se o modelo pedir a mascara como segunda
	// saida, e um modelo de treino, e nao de inferencia.
	if n.OpType == "Dropout" {
		if err := exigeSaidaUnica(n); err != nil {
			return nil, err
		}
	}

	op := novaOp(n, func(ws *nn.Workspace, ins []*tensor.Tensor) ([]*tensor.Tensor, error) {
		return []*tensor.Tensor{ins[0]}, nil
	})
	op.inputs = n.Inputs[:1]
	op.outputs = n.Outputs[:1]
	return op, nil
}

// montaConstant produz um tensor fixo, guardado no proprio no.
func montaConstant(b *builder, n *onnx.Node) (*operation, error) {
	a := n.Attr("value")
	if a == nil || a.T == nil {
		return nil, fmt.Errorf("Constant sem o atributo value (a ERA so trata o valor como tensor)")
	}

	vals, err := a.T.Floats()
	if err != nil {
		return nil, err
	}
	t, err := tensor.FromSlice(vals, a.T.Shape()...)
	if err != nil {
		return nil, err
	}

	op := novaOp(n, func(ws *nn.Workspace, ins []*tensor.Tensor) ([]*tensor.Tensor, error) {
		return []*tensor.Tensor{t}, nil
	})
	op.inputs = nil
	return op, nil
}

// achatarEm colapsa a forma em duas dimensoes, no eixo dado:
//
//	[N,C,H,W] no eixo 1  ->  [N, C*H*W]
//	[N,C,H,W] no eixo 2  ->  [N*C, H*W]
func achatarEm(ws *nn.Workspace, x *tensor.Tensor, eixo int) (*tensor.Tensor, error) {
	if eixo < 0 {
		eixo += x.Rank()
	}
	if eixo < 0 || eixo > x.Rank() {
		return nil, fmt.Errorf("eixo %d fora da forma %v", eixo, x.Shape)
	}

	esquerda, direita := 1, 1
	for _, d := range x.Shape[:eixo] {
		esquerda *= d
	}
	for _, d := range x.Shape[eixo:] {
		direita *= d
	}

	return garanteContiguo(ws, x).Reshape(esquerda, direita)
}

func montaFlatten(b *builder, n *onnx.Node) (*operation, error) {
	eixo := int(n.AttrInt("axis", 1))
	return novaOp(n, func(ws *nn.Workspace, ins []*tensor.Tensor) ([]*tensor.Tensor, error) {
		out, err := achatarEm(ws, ins[0], eixo)
		if err != nil {
			return nil, err
		}
		return []*tensor.Tensor{out}, nil
	}), nil
}

// montaReshape muda a forma sem mover dados.
//
// A nova forma vem como TENSOR de int64, nao como atributo -- foi uma
// mudanca do opset 5, para permitir formas calculadas em execucao. Aqui ela
// precisa ser constante.
//
// Dois valores tem significado especial: 0 copia a dimensao correspondente da
// entrada, e -1 e inferido a partir do total de elementos.
func montaReshape(b *builder, n *onnx.Node) (*operation, error) {
	if len(n.Inputs) != 2 {
		return nil, fmt.Errorf("Reshape espera 2 entradas, recebeu %d", len(n.Inputs))
	}

	formaT, err := b.peso(n, 1, "forma")
	if err != nil {
		return nil, err
	}

	// A forma chega como float32 porque o carregamento converte tudo; os
	// valores sao inteiros exatos e pequenos, entao a volta e segura.
	crua := formaT.Flat()
	alvo := make([]int, len(crua))
	for i, v := range crua {
		alvo[i] = int(v)
	}

	// allowzero (opset 14) inverte o significado do zero: em vez de copiar a
	// dimensao da entrada, passa a significar dimensao vazia mesmo.
	permiteZero := n.AttrInt("allowzero", 0) != 0

	return novaOp(n, func(ws *nn.Workspace, ins []*tensor.Tensor) ([]*tensor.Tensor, error) {
		x := ins[0]
		forma := make([]int, len(alvo))
		copy(forma, alvo)

		for i, d := range forma {
			if d == 0 && !permiteZero {
				if i >= x.Rank() {
					return nil, fmt.Errorf("a forma pede copiar a dimensao %d, que a entrada %v nao tem", i, x.Shape)
				}
				forma[i] = x.Shape[i]
			}
		}

		// Reshape do pacote tensor ja infere o -1 e valida o total.
		out, err := garanteContiguo(ws, x).Reshape(forma...)
		if err != nil {
			return nil, err
		}
		return []*tensor.Tensor{out}, nil
	}), nil
}

func montaTranspose(b *builder, n *onnx.Node) (*operation, error) {
	perm := n.AttrInts("perm", nil)

	return novaOp(n, func(ws *nn.Workspace, ins []*tensor.Tensor) ([]*tensor.Tensor, error) {
		x := ins[0]

		p := make([]int, len(perm))
		for i, v := range perm {
			p[i] = int(v)
		}
		// Sem perm, o ONNX inverte a ordem das dimensoes -- que e o padrao
		// do Transpose do pacote tensor quando nao recebe argumentos.

		vista, err := x.Transpose(p...)
		if err != nil {
			return nil, err
		}

		// Transpose devolve uma view de passos trocados. Materializar aqui
		// custa uma copia, e evita que toda operacao seguinte precise saber
		// lidar com tensores nao contiguos.
		out := ws.Tensor(vista.Shape...)
		copy(out.Data, vista.Flat())
		return []*tensor.Tensor{out}, nil
	}), nil
}

// montaConcat junta tensores ao longo de um eixo.
func montaConcat(b *builder, n *onnx.Node) (*operation, error) {
	if len(n.Inputs) == 0 {
		return nil, fmt.Errorf("Concat sem entradas")
	}
	a := n.Attr("axis")
	if a == nil {
		return nil, fmt.Errorf("Concat sem o atributo axis")
	}
	eixo := int(a.I)

	return novaOp(n, func(ws *nn.Workspace, ins []*tensor.Tensor) ([]*tensor.Tensor, error) {
		out, err := concatenar(ws, ins, eixo)
		if err != nil {
			return nil, err
		}
		return []*tensor.Tensor{out}, nil
	}), nil
}

func concatenar(ws *nn.Workspace, ins []*tensor.Tensor, eixo int) (*tensor.Tensor, error) {
	primeiro := ins[0]
	e := eixo
	if e < 0 {
		e += primeiro.Rank()
	}
	if e < 0 || e >= primeiro.Rank() {
		return nil, fmt.Errorf("eixo %d fora da forma %v", eixo, primeiro.Shape)
	}

	// Todas as formas precisam bater, menos no eixo da juncao.
	forma := append([]int(nil), primeiro.Shape...)
	forma[e] = 0
	for _, t := range ins {
		if t.Rank() != primeiro.Rank() {
			return nil, fmt.Errorf("Concat recebeu ranks diferentes: %v e %v", primeiro.Shape, t.Shape)
		}
		for d := range t.Shape {
			if d != e && t.Shape[d] != primeiro.Shape[d] {
				return nil, fmt.Errorf("Concat: formas %v e %v diferem fora do eixo %d",
					primeiro.Shape, t.Shape, e)
			}
		}
		forma[e] += t.Shape[e]
	}

	out := ws.Tensor(forma...)

	// A copia acontece em blocos: tudo a esquerda do eixo se repete, tudo a
	// direita e contiguo. Assim cada bloco e um unico copy.
	externo := 1
	for _, d := range forma[:e] {
		externo *= d
	}
	interno := 1
	for _, d := range forma[e+1:] {
		interno *= d
	}

	destino := 0
	for o := 0; o < externo; o++ {
		for _, t := range ins {
			bloco := t.Shape[e] * interno
			src := t.Flat()
			copy(out.Data[destino:destino+bloco], src[o*bloco:(o+1)*bloco])
			destino += bloco
		}
	}

	return out, nil
}

// eixosDe le a lista de eixos de Unsqueeze/Squeeze, que ate o opset 12 era
// atributo e a partir do 13 virou entrada.
func eixosDe(b *builder, n *onnx.Node) ([]int, error) {
	if a := n.Attr("axes"); a != nil && len(a.Ints) > 0 {
		out := make([]int, len(a.Ints))
		for i, v := range a.Ints {
			out[i] = int(v)
		}
		return out, nil
	}

	t, err := b.pesoOpcional(n, 1, "eixos")
	if err != nil {
		return nil, err
	}
	if t == nil {
		return nil, nil
	}

	crua := t.Flat()
	out := make([]int, len(crua))
	for i, v := range crua {
		out[i] = int(v)
	}
	return out, nil
}

// montaUnsqueeze insere dimensoes de tamanho 1 nas posicoes dadas.
func montaUnsqueeze(b *builder, n *onnx.Node) (*operation, error) {
	eixos, err := eixosDe(b, n)
	if err != nil {
		return nil, err
	}
	if len(eixos) == 0 {
		return nil, fmt.Errorf("Unsqueeze sem eixos")
	}

	op := novaOp(n, func(ws *nn.Workspace, ins []*tensor.Tensor) ([]*tensor.Tensor, error) {
		x := ins[0]
		novoRank := x.Rank() + len(eixos)

		inserir := make(map[int]bool, len(eixos))
		for _, e := range eixos {
			if e < 0 {
				e += novoRank
			}
			if e < 0 || e >= novoRank {
				return nil, fmt.Errorf("eixo %d fora do rank final %d", e, novoRank)
			}
			inserir[e] = true
		}

		forma := make([]int, 0, novoRank)
		orig := 0
		for i := 0; i < novoRank; i++ {
			if inserir[i] {
				forma = append(forma, 1)
				continue
			}
			forma = append(forma, x.Shape[orig])
			orig++
		}

		out, err := garanteContiguo(ws, x).Reshape(forma...)
		if err != nil {
			return nil, err
		}
		return []*tensor.Tensor{out}, nil
	})
	op.inputs = n.Inputs[:1]
	return op, nil
}

// montaSqueeze remove dimensoes de tamanho 1. Sem eixos, remove todas.
func montaSqueeze(b *builder, n *onnx.Node) (*operation, error) {
	eixos, err := eixosDe(b, n)
	if err != nil {
		return nil, err
	}

	op := novaOp(n, func(ws *nn.Workspace, ins []*tensor.Tensor) ([]*tensor.Tensor, error) {
		x := ins[0]

		remover := make(map[int]bool, len(eixos))
		for _, e := range eixos {
			if e < 0 {
				e += x.Rank()
			}
			if e < 0 || e >= x.Rank() {
				return nil, fmt.Errorf("eixo %d fora da forma %v", e, x.Shape)
			}
			if x.Shape[e] != 1 {
				return nil, fmt.Errorf("Squeeze do eixo %d, que tem tamanho %d e nao 1", e, x.Shape[e])
			}
			remover[e] = true
		}

		forma := make([]int, 0, x.Rank())
		for i, d := range x.Shape {
			if len(eixos) == 0 {
				if d != 1 {
					forma = append(forma, d)
				}
				continue
			}
			if !remover[i] {
				forma = append(forma, d)
			}
		}

		out, err := garanteContiguo(ws, x).Reshape(forma...)
		if err != nil {
			return nil, err
		}
		return []*tensor.Tensor{out}, nil
	})
	op.inputs = n.Inputs[:1]
	return op, nil
}
