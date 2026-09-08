package graph

import (
	"fmt"
	"math"

	"github.com/iafrotamacedo-cloud/era/faces/nn"
	"github.com/iafrotamacedo-cloud/era/faces/onnx"
	"github.com/iafrotamacedo-cloud/era/faces/tensor"
)

// mapaUnario monta uma operacao que aplica a mesma funcao a cada elemento.
func mapaUnario(n *onnx.Node, f func(float32) float32) *operation {
	return novaOp(n, func(ws *nn.Workspace, ins []*tensor.Tensor) ([]*tensor.Tensor, error) {
		src := ins[0].Flat()
		out := ws.Tensor(ins[0].Shape...)
		for i, v := range src {
			out.Data[i] = f(v)
		}
		return []*tensor.Tensor{out}, nil
	})
}

func montaRelu(b *builder, n *onnx.Node) (*operation, error) {
	return mapaUnario(n, func(v float32) float32 {
		if v > 0 {
			return v
		}
		return 0
	}), nil
}

func montaLeakyRelu(b *builder, n *onnx.Node) (*operation, error) {
	alpha := n.AttrFloat("alpha", 0.01)
	return mapaUnario(n, func(v float32) float32 {
		if v > 0 {
			return v
		}
		return alpha * v
	}), nil
}

func montaSigmoid(b *builder, n *onnx.Node) (*operation, error) {
	return mapaUnario(n, func(v float32) float32 {
		return float32(1 / (1 + math.Exp(-float64(v))))
	}), nil
}

func montaTanh(b *builder, n *onnx.Node) (*operation, error) {
	return mapaUnario(n, func(v float32) float32 {
		return float32(math.Tanh(float64(v)))
	}), nil
}

// montaClip corta os valores num intervalo. E como ReLU6 e outras ativacoes
// limitadas aparecem no ONNX.
//
// Ate o opset 10, os limites eram atributos; a partir do 11, viraram
// entradas opcionais. Os dois formatos circulam, e um modelo exportado por
// ferramenta antiga usa o primeiro.
func montaClip(b *builder, n *onnx.Node) (*operation, error) {
	minimo := float32(math.Inf(-1))
	maximo := float32(math.Inf(1))

	if a := n.Attr("min"); a != nil {
		minimo = a.F
	}
	if a := n.Attr("max"); a != nil {
		maximo = a.F
	}

	if t, err := b.pesoOpcional(n, 1, "minimo"); err != nil {
		return nil, err
	} else if t != nil && t.Size() > 0 {
		minimo = t.Flat()[0]
	}
	if t, err := b.pesoOpcional(n, 2, "maximo"); err != nil {
		return nil, err
	} else if t != nil && t.Size() > 0 {
		maximo = t.Flat()[0]
	}

	if minimo > maximo {
		return nil, fmt.Errorf("Clip com minimo %v maior que o maximo %v", minimo, maximo)
	}

	op := mapaUnario(n, func(v float32) float32 {
		if v < minimo {
			return minimo
		}
		if v > maximo {
			return maximo
		}
		return v
	})
	// Clip pode ter 3 entradas, mas so a primeira e consumida em execucao.
	op.inputs = n.Inputs[:1]
	return op, nil
}

// montaSoftmax normaliza ao longo de um eixo, transformando numeros em
// probabilidades que somam 1.
//
// A semantica implementada e a do opset 13 em diante: a normalizacao ocorre
// AO LONGO DO EIXO indicado. Ate o opset 12, o tensor era achatado em duas
// dimensoes a partir do eixo e a normalizacao ocorria sobre tudo a direita --
// resultado diferente para tensores de mais de duas dimensoes. Modelos
// exportados nos ultimos anos usam a semantica nova.
func montaSoftmax(b *builder, n *onnx.Node) (*operation, error) {
	eixo := int(n.AttrInt("axis", -1))

	return novaOp(n, func(ws *nn.Workspace, ins []*tensor.Tensor) ([]*tensor.Tensor, error) {
		x := ins[0]
		e := eixo
		if e < 0 {
			e += x.Rank()
		}
		if e < 0 || e >= x.Rank() {
			return nil, fmt.Errorf("eixo %d fora da forma %v", eixo, x.Shape)
		}

		externo := 1
		for _, d := range x.Shape[:e] {
			externo *= d
		}
		comprimento := x.Shape[e]
		interno := 1
		for _, d := range x.Shape[e+1:] {
			interno *= d
		}

		src := x.Flat()
		out := ws.Tensor(x.Shape...)

		for o := 0; o < externo; o++ {
			for i := 0; i < interno; i++ {
				base := o*comprimento*interno + i

				// Subtrair o maior antes de exponenciar evita estouro: sem
				// isso, um valor alto vira +Inf e a soma inteira vira NaN.
				maior := float32(math.Inf(-1))
				for k := 0; k < comprimento; k++ {
					if v := src[base+k*interno]; v > maior {
						maior = v
					}
				}

				var soma float32
				for k := 0; k < comprimento; k++ {
					v := float32(math.Exp(float64(src[base+k*interno] - maior)))
					out.Data[base+k*interno] = v
					soma += v
				}
				if soma == 0 {
					continue
				}
				for k := 0; k < comprimento; k++ {
					out.Data[base+k*interno] /= soma
				}
			}
		}

		return []*tensor.Tensor{out}, nil
	}), nil
}

// montaBinario cria o montador de uma operacao aritmetica de dois operandos.
func montaBinario(tipo string) montador {
	var f func(a, b float32) float32
	switch tipo {
	case "Add":
		f = func(a, b float32) float32 { return a + b }
	case "Sub":
		f = func(a, b float32) float32 { return a - b }
	case "Mul":
		f = func(a, b float32) float32 { return a * b }
	case "Div":
		f = func(a, b float32) float32 { return a / b }
	}

	return func(b *builder, n *onnx.Node) (*operation, error) {
		if len(n.Inputs) != 2 {
			return nil, fmt.Errorf("%s espera 2 entradas, recebeu %d", tipo, len(n.Inputs))
		}
		return novaOp(n, func(ws *nn.Workspace, ins []*tensor.Tensor) ([]*tensor.Tensor, error) {
			out, err := aplicarBinario(ws, ins[0], ins[1], f)
			if err != nil {
				return nil, err
			}
			return []*tensor.Tensor{out}, nil
		}), nil
	}
}

// aplicarBinario combina dois tensores elemento a elemento, com transmissao
// de forma no estilo do NumPy.
//
// Uma conexao residual soma tensores de forma identica, e ai o caminho rapido
// resolve. Mas escalonamento por canal aparece como [N,C,H,W] vezes [C,1,1],
// e recusar isso quebraria modelos reais.
func aplicarBinario(ws *nn.Workspace, a, b *tensor.Tensor, f func(x, y float32) float32) (*tensor.Tensor, error) {
	af, bf := a.Flat(), b.Flat()

	// Caminho rapido: mesma forma, percurso linear.
	if a.SameShape(b) {
		out := ws.Tensor(a.Shape...)
		for i, v := range af {
			out.Data[i] = f(v, bf[i])
		}
		return out, nil
	}

	forma, err := formaTransmitida(a.Shape, b.Shape)
	if err != nil {
		return nil, err
	}

	sa := stridesTransmitidos(a.Shape, forma)
	sb := stridesTransmitidos(b.Shape, forma)

	out := ws.Tensor(forma...)
	total := out.Size()
	if total == 0 {
		return out, nil
	}

	idx := make([]int, len(forma))
	posA, posB := 0, 0

	for i := 0; i < total; i++ {
		out.Data[i] = f(af[posA], bf[posB])

		// Avanca o indice multidimensional e, junto, as posicoes nos dois
		// operandos. Recalcular a posicao do zero a cada elemento custaria
		// uma multiplicacao por dimensao.
		for d := len(forma) - 1; d >= 0; d-- {
			idx[d]++
			posA += sa[d]
			posB += sb[d]
			if idx[d] < forma[d] {
				break
			}
			posA -= sa[d] * forma[d]
			posB -= sb[d] * forma[d]
			idx[d] = 0
		}
	}

	return out, nil
}

// formaTransmitida calcula a forma resultante da transmissao entre duas
// formas, pelas regras do NumPy: alinha pela direita, e cada par de
// dimensoes precisa ser igual ou ter um dos lados valendo 1.
func formaTransmitida(a, b []int) ([]int, error) {
	n := len(a)
	if len(b) > n {
		n = len(b)
	}

	out := make([]int, n)
	for i := 0; i < n; i++ {
		da, db := 1, 1
		if j := len(a) - n + i; j >= 0 {
			da = a[j]
		}
		if j := len(b) - n + i; j >= 0 {
			db = b[j]
		}

		switch {
		case da == db:
			out[i] = da
		case da == 1:
			out[i] = db
		case db == 1:
			out[i] = da
		default:
			return nil, fmt.Errorf("formas %v e %v nao sao compativeis para transmissao", a, b)
		}
	}
	return out, nil
}

// stridesTransmitidos devolve os passos de um operando no espaco da forma de
// saida. Dimensoes transmitidas -- as de tamanho 1, e as que o operando nem
// tem -- recebem passo zero, o que faz o percurso repetir o mesmo valor.
func stridesTransmitidos(forma, saida []int) []int {
	st := make([]int, len(saida))
	acc := 1
	for i := len(forma) - 1; i >= 0; i-- {
		j := len(saida) - len(forma) + i
		if forma[i] != 1 {
			st[j] = acc
		}
		acc *= forma[i]
	}
	return st
}
