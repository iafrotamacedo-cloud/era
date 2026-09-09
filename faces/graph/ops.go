package graph

import (
	"fmt"

	"github.com/iafrotamacedo-cloud/era/faces/nn"
	"github.com/iafrotamacedo-cloud/era/faces/onnx"
	"github.com/iafrotamacedo-cloud/era/faces/tensor"
)

// montador constroi a operacao executavel de um no.
//
// Toda validacao acontece aqui, na montagem: forma dos pesos, atributos
// coerentes, combinacoes nao suportadas. O que sai daqui roda sem conferir
// nada, uma vez por rosto.
type montador func(b *builder, n *onnx.Node) (*operation, error)

// registro mapeia o nome do operador ONNX para o seu montador.
//
// E preenchido em init para que os montadores possam, no futuro, consultar o
// proprio registro -- um operador de subgrafo, por exemplo -- sem criar ciclo
// de inicializacao.
var registro map[string]montador

func init() {
	registro = map[string]montador{
		// Com pesos
		"Conv":               montaConv,
		"BatchNormalization": montaBatchNorm,
		"PRelu":              montaPRelu,
		"Gemm":               montaGemm,
		"MatMul":             montaMatMul,

		// Espacial
		"Resize":            montaResize,
		"GlobalAveragePool": montaGlobalAvgPool,
		"MaxPool":           montaMaxPool,
		"AveragePool":       montaAveragePool,

		// Ativacoes
		"Relu":      montaRelu,
		"LeakyRelu": montaLeakyRelu,
		"Sigmoid":   montaSigmoid,
		"Tanh":      montaTanh,
		"Clip":      montaClip,
		"Softmax":   montaSoftmax,

		// Aritmetica
		"Add": montaBinario("Add"),
		"Sub": montaBinario("Sub"),
		"Mul": montaBinario("Mul"),
		"Div": montaBinario("Div"),

		// Forma
		"Flatten":   montaFlatten,
		"Reshape":   montaReshape,
		"Transpose": montaTranspose,
		"Concat":    montaConcat,
		"Unsqueeze": montaUnsqueeze,
		"Squeeze":   montaSqueeze,

		// Sem efeito na inferencia
		"Identity": montaIdentity,
		"Dropout":  montaIdentity,
		"Constant": montaConstant,
	}
}

// builder carrega o que os montadores precisam consultar.
type builder struct {
	consts map[string]*tensor.Tensor
}

// peso busca uma entrada que precisa ser constante.
//
// Pesos treinados sao initializers e estao disponiveis na montagem. Uma
// convolucao cujos pesos venham de outro no seria uma rede que se modifica
// sozinha -- coisa de treino, nao de inferencia.
func (b *builder) peso(n *onnx.Node, idx int, papel string) (*tensor.Tensor, error) {
	if idx >= len(n.Inputs) || n.Inputs[idx] == "" {
		return nil, fmt.Errorf("falta a entrada %d (%s)", idx, papel)
	}
	t, ok := b.consts[n.Inputs[idx]]
	if !ok {
		return nil, fmt.Errorf("a entrada %d (%s) e %q, que precisa ser um peso constante",
			idx, papel, n.Inputs[idx])
	}
	return t, nil
}

// pesoOpcional e como peso, mas devolve nil quando a entrada foi omitida.
func (b *builder) pesoOpcional(n *onnx.Node, idx int, papel string) (*tensor.Tensor, error) {
	if idx >= len(n.Inputs) || n.Inputs[idx] == "" {
		return nil, nil
	}
	return b.peso(n, idx, papel)
}

// nomeDe devolve o nome do no, ou algo util quando ele nao tem nome.
func nomeDe(n *onnx.Node) string {
	if n.Name != "" {
		return n.Name
	}
	if len(n.Outputs) > 0 && n.Outputs[0] != "" {
		return n.Outputs[0]
	}
	return n.OpType
}

// novaOp monta a operacao com os campos comuns preenchidos.
func novaOp(n *onnx.Node, exec func(*nn.Workspace, []*tensor.Tensor) ([]*tensor.Tensor, error)) *operation {
	return &operation{
		name:    nomeDe(n),
		opType:  n.OpType,
		inputs:  n.Inputs,
		outputs: n.Outputs,
		exec:    exec,
	}
}

// exigeSaidaUnica recusa nos com mais de uma saida.
//
// Alguns operadores tem saidas extras que so existem no treino --
// BatchNormalization devolve as estatisticas do lote, Dropout devolve a
// mascara. Um modelo exportado com elas nao e um modelo de inferencia, e
// tratar isso como detalhe produziria resultado silenciosamente errado.
func exigeSaidaUnica(n *onnx.Node) error {
	uteis := 0
	for _, o := range n.Outputs {
		if o != "" {
			uteis++
		}
	}
	if uteis != 1 {
		return fmt.Errorf("%s tem %d saidas; a ERA so executa inferencia, que produz uma",
			n.OpType, uteis)
	}
	return nil
}

// par le um atributo de dois inteiros, aplicando o padrao.
func par(n *onnx.Node, nome string, padrao int64) (int, int, error) {
	v := n.AttrInts(nome, []int64{padrao, padrao})
	if len(v) != 2 {
		return 0, 0, fmt.Errorf("%s tem %d valores, a ERA so trata convolucao 2D", nome, len(v))
	}
	return int(v[0]), int(v[1]), nil
}

// padding le o atributo pads do ONNX, que vem como
// [inicio_h, inicio_w, fim_h, fim_w].
//
// O kernel da ERA usa padding simetrico -- um valor por eixo. Padding
// assimetrico existe no formato e e raro em redes de reconhecimento facial;
// tratar como erro e melhor do que aproximar em silencio.
func padding(n *onnx.Node) (int, int, error) {
	autoPad := n.AttrString("auto_pad", "NOTSET")
	switch autoPad {
	case "NOTSET", "":
		// segue para os pads explicitos
	case "VALID":
		return 0, 0, nil
	default:
		return 0, 0, fmt.Errorf("auto_pad=%q depende da forma da entrada e ainda nao e suportado; "+
			"reexporte o modelo com pads explicitos", autoPad)
	}

	p := n.AttrInts("pads", []int64{0, 0, 0, 0})
	if len(p) != 4 {
		return 0, 0, fmt.Errorf("pads tem %d valores, a ERA so trata convolucao 2D (4 valores)", len(p))
	}
	if p[0] != p[2] || p[1] != p[3] {
		return 0, 0, fmt.Errorf("padding assimetrico (inicio %d,%d fim %d,%d) ainda nao e suportado",
			p[0], p[1], p[2], p[3])
	}
	return int(p[0]), int(p[1]), nil
}

func montaConv(b *builder, n *onnx.Node) (*operation, error) {
	if len(n.Inputs) < 2 || len(n.Inputs) > 3 {
		return nil, fmt.Errorf("Conv espera 2 ou 3 entradas, recebeu %d", len(n.Inputs))
	}
	if err := exigeSaidaUnica(n); err != nil {
		return nil, err
	}

	w, err := b.peso(n, 1, "pesos")
	if err != nil {
		return nil, err
	}
	if w.Rank() != 4 {
		return nil, fmt.Errorf("pesos tem forma %v; a ERA so trata convolucao 2D", w.Shape)
	}

	grupos := int(n.AttrInt("group", 1))
	outC, inPorGrupo, kh, kw := w.Shape[0], w.Shape[1], w.Shape[2], w.Shape[3]

	if ks := n.AttrInts("kernel_shape", nil); ks != nil {
		if len(ks) != 2 || int(ks[0]) != kh || int(ks[1]) != kw {
			return nil, fmt.Errorf("kernel_shape %v nao bate com a forma dos pesos %v", ks, w.Shape)
		}
	}

	strideH, strideW, err := par(n, "strides", 1)
	if err != nil {
		return nil, err
	}
	dilH, dilW, err := par(n, "dilations", 1)
	if err != nil {
		return nil, err
	}
	padH, padW, err := padding(n)
	if err != nil {
		return nil, err
	}

	var bias []float32
	if bt, err := b.pesoOpcional(n, 2, "vies"); err != nil {
		return nil, err
	} else if bt != nil {
		bias = bt.Flat()
	}

	camada, err := nn.NewConv2D(nomeDe(n), nn.Conv2DConfig{
		InC: inPorGrupo * grupos, OutC: outC,
		KH: kh, KW: kw,
		StrideH: strideH, StrideW: strideW,
		PadH: padH, PadW: padW,
		DilH: dilH, DilW: dilW,
		Groups: grupos,
	}, w.Flat(), bias)
	if err != nil {
		return nil, err
	}

	return novaOp(n, func(ws *nn.Workspace, ins []*tensor.Tensor) ([]*tensor.Tensor, error) {
		out, err := camada.Forward(ws, ins[0])
		if err != nil {
			return nil, err
		}
		return []*tensor.Tensor{out}, nil
	}), nil
}

func montaBatchNorm(b *builder, n *onnx.Node) (*operation, error) {
	if len(n.Inputs) != 5 {
		return nil, fmt.Errorf("BatchNormalization espera 5 entradas, recebeu %d", len(n.Inputs))
	}
	if err := exigeSaidaUnica(n); err != nil {
		return nil, err
	}

	papeis := []string{"", "gama", "beta", "media", "variancia"}
	vals := make([][]float32, 5)
	for i := 1; i < 5; i++ {
		t, err := b.peso(n, i, papeis[i])
		if err != nil {
			return nil, err
		}
		vals[i] = t.Flat()
	}

	eps := n.AttrFloat("epsilon", 1e-5)
	camada, err := nn.NewBatchNorm(nomeDe(n), vals[1], vals[2], vals[3], vals[4], eps)
	if err != nil {
		return nil, err
	}

	return novaOp(n, func(ws *nn.Workspace, ins []*tensor.Tensor) ([]*tensor.Tensor, error) {
		out, err := camada.Forward(ws, ins[0])
		if err != nil {
			return nil, err
		}
		return []*tensor.Tensor{out}, nil
	}), nil
}

func montaPRelu(b *builder, n *onnx.Node) (*operation, error) {
	if len(n.Inputs) != 2 {
		return nil, fmt.Errorf("PRelu espera 2 entradas, recebeu %d", len(n.Inputs))
	}

	// O ONNX grava a inclinacao como [C], [C,1,1] ou escalar. Todas
	// significam um valor por canal (ou um so, compartilhado), entao achatar
	// resolve as tres.
	slope, err := b.peso(n, 1, "inclinacao")
	if err != nil {
		return nil, err
	}

	camada, err := nn.NewPReLU(nomeDe(n), slope.Flat())
	if err != nil {
		return nil, err
	}

	return novaOp(n, func(ws *nn.Workspace, ins []*tensor.Tensor) ([]*tensor.Tensor, error) {
		out, err := camada.Forward(ws, ins[0])
		if err != nil {
			return nil, err
		}
		return []*tensor.Tensor{out}, nil
	}), nil
}

// montaGemm trata a camada totalmente conectada.
//
//	Y = alpha * A' * B' + beta * C
//
// onde A' e A transposta se transA, e B' idem. Na pratica, uma camada
// densa exportada vem sempre com alpha = beta = 1 e transA = 0. Os demais
// casos existem na especificacao e nao aparecem em redes de reconhecimento
// facial -- recusar e melhor que implementar sem ter como testar.
func montaGemm(b *builder, n *onnx.Node) (*operation, error) {
	if len(n.Inputs) < 2 || len(n.Inputs) > 3 {
		return nil, fmt.Errorf("Gemm espera 2 ou 3 entradas, recebeu %d", len(n.Inputs))
	}

	if a := n.AttrFloat("alpha", 1); a != 1 {
		return nil, fmt.Errorf("Gemm com alpha=%v ainda nao e suportado", a)
	}
	if be := n.AttrFloat("beta", 1); be != 1 {
		return nil, fmt.Errorf("Gemm com beta=%v ainda nao e suportado", be)
	}
	if n.AttrInt("transA", 0) != 0 {
		return nil, fmt.Errorf("Gemm com transA=1 ainda nao e suportado")
	}

	w, err := b.peso(n, 1, "pesos")
	if err != nil {
		return nil, err
	}
	if w.Rank() != 2 {
		return nil, fmt.Errorf("os pesos do Gemm tem forma %v, quero 2 dimensoes", w.Shape)
	}

	// nn.Linear guarda os pesos como [saidas, entradas], que e exatamente o
	// layout de transB=1. Com transB=0 e preciso transpor uma vez, aqui.
	var pesos []float32
	var inF, outF int

	if n.AttrInt("transB", 0) != 0 {
		outF, inF = w.Shape[0], w.Shape[1]
		pesos = w.Flat()
	} else {
		inF, outF = w.Shape[0], w.Shape[1]
		wt, err := w.Transpose()
		if err != nil {
			return nil, err
		}
		pesos = wt.Flat()
	}

	var bias []float32
	if bt, err := b.pesoOpcional(n, 2, "vies"); err != nil {
		return nil, err
	} else if bt != nil {
		bias = bt.Flat()
		if len(bias) != outF {
			return nil, fmt.Errorf("o vies tem %d valores, a saida tem %d", len(bias), outF)
		}
	}

	camada, err := nn.NewLinear(nomeDe(n), inF, outF, pesos, bias)
	if err != nil {
		return nil, err
	}

	return novaOp(n, func(ws *nn.Workspace, ins []*tensor.Tensor) ([]*tensor.Tensor, error) {
		x := ins[0]
		// Gemm exige entrada 2D. Um modelo que venha de um mapa de
		// caracteristicas costuma trazer um Flatten antes, mas nem sempre.
		if x.Rank() != 2 {
			achatado, err := achatarEm(ws, x, 1)
			if err != nil {
				return nil, err
			}
			x = achatado
		}
		out, err := camada.Forward(ws, x)
		if err != nil {
			return nil, err
		}
		return []*tensor.Tensor{out}, nil
	}), nil
}

// montaMatMul trata a multiplicacao de matrizes com o segundo operando
// constante -- que e como uma camada densa sem vies costuma aparecer.
func montaMatMul(b *builder, n *onnx.Node) (*operation, error) {
	if len(n.Inputs) != 2 {
		return nil, fmt.Errorf("MatMul espera 2 entradas, recebeu %d", len(n.Inputs))
	}

	w, err := b.pesoOpcional(n, 1, "segundo operando")
	if err != nil {
		return nil, err
	}
	if w == nil {
		return nil, fmt.Errorf("MatMul com os dois operandos dinamicos ainda nao e suportado")
	}
	if w.Rank() != 2 {
		return nil, fmt.Errorf("o segundo operando tem forma %v, quero 2 dimensoes", w.Shape)
	}

	// [K,M] no ONNX; nn.Linear quer [M,K].
	inF, outF := w.Shape[0], w.Shape[1]
	wt, err := w.Transpose()
	if err != nil {
		return nil, err
	}

	camada, err := nn.NewLinear(nomeDe(n), inF, outF, wt.Flat(), nil)
	if err != nil {
		return nil, err
	}

	return novaOp(n, func(ws *nn.Workspace, ins []*tensor.Tensor) ([]*tensor.Tensor, error) {
		x := ins[0]
		if x.Rank() != 2 {
			achatado, err := achatarEm(ws, x, x.Rank()-1)
			if err != nil {
				return nil, err
			}
			x = achatado
		}
		out, err := camada.Forward(ws, x)
		if err != nil {
			return nil, err
		}
		return []*tensor.Tensor{out}, nil
	}), nil
}

func montaGlobalAvgPool(b *builder, n *onnx.Node) (*operation, error) {
	if len(n.Inputs) != 1 {
		return nil, fmt.Errorf("GlobalAveragePool espera 1 entrada, recebeu %d", len(n.Inputs))
	}
	camada := &nn.GlobalAvgPool{}

	return novaOp(n, func(ws *nn.Workspace, ins []*tensor.Tensor) ([]*tensor.Tensor, error) {
		out, err := camada.Forward(ws, ins[0])
		if err != nil {
			return nil, err
		}
		return []*tensor.Tensor{out}, nil
	}), nil
}

func montaMaxPool(b *builder, n *onnx.Node) (*operation, error) {
	if len(n.Inputs) != 1 {
		return nil, fmt.Errorf("MaxPool espera 1 entrada, recebeu %d", len(n.Inputs))
	}
	if err := exigeSaidaUnica(n); err != nil {
		return nil, err
	}
	if n.AttrInt("ceil_mode", 0) != 0 {
		return nil, fmt.Errorf("MaxPool com ceil_mode=1 ainda nao e suportado")
	}

	kh, kw, err := par(n, "kernel_shape", 0)
	if err != nil {
		return nil, err
	}
	if kh == 0 || kw == 0 {
		return nil, fmt.Errorf("MaxPool sem kernel_shape")
	}

	strideH, strideW, err := par(n, "strides", 0)
	if err != nil {
		return nil, err
	}
	padH, padW, err := padding(n)
	if err != nil {
		return nil, err
	}

	camada, err := nn.NewMaxPool2D(nomeDe(n), kh, kw, strideH, strideW, padH, padW)
	if err != nil {
		return nil, err
	}

	return novaOp(n, func(ws *nn.Workspace, ins []*tensor.Tensor) ([]*tensor.Tensor, error) {
		out, err := camada.Forward(ws, ins[0])
		if err != nil {
			return nil, err
		}
		return []*tensor.Tensor{out}, nil
	}), nil
}

// montaAveragePool implementa a media por janela.
//
// count_include_pad decide se as posicoes do padding entram no divisor. O
// padrao do ONNX e nao entrarem: a media e sobre os pixels reais da janela.
func montaAveragePool(b *builder, n *onnx.Node) (*operation, error) {
	if len(n.Inputs) != 1 {
		return nil, fmt.Errorf("AveragePool espera 1 entrada, recebeu %d", len(n.Inputs))
	}
	if n.AttrInt("ceil_mode", 0) != 0 {
		return nil, fmt.Errorf("AveragePool com ceil_mode=1 ainda nao e suportado")
	}

	kh, kw, err := par(n, "kernel_shape", 0)
	if err != nil {
		return nil, err
	}
	if kh == 0 || kw == 0 {
		return nil, fmt.Errorf("AveragePool sem kernel_shape")
	}

	strideH, strideW, err := par(n, "strides", 0)
	if err != nil {
		return nil, err
	}
	if strideH == 0 {
		strideH = kh
	}
	if strideW == 0 {
		strideW = kw
	}

	padH, padW, err := padding(n)
	if err != nil {
		return nil, err
	}
	incluiPad := n.AttrInt("count_include_pad", 0) != 0

	return novaOp(n, func(ws *nn.Workspace, ins []*tensor.Tensor) ([]*tensor.Tensor, error) {
		out, err := mediaPorJanela(ws, ins[0], kh, kw, strideH, strideW, padH, padW, incluiPad)
		if err != nil {
			return nil, err
		}
		return []*tensor.Tensor{out}, nil
	}), nil
}

func mediaPorJanela(ws *nn.Workspace, x *tensor.Tensor, kh, kw, sh, sw, ph, pw int, incluiPad bool) (*tensor.Tensor, error) {
	if x.Rank() != 4 {
		return nil, fmt.Errorf("AveragePool espera [N,C,H,W], recebeu %v", x.Shape)
	}

	n, c, h, w := x.Shape[0], x.Shape[1], x.Shape[2], x.Shape[3]
	oh := (h+2*ph-kh)/sh + 1
	ow := (w+2*pw-kw)/sw + 1
	if oh <= 0 || ow <= 0 {
		return nil, fmt.Errorf("AveragePool produz saida vazia %dx%d", oh, ow)
	}

	src := x.Flat()
	out := ws.Tensor(n, c, oh, ow)

	for plano := 0; plano < n*c; plano++ {
		in := src[plano*h*w : (plano+1)*h*w]
		o := out.Data[plano*oh*ow : (plano+1)*oh*ow]

		for y := 0; y < oh; y++ {
			for xx := 0; xx < ow; xx++ {
				var soma float32
				validos := 0

				for i := 0; i < kh; i++ {
					iy := y*sh - ph + i
					if iy < 0 || iy >= h {
						continue
					}
					for j := 0; j < kw; j++ {
						ix := xx*sw - pw + j
						if ix < 0 || ix >= w {
							continue
						}
						soma += in[iy*w+ix]
						validos++
					}
				}

				divisor := validos
				if incluiPad {
					divisor = kh * kw
				}
				if divisor == 0 {
					o[y*ow+xx] = 0
					continue
				}
				o[y*ow+xx] = soma / float32(divisor)
			}
		}
	}

	return out, nil
}

// garanteContiguo devolve um tensor com os dados em ordem natural, usando o
// workspace quando for preciso materializar.
func garanteContiguo(ws *nn.Workspace, x *tensor.Tensor) *tensor.Tensor {
	if x.Offset == 0 && x.IsContiguous() && len(x.Data) == x.Size() {
		return x
	}
	out := ws.Tensor(x.Shape...)
	copy(out.Data, x.Flat())
	return out
}
