package graph

import (
	"math"
	"math/rand"
	"reflect"
	"strings"
	"testing"

	"github.com/iafrotamacedo-cloud/era/faces/nn"
	"github.com/iafrotamacedo-cloud/era/faces/onnx"
	"github.com/iafrotamacedo-cloud/era/faces/tensor"
)

// Os modelos destes testes sao montados como structs Go, e nao serializados
// em protobuf. A leitura do arquivo ja tem os seus proprios testes no pacote
// onnx; aqui o que esta sob teste e a execucao.

func no(opType, nome string, ins, outs []string, attrs ...*onnx.Attribute) *onnx.Node {
	return &onnx.Node{OpType: opType, Name: nome, Inputs: ins, Outputs: outs, Attributes: attrs}
}

func aInts(nome string, vals ...int64) *onnx.Attribute {
	return &onnx.Attribute{Name: nome, Type: onnx.AttrInts, Ints: vals}
}

func aInt(nome string, v int64) *onnx.Attribute {
	return &onnx.Attribute{Name: nome, Type: onnx.AttrInt, I: v}
}

func aFloat(nome string, v float32) *onnx.Attribute {
	return &onnx.Attribute{Name: nome, Type: onnx.AttrFloat, F: v}
}

func aString(nome, v string) *onnx.Attribute {
	return &onnx.Attribute{Name: nome, Type: onnx.AttrString, S: []byte(v)}
}

func peso(nome string, dims []int64, vals []float32) *onnx.Tensor {
	return &onnx.Tensor{Name: nome, DataType: onnx.Float, Dims: dims, FloatData: vals}
}

func modelo(g *onnx.Graph) *onnx.Model {
	return &onnx.Model{IRVersion: 8, OpsetImports: []onnx.OpsetID{{Version: 13}}, Graph: g}
}

func randSlice(r *rand.Rand, n int) []float32 {
	s := make([]float32, n)
	for i := range s {
		s[i] = r.Float32()*2 - 1
	}
	return s
}

func positivos(r *rand.Rand, n int) []float32 {
	s := make([]float32, n)
	for i := range s {
		s[i] = r.Float32() + 0.1
	}
	return s
}

func closeEnough(got, want float32) bool {
	d := got - want
	if d < 0 {
		d = -d
	}
	m := want
	if m < 0 {
		m = -m
	}
	return d <= 1e-4+1e-4*m
}

// rodar executa um grafo com uma entrada so e devolve a saida achatada.
func rodar(t testing.TB, m *onnx.Model, entrada *tensor.Tensor) []float32 {
	t.Helper()

	g, err := New(m)
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	ws := nn.NewWorkspace()
	nome := g.Inputs()[0]
	outs, err := g.Run(ws, map[string]*tensor.Tensor{nome: entrada})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}

	saida := outs[g.Outputs()[0]]
	if saida == nil {
		t.Fatal("o grafo nao produziu a saida declarada")
	}
	return append([]float32(nil), saida.Flat()...)
}

// grafoSimples monta um grafo de uma operacao so, com a entrada "x" e a
// saida "y".
func grafoSimples(n *onnx.Node, inits ...*onnx.Tensor) *onnx.Model {
	return modelo(&onnx.Graph{
		Name:         "simples",
		Nodes:        []*onnx.Node{n},
		Initializers: inits,
		Inputs:       []*onnx.ValueInfo{{Name: "x"}},
		Outputs:      []*onnx.ValueInfo{{Name: "y"}},
	})
}

// ---------- ordenacao topologica ----------

// TestOrdenaNosForaDeOrdem cobre a razao de o pacote nao confiar na ordem do
// arquivo. A especificacao exige ordem topologica, mas um exportador que a
// desrespeite produziria um erro no meio da rede, sem pista da causa.
func TestOrdenaNosForaDeOrdem(t *testing.T) {
	// Gravado de tras para a frente: o Relu vem antes do Conv que o alimenta.
	m := modelo(&onnx.Graph{
		Name: "invertido",
		Nodes: []*onnx.Node{
			no("Relu", "relu", []string{"meio"}, []string{"y"}),
			no("Conv", "conv", []string{"x", "w"}, []string{"meio"},
				aInts("kernel_shape", 1, 1)),
		},
		Initializers: []*onnx.Tensor{peso("w", []int64{1, 1, 1, 1}, []float32{2})},
		Inputs:       []*onnx.ValueInfo{{Name: "x"}},
		Outputs:      []*onnx.ValueInfo{{Name: "y"}},
	})

	x := tensor.MustFromSlice([]float32{1, -3}, 1, 1, 1, 2)
	got := rodar(t, m, x)

	// conv multiplica por 2 -> [2, -6]; relu zera o negativo -> [2, 0]
	if want := []float32{2, 0}; !reflect.DeepEqual(got, want) {
		t.Errorf("saida = %v, quero %v", got, want)
	}
}

func TestDetectaCiclo(t *testing.T) {
	m := modelo(&onnx.Graph{
		Nodes: []*onnx.Node{
			no("Relu", "a", []string{"y"}, []string{"meio"}),
			no("Relu", "b", []string{"meio"}, []string{"y"}),
		},
		Outputs: []*onnx.ValueInfo{{Name: "y"}},
	})

	_, err := New(m)
	if err == nil {
		t.Fatal("grafo com ciclo deveria dar erro")
	}
	if !strings.Contains(err.Error(), "ciclo") {
		t.Errorf("a mensagem deveria falar em ciclo: %v", err)
	}
}

func TestDetectaValorSemProdutor(t *testing.T) {
	m := grafoSimples(no("Relu", "r", []string{"nao_existe"}, []string{"y"}))
	_, err := New(m)
	if err == nil {
		t.Fatal("valor sem produtor deveria dar erro")
	}
	if !strings.Contains(err.Error(), "nao_existe") {
		t.Errorf("a mensagem deveria citar o valor: %v", err)
	}
}

func TestDetectaProdutorDuplicado(t *testing.T) {
	m := modelo(&onnx.Graph{
		Nodes: []*onnx.Node{
			no("Relu", "a", []string{"x"}, []string{"y"}),
			no("Relu", "b", []string{"x"}, []string{"y"}),
		},
		Inputs:  []*onnx.ValueInfo{{Name: "x"}},
		Outputs: []*onnx.ValueInfo{{Name: "y"}},
	})

	if _, err := New(m); err == nil {
		t.Fatal("dois nos produzindo o mesmo valor deveria dar erro")
	}
}

func TestOperadorNaoImplementado(t *testing.T) {
	m := modelo(&onnx.Graph{
		Nodes: []*onnx.Node{
			no("OperadorInventado", "a", []string{"x"}, []string{"m"}),
			no("OutroInventado", "b", []string{"m"}, []string{"y"}),
			no("OperadorInventado", "c", []string{"y"}, []string{"z"}),
		},
		Inputs:  []*onnx.ValueInfo{{Name: "x"}},
		Outputs: []*onnx.ValueInfo{{Name: "z"}},
	})

	_, err := New(m)
	if err == nil {
		t.Fatal("operador desconhecido deveria dar erro")
	}

	// Todos de uma vez, sem repetir: carregar um modelo novo nao pode virar
	// uma sequencia de tentativas.
	msg := err.Error()
	for _, esperado := range []string{"OperadorInventado", "OutroInventado"} {
		if !strings.Contains(msg, esperado) {
			t.Errorf("a mensagem deveria citar %s: %v", esperado, err)
		}
	}
	if strings.Count(msg, "OperadorInventado") != 1 {
		t.Errorf("o operador repetido deveria aparecer uma vez so: %v", err)
	}
}

// ---------- execucao ----------

func TestEntradaFaltando(t *testing.T) {
	m := grafoSimples(no("Relu", "r", []string{"x"}, []string{"y"}))
	g, err := New(m)
	if err != nil {
		t.Fatal(err)
	}

	if _, err := g.Run(nn.NewWorkspace(), nil); err == nil {
		t.Error("rodar sem a entrada deveria dar erro")
	}
	if _, err := g.Run(nil, map[string]*tensor.Tensor{"x": tensor.New(1, 1)}); err == nil {
		t.Error("workspace nulo deveria dar erro")
	}
}

// TestInitializerNaoEntradaObrigatoria: muitos exportadores listam os pesos
// tambem como inputs do grafo, por compatibilidade com versoes antigas do
// formato. Exigi-los de quem chama tornaria todo modelo real inutilizavel.
func TestInitializerNaoEntradaObrigatoria(t *testing.T) {
	m := modelo(&onnx.Graph{
		Nodes: []*onnx.Node{
			no("Conv", "conv", []string{"x", "w"}, []string{"y"}, aInts("kernel_shape", 1, 1)),
		},
		Initializers: []*onnx.Tensor{peso("w", []int64{1, 1, 1, 1}, []float32{3})},
		Inputs: []*onnx.ValueInfo{
			{Name: "x"},
			{Name: "w"}, // o peso, listado como entrada
		},
		Outputs: []*onnx.ValueInfo{{Name: "y"}},
	})

	g, err := New(m)
	if err != nil {
		t.Fatal(err)
	}
	if want := []string{"x"}; !reflect.DeepEqual(g.Inputs(), want) {
		t.Errorf("Inputs = %v, quero %v (o peso nao deveria ser exigido)", g.Inputs(), want)
	}
}

func TestConvComVies(t *testing.T) {
	m := grafoSimples(
		no("Conv", "conv", []string{"x", "w", "b"}, []string{"y"},
			aInts("kernel_shape", 1, 1)),
		peso("w", []int64{1, 1, 1, 1}, []float32{2}),
		peso("b", []int64{1}, []float32{10}),
	)

	got := rodar(t, m, tensor.MustFromSlice([]float32{1, 2, 3}, 1, 1, 1, 3))
	if want := []float32{12, 14, 16}; !reflect.DeepEqual(got, want) {
		t.Errorf("saida = %v, quero %v", got, want)
	}
}

func TestConvDepthwise(t *testing.T) {
	// group = canais: cada canal com o seu filtro, sem mistura.
	m := grafoSimples(
		no("Conv", "dw", []string{"x", "w"}, []string{"y"},
			aInts("kernel_shape", 1, 1), aInt("group", 2)),
		peso("w", []int64{2, 1, 1, 1}, []float32{2, 10}),
	)

	got := rodar(t, m, tensor.MustFromSlice([]float32{1, 2, 3, 4}, 1, 2, 1, 2))
	if want := []float32{2, 4, 30, 40}; !reflect.DeepEqual(got, want) {
		t.Errorf("saida = %v, quero %v", got, want)
	}
}

func TestConvRecusaOQueNaoSuporta(t *testing.T) {
	casos := []struct {
		nome   string
		attrs  []*onnx.Attribute
		trecho string
	}{
		{"padding assimetrico", []*onnx.Attribute{
			aInts("kernel_shape", 3, 3), aInts("pads", 1, 1, 0, 0)}, "assimetrico"},
		{"auto_pad SAME_UPPER", []*onnx.Attribute{
			aInts("kernel_shape", 3, 3), aString("auto_pad", "SAME_UPPER")}, "auto_pad"},
		{"kernel_shape divergente", []*onnx.Attribute{
			aInts("kernel_shape", 5, 5)}, "nao bate"},
	}

	for _, c := range casos {
		t.Run(c.nome, func(t *testing.T) {
			m := grafoSimples(
				no("Conv", "conv", []string{"x", "w"}, []string{"y"}, c.attrs...),
				peso("w", []int64{1, 1, 3, 3}, make([]float32, 9)),
			)
			_, err := New(m)
			if err == nil {
				t.Fatal("deveria dar erro")
			}
			if !strings.Contains(err.Error(), c.trecho) {
				t.Errorf("a mensagem deveria conter %q: %v", c.trecho, err)
			}
		})
	}
}

func TestBatchNormalization(t *testing.T) {
	// gamma=2, beta=1, mean=5, var=4, eps=0 -> escala 1, deslocamento -4
	m := grafoSimples(
		no("BatchNormalization", "bn", []string{"x", "g", "b", "m", "v"}, []string{"y"},
			aFloat("epsilon", 0)),
		peso("g", []int64{1}, []float32{2}),
		peso("b", []int64{1}, []float32{1}),
		peso("m", []int64{1}, []float32{5}),
		peso("v", []int64{1}, []float32{4}),
	)

	got := rodar(t, m, tensor.MustFromSlice([]float32{5, 7, 3}, 1, 1, 1, 3))
	if want := []float32{1, 3, -1}; !reflect.DeepEqual(got, want) {
		t.Errorf("saida = %v, quero %v", got, want)
	}
}

// TestBatchNormComSaidasDeTreinoDaErro: BatchNormalization devolve as
// estatisticas do lote quando exportado em modo de treino. Ignorar isso e
// executar so a primeira saida rodaria, e daria resultado errado.
func TestBatchNormComSaidasDeTreinoDaErro(t *testing.T) {
	m := modelo(&onnx.Graph{
		Nodes: []*onnx.Node{
			no("BatchNormalization", "bn", []string{"x", "g", "b", "m", "v"},
				[]string{"y", "media_do_lote", "var_do_lote"}),
		},
		Initializers: []*onnx.Tensor{
			peso("g", []int64{1}, []float32{1}), peso("b", []int64{1}, []float32{0}),
			peso("m", []int64{1}, []float32{0}), peso("v", []int64{1}, []float32{1}),
		},
		Inputs:  []*onnx.ValueInfo{{Name: "x"}},
		Outputs: []*onnx.ValueInfo{{Name: "y"}},
	})

	if _, err := New(m); err == nil {
		t.Error("BatchNormalization em modo de treino deveria dar erro")
	}
}

func TestPRelu(t *testing.T) {
	m := grafoSimples(
		no("PRelu", "prelu", []string{"x", "s"}, []string{"y"}),
		peso("s", []int64{2, 1, 1}, []float32{0.25, 0.5}), // forma [C,1,1]
	)

	got := rodar(t, m, tensor.MustFromSlice([]float32{2, -4, 2, -4}, 1, 2, 1, 2))
	if want := []float32{2, -1, 2, -2}; !reflect.DeepEqual(got, want) {
		t.Errorf("saida = %v, quero %v", got, want)
	}
}

func TestAtivacoes(t *testing.T) {
	entrada := []float32{-2, -0.5, 0, 0.5, 2}

	casos := []struct {
		op    string
		attrs []*onnx.Attribute
		f     func(float32) float32
	}{
		{"Relu", nil, func(v float32) float32 {
			if v > 0 {
				return v
			}
			return 0
		}},
		{"LeakyRelu", []*onnx.Attribute{aFloat("alpha", 0.1)}, func(v float32) float32 {
			if v > 0 {
				return v
			}
			return 0.1 * v
		}},
		{"Sigmoid", nil, func(v float32) float32 {
			return float32(1 / (1 + math.Exp(-float64(v))))
		}},
		{"Tanh", nil, func(v float32) float32 {
			return float32(math.Tanh(float64(v)))
		}},
	}

	for _, c := range casos {
		t.Run(c.op, func(t *testing.T) {
			m := grafoSimples(no(c.op, "a", []string{"x"}, []string{"y"}, c.attrs...))
			got := rodar(t, m, tensor.MustFromSlice(append([]float32(nil), entrada...), 1, 5))

			for i, v := range entrada {
				if want := c.f(v); !closeEnough(got[i], want) {
					t.Errorf("%s(%v) = %v, quero %v", c.op, v, got[i], want)
				}
			}
		})
	}
}

// TestClipDosDoisJeitos: ate o opset 10 os limites eram atributos; a partir
// do 11 viraram entradas. Os dois formatos circulam.
func TestClipDosDoisJeitos(t *testing.T) {
	entrada := []float32{-5, 0, 3, 10}
	quero := []float32{0, 0, 3, 6}

	t.Run("por atributo", func(t *testing.T) {
		m := grafoSimples(no("Clip", "c", []string{"x"}, []string{"y"},
			aFloat("min", 0), aFloat("max", 6)))
		got := rodar(t, m, tensor.MustFromSlice(append([]float32(nil), entrada...), 1, 4))
		if !reflect.DeepEqual(got, quero) {
			t.Errorf("saida = %v, quero %v", got, quero)
		}
	})

	t.Run("por entrada", func(t *testing.T) {
		m := grafoSimples(
			no("Clip", "c", []string{"x", "lo", "hi"}, []string{"y"}),
			peso("lo", []int64{}, []float32{0}),
			peso("hi", []int64{}, []float32{6}),
		)
		got := rodar(t, m, tensor.MustFromSlice(append([]float32(nil), entrada...), 1, 4))
		if !reflect.DeepEqual(got, quero) {
			t.Errorf("saida = %v, quero %v", got, quero)
		}
	})
}

func TestSoftmax(t *testing.T) {
	m := grafoSimples(no("Softmax", "s", []string{"x"}, []string{"y"}, aInt("axis", -1)))
	got := rodar(t, m, tensor.MustFromSlice([]float32{1, 2, 3}, 1, 3))

	var soma float32
	for _, v := range got {
		soma += v
	}
	if !closeEnough(soma, 1) {
		t.Errorf("a soma deu %v, quero 1", soma)
	}
	if !(got[0] < got[1] && got[1] < got[2]) {
		t.Errorf("softmax deveria preservar a ordem: %v", got)
	}
}

// TestSoftmaxNaoEstoura: sem subtrair o maior antes de exponenciar, um valor
// alto vira +Inf e a soma inteira vira NaN.
func TestSoftmaxNaoEstoura(t *testing.T) {
	m := grafoSimples(no("Softmax", "s", []string{"x"}, []string{"y"}, aInt("axis", -1)))
	got := rodar(t, m, tensor.MustFromSlice([]float32{1000, 1001, 1002}, 1, 3))

	for i, v := range got {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("saida[%d] = %v com entradas grandes", i, v)
		}
	}
	var soma float32
	for _, v := range got {
		soma += v
	}
	if !closeEnough(soma, 1) {
		t.Errorf("a soma deu %v, quero 1", soma)
	}
}

func TestSoftmaxNoEixoDoMeio(t *testing.T) {
	// Eixo 1 de [1,2,2]: cada par ao longo do canal precisa somar 1.
	m := grafoSimples(no("Softmax", "s", []string{"x"}, []string{"y"}, aInt("axis", 1)))
	got := rodar(t, m, tensor.MustFromSlice([]float32{1, 2, 3, 4}, 1, 2, 2))

	for i := 0; i < 2; i++ {
		soma := got[i] + got[i+2]
		if !closeEnough(soma, 1) {
			t.Errorf("posicao %d: soma no eixo = %v, quero 1", i, soma)
		}
	}
}

// ---------- aritmetica e transmissao de forma ----------

func TestAritmetica(t *testing.T) {
	casos := []struct {
		op    string
		quero []float32
	}{
		{"Add", []float32{11, 22}},
		{"Sub", []float32{-9, -18}},
		{"Mul", []float32{10, 40}},
		{"Div", []float32{0.1, 0.1}},
	}

	for _, c := range casos {
		t.Run(c.op, func(t *testing.T) {
			m := grafoSimples(
				no(c.op, "a", []string{"x", "k"}, []string{"y"}),
				peso("k", []int64{1, 2}, []float32{10, 20}),
			)
			got := rodar(t, m, tensor.MustFromSlice([]float32{1, 2}, 1, 2))
			for i := range c.quero {
				if !closeEnough(got[i], c.quero[i]) {
					t.Errorf("%s: saida = %v, quero %v", c.op, got, c.quero)
					break
				}
			}
		})
	}
}

// TestTransmissaoDeForma: escalonamento por canal aparece como
// [N,C,H,W] vezes [C,1,1]. Recusar isso quebraria modelos reais.
func TestTransmissaoDeForma(t *testing.T) {
	casos := []struct {
		nome  string
		dims  []int64
		vals  []float32
		quero []float32
	}{
		{"escalar", []int64{1}, []float32{10}, []float32{10, 20, 30, 40}},
		{"por canal [C,1,1]", []int64{2, 1, 1}, []float32{10, 100}, []float32{10, 20, 300, 400}},
		{"por coluna [1,1,2]", []int64{1, 1, 2}, []float32{10, 100}, []float32{10, 200, 30, 400}},
	}

	for _, c := range casos {
		t.Run(c.nome, func(t *testing.T) {
			m := grafoSimples(
				no("Mul", "m", []string{"x", "k"}, []string{"y"}),
				peso("k", c.dims, c.vals),
			)
			got := rodar(t, m, tensor.MustFromSlice([]float32{1, 2, 3, 4}, 1, 2, 1, 2))
			if !reflect.DeepEqual(got, c.quero) {
				t.Errorf("saida = %v, quero %v", got, c.quero)
			}
		})
	}
}

func TestTransmissaoIncompativelDaErro(t *testing.T) {
	m := grafoSimples(
		no("Add", "a", []string{"x", "k"}, []string{"y"}),
		peso("k", []int64{3}, []float32{1, 2, 3}),
	)

	g, err := New(m)
	if err != nil {
		t.Fatal(err)
	}
	_, err = g.Run(nn.NewWorkspace(), map[string]*tensor.Tensor{
		"x": tensor.MustFromSlice([]float32{1, 2}, 1, 2),
	})
	if err == nil {
		t.Error("formas incompativeis deveriam dar erro")
	}
}

// ---------- forma ----------

func TestFlattenEReshape(t *testing.T) {
	t.Run("Flatten eixo padrao", func(t *testing.T) {
		m := grafoSimples(no("Flatten", "f", []string{"x"}, []string{"y"}))
		g, _ := New(m)
		outs, err := g.Run(nn.NewWorkspace(), map[string]*tensor.Tensor{
			"x": tensor.MustFromSlice([]float32{1, 2, 3, 4, 5, 6, 7, 8}, 2, 2, 1, 2),
		})
		if err != nil {
			t.Fatal(err)
		}
		if want := []int{2, 4}; !reflect.DeepEqual(outs["y"].Shape, want) {
			t.Errorf("forma = %v, quero %v", outs["y"].Shape, want)
		}
	})

	t.Run("Reshape com -1", func(t *testing.T) {
		m := grafoSimples(
			no("Reshape", "r", []string{"x", "forma"}, []string{"y"}),
			peso("forma", []int64{2}, []float32{2, -1}),
		)
		g, _ := New(m)
		outs, err := g.Run(nn.NewWorkspace(), map[string]*tensor.Tensor{
			"x": tensor.MustFromSlice([]float32{1, 2, 3, 4, 5, 6}, 1, 6),
		})
		if err != nil {
			t.Fatal(err)
		}
		if want := []int{2, 3}; !reflect.DeepEqual(outs["y"].Shape, want) {
			t.Errorf("forma = %v, quero %v", outs["y"].Shape, want)
		}
	})

	t.Run("Reshape com 0 copia a dimensao", func(t *testing.T) {
		m := grafoSimples(
			no("Reshape", "r", []string{"x", "forma"}, []string{"y"}),
			peso("forma", []int64{2}, []float32{0, -1}),
		)
		g, _ := New(m)
		outs, err := g.Run(nn.NewWorkspace(), map[string]*tensor.Tensor{
			"x": tensor.MustFromSlice([]float32{1, 2, 3, 4, 5, 6}, 3, 2),
		})
		if err != nil {
			t.Fatal(err)
		}
		if want := []int{3, 2}; !reflect.DeepEqual(outs["y"].Shape, want) {
			t.Errorf("forma = %v, quero %v (0 deveria copiar a dimensao da entrada)",
				outs["y"].Shape, want)
		}
	})
}

func TestTranspose(t *testing.T) {
	m := grafoSimples(no("Transpose", "t", []string{"x"}, []string{"y"}, aInts("perm", 0, 2, 1)))

	g, _ := New(m)
	outs, err := g.Run(nn.NewWorkspace(), map[string]*tensor.Tensor{
		"x": tensor.MustFromSlice([]float32{1, 2, 3, 4, 5, 6}, 1, 2, 3),
	})
	if err != nil {
		t.Fatal(err)
	}

	if want := []int{1, 3, 2}; !reflect.DeepEqual(outs["y"].Shape, want) {
		t.Errorf("forma = %v, quero %v", outs["y"].Shape, want)
	}
	if want := []float32{1, 4, 2, 5, 3, 6}; !reflect.DeepEqual(outs["y"].Flat(), want) {
		t.Errorf("dados = %v, quero %v", outs["y"].Flat(), want)
	}
}

func TestConcat(t *testing.T) {
	m := modelo(&onnx.Graph{
		Nodes: []*onnx.Node{
			no("Concat", "c", []string{"x", "k"}, []string{"y"}, aInt("axis", 1)),
		},
		Initializers: []*onnx.Tensor{peso("k", []int64{1, 1, 2}, []float32{9, 9})},
		Inputs:       []*onnx.ValueInfo{{Name: "x"}},
		Outputs:      []*onnx.ValueInfo{{Name: "y"}},
	})

	g, _ := New(m)
	outs, err := g.Run(nn.NewWorkspace(), map[string]*tensor.Tensor{
		"x": tensor.MustFromSlice([]float32{1, 2, 3, 4}, 1, 2, 2),
	})
	if err != nil {
		t.Fatal(err)
	}

	if want := []int{1, 3, 2}; !reflect.DeepEqual(outs["y"].Shape, want) {
		t.Errorf("forma = %v, quero %v", outs["y"].Shape, want)
	}
	if want := []float32{1, 2, 3, 4, 9, 9}; !reflect.DeepEqual(outs["y"].Flat(), want) {
		t.Errorf("dados = %v, quero %v", outs["y"].Flat(), want)
	}
}

func TestSqueezeUnsqueeze(t *testing.T) {
	t.Run("Unsqueeze", func(t *testing.T) {
		m := grafoSimples(no("Unsqueeze", "u", []string{"x"}, []string{"y"}, aInts("axes", 0, 3)))
		g, _ := New(m)
		outs, _ := g.Run(nn.NewWorkspace(), map[string]*tensor.Tensor{
			"x": tensor.MustFromSlice([]float32{1, 2, 3, 4, 5, 6}, 2, 3),
		})
		if want := []int{1, 2, 3, 1}; !reflect.DeepEqual(outs["y"].Shape, want) {
			t.Errorf("forma = %v, quero %v", outs["y"].Shape, want)
		}
	})

	t.Run("Squeeze com eixos", func(t *testing.T) {
		m := grafoSimples(no("Squeeze", "s", []string{"x"}, []string{"y"}, aInts("axes", 0)))
		g, _ := New(m)
		outs, _ := g.Run(nn.NewWorkspace(), map[string]*tensor.Tensor{
			"x": tensor.MustFromSlice([]float32{1, 2}, 1, 2),
		})
		if want := []int{2}; !reflect.DeepEqual(outs["y"].Shape, want) {
			t.Errorf("forma = %v, quero %v", outs["y"].Shape, want)
		}
	})

	t.Run("Squeeze de eixo que nao vale 1", func(t *testing.T) {
		m := grafoSimples(no("Squeeze", "s", []string{"x"}, []string{"y"}, aInts("axes", 1)))
		g, _ := New(m)
		_, err := g.Run(nn.NewWorkspace(), map[string]*tensor.Tensor{
			"x": tensor.MustFromSlice([]float32{1, 2}, 1, 2),
		})
		if err == nil {
			t.Error("Squeeze de dimensao maior que 1 deveria dar erro")
		}
	})
}

func TestIdentityEDropout(t *testing.T) {
	for _, op := range []string{"Identity", "Dropout"} {
		t.Run(op, func(t *testing.T) {
			m := grafoSimples(no(op, "i", []string{"x"}, []string{"y"}))
			got := rodar(t, m, tensor.MustFromSlice([]float32{1, -2, 3}, 1, 3))
			if want := []float32{1, -2, 3}; !reflect.DeepEqual(got, want) {
				t.Errorf("saida = %v, quero %v", got, want)
			}
		})
	}
}

func TestConstant(t *testing.T) {
	m := modelo(&onnx.Graph{
		Nodes: []*onnx.Node{
			no("Constant", "c", nil, []string{"k"},
				&onnx.Attribute{Name: "value", Type: onnx.AttrTensor,
					T: peso("", []int64{1, 2}, []float32{5, 6})}),
			no("Add", "a", []string{"x", "k"}, []string{"y"}),
		},
		Inputs:  []*onnx.ValueInfo{{Name: "x"}},
		Outputs: []*onnx.ValueInfo{{Name: "y"}},
	})

	got := rodar(t, m, tensor.MustFromSlice([]float32{1, 2}, 1, 2))
	if want := []float32{6, 8}; !reflect.DeepEqual(got, want) {
		t.Errorf("saida = %v, quero %v", got, want)
	}
}

// ---------- pooling ----------

func TestGlobalAveragePool(t *testing.T) {
	m := grafoSimples(no("GlobalAveragePool", "p", []string{"x"}, []string{"y"}))
	got := rodar(t, m, tensor.MustFromSlice([]float32{1, 2, 3, 4, 10, 20, 30, 40}, 1, 2, 2, 2))
	if want := []float32{2.5, 25}; !reflect.DeepEqual(got, want) {
		t.Errorf("saida = %v, quero %v", got, want)
	}
}

func TestMaxPool(t *testing.T) {
	m := grafoSimples(no("MaxPool", "p", []string{"x"}, []string{"y"},
		aInts("kernel_shape", 2, 2)))

	got := rodar(t, m, tensor.MustFromSlice([]float32{
		1, 2, 3, 4,
		5, 6, 7, 8,
		9, 10, 11, 12,
		13, 14, 15, 16,
	}, 1, 1, 4, 4))

	if want := []float32{6, 8, 14, 16}; !reflect.DeepEqual(got, want) {
		t.Errorf("saida = %v, quero %v", got, want)
	}
}

// TestAveragePoolContagemDoPadding cobre count_include_pad, que decide se as
// posicoes do padding entram no divisor. O padrao do ONNX e nao entrarem.
func TestAveragePoolContagemDoPadding(t *testing.T) {
	entrada := []float32{1, 2, 3, 4}

	t.Run("padrao: media so dos pixels reais", func(t *testing.T) {
		m := grafoSimples(no("AveragePool", "p", []string{"x"}, []string{"y"},
			aInts("kernel_shape", 2, 2), aInts("strides", 2, 2), aInts("pads", 1, 1, 1, 1)))
		got := rodar(t, m, tensor.MustFromSlice(append([]float32(nil), entrada...), 1, 1, 2, 2))
		// O canto superior esquerdo ve so o pixel 1, e 3 posicoes de padding.
		if !closeEnough(got[0], 1) {
			t.Errorf("canto = %v, quero 1 (media de um pixel so)", got[0])
		}
	})

	t.Run("count_include_pad=1", func(t *testing.T) {
		m := grafoSimples(no("AveragePool", "p", []string{"x"}, []string{"y"},
			aInts("kernel_shape", 2, 2), aInts("strides", 2, 2), aInts("pads", 1, 1, 1, 1),
			aInt("count_include_pad", 1)))
		got := rodar(t, m, tensor.MustFromSlice(append([]float32(nil), entrada...), 1, 1, 2, 2))
		// Agora divide por 4: 1/4 = 0.25
		if !closeEnough(got[0], 0.25) {
			t.Errorf("canto = %v, quero 0.25 (divisor incluindo o padding)", got[0])
		}
	})
}

// ---------- camada densa ----------

func TestGemm(t *testing.T) {
	// transB=1: pesos em [saidas, entradas], que e o formato usual.
	m := grafoSimples(
		no("Gemm", "fc", []string{"x", "w", "b"}, []string{"y"}, aInt("transB", 1)),
		peso("w", []int64{2, 3}, []float32{1, 0, 0, 0, 1, 0}),
		peso("b", []int64{2}, []float32{10, 20}),
	)

	got := rodar(t, m, tensor.MustFromSlice([]float32{1, 2, 3}, 1, 3))
	if want := []float32{11, 22}; !reflect.DeepEqual(got, want) {
		t.Errorf("saida = %v, quero %v", got, want)
	}
}

func TestGemmSemTransB(t *testing.T) {
	// transB=0: pesos em [entradas, saidas]. Mesmo resultado do teste acima.
	m := grafoSimples(
		no("Gemm", "fc", []string{"x", "w", "b"}, []string{"y"}),
		peso("w", []int64{3, 2}, []float32{1, 0, 0, 1, 0, 0}),
		peso("b", []int64{2}, []float32{10, 20}),
	)

	got := rodar(t, m, tensor.MustFromSlice([]float32{1, 2, 3}, 1, 3))
	if want := []float32{11, 22}; !reflect.DeepEqual(got, want) {
		t.Errorf("saida = %v, quero %v", got, want)
	}
}

func TestGemmRecusaOQueNaoSuporta(t *testing.T) {
	casos := []struct {
		nome string
		attr *onnx.Attribute
	}{
		{"alpha", aFloat("alpha", 2)},
		{"beta", aFloat("beta", 0.5)},
		{"transA", aInt("transA", 1)},
	}

	for _, c := range casos {
		t.Run(c.nome, func(t *testing.T) {
			m := grafoSimples(
				no("Gemm", "fc", []string{"x", "w"}, []string{"y"}, c.attr),
				peso("w", []int64{2, 3}, make([]float32, 6)),
			)
			if _, err := New(m); err == nil {
				t.Errorf("Gemm com %s incomum deveria dar erro em vez de calcular errado", c.nome)
			}
		})
	}
}

func TestMatMul(t *testing.T) {
	m := grafoSimples(
		no("MatMul", "mm", []string{"x", "w"}, []string{"y"}),
		peso("w", []int64{3, 2}, []float32{1, 0, 0, 1, 0, 0}), // [K,M]
	)

	got := rodar(t, m, tensor.MustFromSlice([]float32{1, 2, 3}, 1, 3))
	if want := []float32{1, 2}; !reflect.DeepEqual(got, want) {
		t.Errorf("saida = %v, quero %v", got, want)
	}
}

// ---------- teste de ponta a ponta ----------

// TestGrafoBateComRedeMontadaAMao e o teste que fecha a Fase 3: a mesma rede,
// com os mesmos pesos, montada de duas formas -- pelo grafo ONNX e a mao com
// o pacote nn -- precisa dar o mesmo resultado.
//
// Se bater, o executor esta traduzindo os operadores corretamente. Se nao
// bater, o erro esta na traducao e nao nas camadas, que ja tem os seus
// proprios testes.
func TestGrafoBateComRedeMontadaAMao(t *testing.T) {
	r := rand.New(rand.NewSource(20260908))

	const inC, outC, canais = 3, 8, 8
	pesosConv := randSlice(r, outC*inC*9)
	gama := randSlice(r, canais)
	beta := randSlice(r, canais)
	media := randSlice(r, canais)
	variancia := positivos(r, canais)
	alfa := randSlice(r, canais)
	pesosFC := randSlice(r, 16*canais)
	viesFC := randSlice(r, 16)
	entrada := randSlice(r, 1*inC*16*16)

	// --- caminho 1: pelo grafo ONNX ---
	m := modelo(&onnx.Graph{
		Name: "mobileface-ish",
		Nodes: []*onnx.Node{
			no("Conv", "conv1", []string{"x", "w"}, []string{"c1"},
				aInts("kernel_shape", 3, 3), aInts("pads", 1, 1, 1, 1), aInts("strides", 2, 2)),
			no("BatchNormalization", "bn1", []string{"c1", "g", "b", "m", "v"}, []string{"b1"},
				aFloat("epsilon", 1e-5)),
			no("PRelu", "prelu1", []string{"b1", "s"}, []string{"p1"}),
			no("GlobalAveragePool", "pool", []string{"p1"}, []string{"g1"}),
			no("Flatten", "flat", []string{"g1"}, []string{"f1"}, aInt("axis", 1)),
			no("Gemm", "fc", []string{"f1", "wfc", "bfc"}, []string{"y"}, aInt("transB", 1)),
		},
		Initializers: []*onnx.Tensor{
			peso("w", []int64{outC, inC, 3, 3}, pesosConv),
			peso("g", []int64{canais}, gama),
			peso("b", []int64{canais}, beta),
			peso("m", []int64{canais}, media),
			peso("v", []int64{canais}, variancia),
			peso("s", []int64{canais}, alfa),
			peso("wfc", []int64{16, canais}, pesosFC),
			peso("bfc", []int64{16}, viesFC),
		},
		Inputs:  []*onnx.ValueInfo{{Name: "x"}},
		Outputs: []*onnx.ValueInfo{{Name: "y"}},
	})

	x := tensor.MustFromSlice(append([]float32(nil), entrada...), 1, inC, 16, 16)
	viaGrafo := rodar(t, m, x)

	// --- caminho 2: a mao, com o pacote nn ---
	conv, err := nn.NewConv2D("conv1", nn.Conv2DConfig{
		InC: inC, OutC: outC, KH: 3, KW: 3, PadH: 1, PadW: 1, StrideH: 2, StrideW: 2,
	}, append([]float32(nil), pesosConv...), nil)
	if err != nil {
		t.Fatal(err)
	}
	bn, err := nn.NewBatchNorm("bn1", gama, beta, media, variancia, 1e-5)
	if err != nil {
		t.Fatal(err)
	}
	prelu, err := nn.NewPReLU("prelu1", alfa)
	if err != nil {
		t.Fatal(err)
	}
	fc, err := nn.NewLinear("fc", canais, 16, append([]float32(nil), pesosFC...), viesFC)
	if err != nil {
		t.Fatal(err)
	}

	rede := nn.NewSequential("mao", conv, bn, prelu, &nn.GlobalAvgPool{}, &nn.Flatten{}, fc)
	saida, err := rede.Forward(nn.NewWorkspace(),
		tensor.MustFromSlice(append([]float32(nil), entrada...), 1, inC, 16, 16))
	if err != nil {
		t.Fatal(err)
	}
	aMao := saida.Flat()

	if len(viaGrafo) != len(aMao) {
		t.Fatalf("o grafo deu %d valores, a rede a mao deu %d", len(viaGrafo), len(aMao))
	}
	for i := range aMao {
		if !closeEnough(viaGrafo[i], aMao[i]) {
			t.Fatalf("saida[%d]: grafo deu %v, rede a mao deu %v", i, viaGrafo[i], aMao[i])
		}
	}
}

// TestConexaoResidual: o Add que fecha um atalho recebe dois valores
// produzidos por nos diferentes -- o caso que Sequential nao cobre e que
// motivou o executor de grafo.
func TestConexaoResidual(t *testing.T) {
	m := modelo(&onnx.Graph{
		Name: "residual",
		Nodes: []*onnx.Node{
			no("Conv", "conv", []string{"x", "w"}, []string{"ramo"}, aInts("kernel_shape", 1, 1)),
			no("Relu", "relu", []string{"ramo"}, []string{"ativo"}),
			no("Add", "soma", []string{"ativo", "x"}, []string{"y"}), // atalho
		},
		Initializers: []*onnx.Tensor{peso("w", []int64{1, 1, 1, 1}, []float32{2})},
		Inputs:       []*onnx.ValueInfo{{Name: "x"}},
		Outputs:      []*onnx.ValueInfo{{Name: "y"}},
	})

	got := rodar(t, m, tensor.MustFromSlice([]float32{1, -3}, 1, 1, 1, 2))
	// conv: [2,-6]; relu: [2,0]; add com x: [3,-3]
	if want := []float32{3, -3}; !reflect.DeepEqual(got, want) {
		t.Errorf("saida = %v, quero %v", got, want)
	}
}

// TestWorkspaceReutilizadoEntrePassagens confere que rodar o mesmo grafo
// varias vezes nao acumula memoria nem muda o resultado.
func TestWorkspaceReutilizadoEntrePassagens(t *testing.T) {
	r := rand.New(rand.NewSource(5))
	m := grafoSimples(
		no("Conv", "conv", []string{"x", "w"}, []string{"y"},
			aInts("kernel_shape", 3, 3), aInts("pads", 1, 1, 1, 1)),
		peso("w", []int64{4, 2, 3, 3}, randSlice(r, 4*2*9)),
	)

	g, err := New(m)
	if err != nil {
		t.Fatal(err)
	}

	x := tensor.MustFromSlice(randSlice(r, 1*2*8*8), 1, 2, 8, 8)
	ws := nn.NewWorkspace()

	outs, err := g.Run(ws, map[string]*tensor.Tensor{"x": x})
	if err != nil {
		t.Fatal(err)
	}
	primeira := append([]float32(nil), outs["y"].Flat()...)
	capInicial := ws.Cap()

	for i := 0; i < 10; i++ {
		ws.Reset()
		outs, err := g.Run(ws, map[string]*tensor.Tensor{"x": x})
		if err != nil {
			t.Fatal(err)
		}
		for j, v := range outs["y"].Flat() {
			if v != primeira[j] {
				t.Fatalf("passagem %d, saida[%d] = %v, na primeira deu %v", i, j, v, primeira[j])
			}
		}
	}

	if ws.Cap() != capInicial {
		t.Errorf("o workspace cresceu de %d para %d em passagens repetidas", capInicial, ws.Cap())
	}
}

func TestString(t *testing.T) {
	m := grafoSimples(no("Relu", "r", []string{"x"}, []string{"y"}))
	g, err := New(m)
	if err != nil {
		t.Fatal(err)
	}

	s := g.String()
	for _, trecho := range []string{"Relu", "simples", "x", "y"} {
		if !strings.Contains(s, trecho) {
			t.Errorf("String() deveria conter %q:\n%s", trecho, s)
		}
	}
	if g.Ops() != 1 {
		t.Errorf("Ops = %d, quero 1", g.Ops())
	}
}

func TestModeloNulo(t *testing.T) {
	if _, err := New(nil); err == nil {
		t.Error("modelo nulo deveria dar erro")
	}
	if _, err := New(&onnx.Model{}); err == nil {
		t.Error("modelo sem grafo deveria dar erro")
	}
}

func TestDominioDesconhecido(t *testing.T) {
	m := grafoSimples(&onnx.Node{
		OpType: "Relu", Name: "r", Domain: "com.exemplo.custom",
		Inputs: []string{"x"}, Outputs: []string{"y"},
	})
	if _, err := New(m); err == nil {
		t.Error("dominio desconhecido deveria dar erro")
	}
}

func BenchmarkRunGrafo(b *testing.B) {
	r := rand.New(rand.NewSource(1))

	m := modelo(&onnx.Graph{
		Nodes: []*onnx.Node{
			no("Conv", "conv", []string{"x", "w"}, []string{"c"},
				aInts("kernel_shape", 3, 3), aInts("pads", 1, 1, 1, 1), aInts("strides", 2, 2)),
			no("BatchNormalization", "bn", []string{"c", "g", "bb", "m", "v"}, []string{"bnout"}),
			no("PRelu", "prelu", []string{"bnout", "s"}, []string{"y"}),
		},
		Initializers: []*onnx.Tensor{
			peso("w", []int64{64, 3, 3, 3}, randSlice(r, 64*3*9)),
			peso("g", []int64{64}, randSlice(r, 64)),
			peso("bb", []int64{64}, randSlice(r, 64)),
			peso("m", []int64{64}, randSlice(r, 64)),
			peso("v", []int64{64}, positivos(r, 64)),
			peso("s", []int64{64}, randSlice(r, 64)),
		},
		Inputs:  []*onnx.ValueInfo{{Name: "x"}},
		Outputs: []*onnx.ValueInfo{{Name: "y"}},
	})

	g, err := New(m)
	if err != nil {
		b.Fatal(err)
	}

	x := tensor.MustFromSlice(randSlice(r, 1*3*112*112), 1, 3, 112, 112)
	ws := nn.NewWorkspace()
	ins := map[string]*tensor.Tensor{"x": x}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ws.Reset()
		if _, err := g.Run(ws, ins); err != nil {
			b.Fatal(err)
		}
	}
}
