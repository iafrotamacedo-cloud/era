package onnx

import (
	"math"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/iafrotamacedo-cloud/era/internal/protowire"
)

// Este arquivo monta modelos .onnx sinteticos, campo a campo, em vez de
// depender de um modelo baixado. Tres razoes: o repositorio nao guarda pesos,
// o teste roda offline, e -- a mais importante -- da para construir os casos
// dificeis. Um arquivo truncado, um tipo inesperado, pesos em arquivo externo:
// coisas que ninguem consegue de proposito no mundo real.

// tensorProto monta um TensorProto com dados em raw_data.
func tensorProto(nome string, dims []int64, vals []float32) *protowire.Writer {
	w := protowire.NewWriter()
	w.StringField(fTensorName, nome)
	w.Int32Field(fTensorDataType, int32(Float))
	w.PackedInt64sField(fTensorDims, dims)
	w.BytesField(fTensorRawData, protowire.RawFloats(vals))
	return w
}

// valueInfo monta um ValueInfoProto. Uma dimensao com valor 0 vira simbolica,
// com o nome dado em simbolos.
func valueInfo(nome string, dims []int64, simbolo string) *protowire.Writer {
	shape := protowire.NewWriter()
	for _, d := range dims {
		dim := protowire.NewWriter()
		if d == 0 && simbolo != "" {
			dim.StringField(fDimParam, simbolo)
		} else {
			dim.Int64Field(fDimValue, d)
		}
		shape.MessageField(fShapeDim, dim)
	}

	tensorType := protowire.NewWriter()
	tensorType.Int32Field(fTensorTypeElem, int32(Float))
	tensorType.MessageField(fTensorTypeShape, shape)

	typeProto := protowire.NewWriter()
	typeProto.MessageField(fTypeTensorType, tensorType)

	vi := protowire.NewWriter()
	vi.StringField(fValueInfoName, nome)
	vi.MessageField(fValueInfoType, typeProto)
	return vi
}

// modeloSintetico monta um modelo minimo mas completo: uma convolucao
// seguida de uma soma.
func modeloSintetico() []byte {
	// Conv com atributos tipicos.
	conv := protowire.NewWriter()
	conv.StringField(fNodeInput, "entrada")
	conv.StringField(fNodeInput, "conv1.peso")
	conv.StringField(fNodeOutput, "conv1.saida")
	conv.StringField(fNodeName, "conv1")
	conv.StringField(fNodeOpType, "Conv")

	attr := func(nome string, ints []int64) *protowire.Writer {
		a := protowire.NewWriter()
		a.StringField(fAttrName, nome)
		a.Int32Field(fAttrType, int32(AttrInts))
		a.PackedInt64sField(fAttrInts, ints)
		return a
	}
	conv.MessageField(fNodeAttribute, attr("kernel_shape", []int64{3, 3}))
	conv.MessageField(fNodeAttribute, attr("strides", []int64{2, 2}))
	conv.MessageField(fNodeAttribute, attr("pads", []int64{1, 1, 1, 1}))

	grupo := protowire.NewWriter()
	grupo.StringField(fAttrName, "group")
	grupo.Int32Field(fAttrType, int32(AttrInt))
	grupo.Int64Field(fAttrI, 1)
	conv.MessageField(fNodeAttribute, grupo)

	// Add, para haver mais de um no e uma ligacao entre eles.
	add := protowire.NewWriter()
	add.StringField(fNodeInput, "conv1.saida")
	add.StringField(fNodeInput, "vies")
	add.StringField(fNodeOutput, "saida")
	add.StringField(fNodeName, "add1")
	add.StringField(fNodeOpType, "Add")

	graph := protowire.NewWriter()
	graph.StringField(fGraphName, "rede-de-teste")
	graph.MessageField(fGraphNode, conv)
	graph.MessageField(fGraphNode, add)
	graph.MessageField(fGraphInitializer, tensorProto("conv1.peso", []int64{2, 1, 3, 3},
		make([]float32, 18)))
	graph.MessageField(fGraphInitializer, tensorProto("vies", []int64{2}, []float32{0.5, -0.5}))
	graph.MessageField(fGraphInput, valueInfo("entrada", []int64{0, 1, 8, 8}, "N"))
	graph.MessageField(fGraphOutput, valueInfo("saida", []int64{0, 2, 4, 4}, "N"))

	opset := protowire.NewWriter()
	opset.StringField(fOpsetIDDomain, "")
	opset.Int64Field(fOpsetIDVersion, 13)

	model := protowire.NewWriter()
	model.Int64Field(fModelIRVersion, 8)
	model.StringField(fModelProducerName, "era-test")
	model.StringField(fModelProducerVersion, "1.0")
	model.Int64Field(fModelVersion, 1)
	model.MessageField(fModelOpsetImport, opset)
	model.MessageField(fModelGraph, graph)

	return model.Bytes()
}

func TestParseModeloCompleto(t *testing.T) {
	m, err := Parse(modeloSintetico())
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}

	if m.IRVersion != 8 {
		t.Errorf("IRVersion = %d, quero 8", m.IRVersion)
	}
	if m.ProducerName != "era-test" {
		t.Errorf("ProducerName = %q, quero era-test", m.ProducerName)
	}
	if m.Opset() != 13 {
		t.Errorf("Opset = %d, quero 13", m.Opset())
	}
	if m.Graph.Name != "rede-de-teste" {
		t.Errorf("Graph.Name = %q, quero rede-de-teste", m.Graph.Name)
	}
	if len(m.Graph.Nodes) != 2 {
		t.Fatalf("%d nos, quero 2", len(m.Graph.Nodes))
	}
}

func TestParseNo(t *testing.T) {
	m, err := Parse(modeloSintetico())
	if err != nil {
		t.Fatal(err)
	}

	conv := m.Graph.Nodes[0]
	if conv.OpType != "Conv" {
		t.Errorf("OpType = %q, quero Conv", conv.OpType)
	}
	if conv.Name != "conv1" {
		t.Errorf("Name = %q, quero conv1", conv.Name)
	}

	// A ordem das entradas e significativa no ONNX: entrada, peso, vies.
	if want := []string{"entrada", "conv1.peso"}; !reflect.DeepEqual(conv.Inputs, want) {
		t.Errorf("Inputs = %v, quero %v", conv.Inputs, want)
	}
	if want := []string{"conv1.saida"}; !reflect.DeepEqual(conv.Outputs, want) {
		t.Errorf("Outputs = %v, quero %v", conv.Outputs, want)
	}
}

func TestParseAtributos(t *testing.T) {
	m, err := Parse(modeloSintetico())
	if err != nil {
		t.Fatal(err)
	}
	conv := m.Graph.Nodes[0]

	casos := []struct {
		nome  string
		quero []int64
	}{
		{"kernel_shape", []int64{3, 3}},
		{"strides", []int64{2, 2}},
		{"pads", []int64{1, 1, 1, 1}},
	}
	for _, c := range casos {
		if got := conv.AttrInts(c.nome, nil); !reflect.DeepEqual(got, c.quero) {
			t.Errorf("%s = %v, quero %v", c.nome, got, c.quero)
		}
	}

	if got := conv.AttrInt("group", 0); got != 1 {
		t.Errorf("group = %d, quero 1", got)
	}
}

// TestAtributoAusenteUsaPadrao cobre a regra que mais importa na leitura de
// ONNX: atributo opcional ausente significa o padrao, nao erro. "strides"
// ausente e passo 1.
func TestAtributoAusenteUsaPadrao(t *testing.T) {
	m, err := Parse(modeloSintetico())
	if err != nil {
		t.Fatal(err)
	}
	conv := m.Graph.Nodes[0]

	if got := conv.AttrInts("dilations", []int64{1, 1}); !reflect.DeepEqual(got, []int64{1, 1}) {
		t.Errorf("dilations ausente = %v, quero o padrao [1 1]", got)
	}
	if got := conv.AttrInt("nao_existe", 42); got != 42 {
		t.Errorf("AttrInt de atributo ausente = %d, quero 42", got)
	}
	if got := conv.AttrFloat("nao_existe", 1.5); got != 1.5 {
		t.Errorf("AttrFloat de atributo ausente = %v, quero 1.5", got)
	}
	if got := conv.AttrString("nao_existe", "auto"); got != "auto" {
		t.Errorf("AttrString de atributo ausente = %q, quero auto", got)
	}
	if conv.Attr("nao_existe") != nil {
		t.Error("Attr de atributo ausente deveria ser nil")
	}
}

func TestParseInitializers(t *testing.T) {
	m, err := Parse(modeloSintetico())
	if err != nil {
		t.Fatal(err)
	}

	vies := m.Graph.Initializer("vies")
	if vies == nil {
		t.Fatal("nao achei o initializer vies")
	}
	if !reflect.DeepEqual(vies.Shape(), []int{2}) {
		t.Errorf("Shape = %v, quero [2]", vies.Shape())
	}

	vals, err := vies.Floats()
	if err != nil {
		t.Fatal(err)
	}
	if want := []float32{0.5, -0.5}; !reflect.DeepEqual(vals, want) {
		t.Errorf("Floats = %v, quero %v", vals, want)
	}

	if m.Graph.Initializer("nao_existe") != nil {
		t.Error("Initializer de nome inexistente deveria ser nil")
	}
}

// TestDimensaoSimbolica cobre a primeira dimensao de quase todo modelo:
// declarada como simbolo para aceitar qualquer tamanho de lote.
func TestDimensaoSimbolica(t *testing.T) {
	m, err := Parse(modeloSintetico())
	if err != nil {
		t.Fatal(err)
	}

	entrada := m.Graph.Inputs[0]
	if entrada.Name != "entrada" {
		t.Errorf("Name = %q, quero entrada", entrada.Name)
	}
	if entrada.ElemType != Float {
		t.Errorf("ElemType = %v, quero float32", entrada.ElemType)
	}
	if len(entrada.Shape) != 4 {
		t.Fatalf("%d dimensoes, quero 4", len(entrada.Shape))
	}

	if !entrada.Shape[0].Simbolica() {
		t.Error("a primeira dimensao deveria ser simbolica")
	}
	if entrada.Shape[0].String() != "N" {
		t.Errorf("simbolo = %q, quero N", entrada.Shape[0].String())
	}
	for i, quero := range []int64{1, 8, 8} {
		d := entrada.Shape[i+1]
		if d.Simbolica() || d.Value != quero {
			t.Errorf("dimensao %d = %v, quero %d", i+1, d, quero)
		}
	}
}

// TestRepetidosEmpacotadosOuNao: um exportador pode gravar campos numericos
// repetidos das duas formas, e as duas sao validas. Ler so uma delas
// funcionaria com um modelo e falharia com outro.
func TestRepetidosEmpacotadosOuNao(t *testing.T) {
	monta := func(empacotado bool) []byte {
		tp := protowire.NewWriter()
		tp.StringField(fTensorName, "t")
		tp.Int32Field(fTensorDataType, int32(Float))

		if empacotado {
			tp.PackedInt64sField(fTensorDims, []int64{2, 3})
			tp.PackedFloatsField(fTensorFloatData, []float32{1, 2, 3, 4, 5, 6})
		} else {
			tp.Int64Field(fTensorDims, 2)
			tp.Int64Field(fTensorDims, 3)
			for _, v := range []float32{1, 2, 3, 4, 5, 6} {
				tp.FloatField(fTensorFloatData, v)
			}
		}

		g := protowire.NewWriter()
		g.MessageField(fGraphInitializer, tp)
		m := protowire.NewWriter()
		m.MessageField(fModelGraph, g)
		return m.Bytes()
	}

	for _, empacotado := range []bool{true, false} {
		nome := map[bool]string{true: "empacotado", false: "um a um"}[empacotado]
		t.Run(nome, func(t *testing.T) {
			m, err := Parse(monta(empacotado))
			if err != nil {
				t.Fatal(err)
			}

			tsr := m.Graph.Initializers[0]
			if !reflect.DeepEqual(tsr.Dims, []int64{2, 3}) {
				t.Errorf("Dims = %v, quero [2 3]", tsr.Dims)
			}

			vals, err := tsr.Floats()
			if err != nil {
				t.Fatal(err)
			}
			if want := []float32{1, 2, 3, 4, 5, 6}; !reflect.DeepEqual(vals, want) {
				t.Errorf("Floats = %v, quero %v", vals, want)
			}
		})
	}
}

func TestTensorTiposCrus(t *testing.T) {
	casos := []struct {
		nome  string
		dt    DataType
		raw   []byte
		quero []float32
	}{
		{"float32", Float, protowire.RawFloats([]float32{1.5, -2.5}), []float32{1.5, -2.5}},
		{"int32", Int32, []byte{7, 0, 0, 0, 0xff, 0xff, 0xff, 0xff}, []float32{7, -1}},
		{"int64", Int64, []byte{5, 0, 0, 0, 0, 0, 0, 0}, []float32{5}},
		{"uint8", Uint8, []byte{0, 128, 255}, []float32{0, 128, 255}},
		{"int8", Int8, []byte{0, 0x80, 0xff}, []float32{0, -128, -1}},
		{"bool", Bool, []byte{0, 1}, []float32{0, 1}},
		{"float64", Double, []byte{0, 0, 0, 0, 0, 0, 0xf0, 0x3f}, []float32{1}},
		{"bfloat16", BFloat16, []byte{0x80, 0x3f}, []float32{1}},
	}

	for _, c := range casos {
		t.Run(c.nome, func(t *testing.T) {
			tsr := &Tensor{Name: "t", DataType: c.dt, RawData: c.raw,
				Dims: []int64{int64(len(c.quero))}}

			got, err := tsr.Floats()
			if err != nil {
				t.Fatalf("Floats: %v", err)
			}
			if !reflect.DeepEqual(got, c.quero) {
				t.Errorf("Floats = %v, quero %v", got, c.quero)
			}
		})
	}
}

// TestFloat16 confere a conversao de meia precisao, incluindo os tres casos
// que nao seguem a regra geral: zero, subnormais e infinito/NaN.
func TestFloat16(t *testing.T) {
	casos := []struct {
		nome  string
		bits  uint16
		quero float32
	}{
		{"zero", 0x0000, 0},
		{"zero negativo", 0x8000, 0},
		{"um", 0x3c00, 1},
		{"menos um", 0xbc00, -1},
		{"dois", 0x4000, 2},
		{"meio", 0x3800, 0.5},
		{"maior finito", 0x7bff, 65504},
		{"menor normal", 0x0400, 6.103515625e-05},
		{"subnormal", 0x0001, 5.9604645e-08},
		{"maior subnormal", 0x03ff, 6.0975552e-05},
	}

	for _, c := range casos {
		t.Run(c.nome, func(t *testing.T) {
			got := float16ToFloat32(c.bits)
			if got != c.quero {
				t.Errorf("float16(%#04x) = %v, quero %v", c.bits, got, c.quero)
			}
		})
	}

	if !math.IsInf(float64(float16ToFloat32(0x7c00)), 1) {
		t.Error("0x7c00 deveria ser +Inf")
	}
	if !math.IsInf(float64(float16ToFloat32(0xfc00)), -1) {
		t.Error("0xfc00 deveria ser -Inf")
	}
	if !math.IsNaN(float64(float16ToFloat32(0x7e00))) {
		t.Error("0x7e00 deveria ser NaN")
	}
	if math.Signbit(float64(float16ToFloat32(0x8000))) != true {
		t.Error("zero negativo deveria manter o sinal")
	}
}

func TestTensorFloat16NoRaw(t *testing.T) {
	// 1.0 e 2.0 em float16, little-endian.
	tsr := &Tensor{Name: "t", DataType: Float16,
		RawData: []byte{0x00, 0x3c, 0x00, 0x40}, Dims: []int64{2}}

	got, err := tsr.Floats()
	if err != nil {
		t.Fatal(err)
	}
	if want := []float32{1, 2}; !reflect.DeepEqual(got, want) {
		t.Errorf("Floats = %v, quero %v", got, want)
	}
}

func TestTensorInts(t *testing.T) {
	// O formato de um Reshape vem como tensor de int64, nao como atributo.
	tsr := &Tensor{Name: "shape", DataType: Int64, Dims: []int64{2},
		RawData: []byte{
			1, 0, 0, 0, 0, 0, 0, 0,
			0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, // -1
		}}

	got, err := tsr.Ints()
	if err != nil {
		t.Fatal(err)
	}
	if want := []int64{1, -1}; !reflect.DeepEqual(got, want) {
		t.Errorf("Ints = %v, quero %v", got, want)
	}
}

// TestPesosExternosDaErroClaro: modelos grandes as vezes guardam os pesos
// fora do .onnx. Devolver tensores vazios em silencio produziria uma rede que
// roda e da resultado errado -- o pior desfecho possivel.
func TestPesosExternosDaErroClaro(t *testing.T) {
	tsr := &Tensor{
		Name: "conv1.peso", DataType: Float, Dims: []int64{10},
		DataLocation: LocationExternal, ExternalFile: "pesos.bin",
	}

	_, err := tsr.Floats()
	if err == nil {
		t.Fatal("pesos externos deveriam dar erro")
	}
	if !strings.Contains(err.Error(), "pesos.bin") {
		t.Errorf("a mensagem deveria citar o arquivo externo: %v", err)
	}

	if _, err := tsr.Ints(); err == nil {
		t.Error("Ints com pesos externos tambem deveria dar erro")
	}
}

func TestParseExternalData(t *testing.T) {
	entry := protowire.NewWriter()
	entry.StringField(fStringEntryKey, "location")
	entry.StringField(fStringEntryValue, "modelo.bin")

	tp := protowire.NewWriter()
	tp.StringField(fTensorName, "w")
	tp.Int32Field(fTensorDataType, int32(Float))
	tp.PackedInt64sField(fTensorDims, []int64{4})
	tp.Int32Field(fTensorDataLocation, int32(LocationExternal))
	tp.MessageField(fTensorExternalData, entry)

	g := protowire.NewWriter()
	g.MessageField(fGraphInitializer, tp)
	m := protowire.NewWriter()
	m.MessageField(fModelGraph, g)

	mod, err := Parse(m.Bytes())
	if err != nil {
		t.Fatal(err)
	}

	tsr := mod.Graph.Initializers[0]
	if tsr.DataLocation != LocationExternal {
		t.Errorf("DataLocation = %v, quero externo", tsr.DataLocation)
	}
	if tsr.ExternalFile != "modelo.bin" {
		t.Errorf("ExternalFile = %q, quero modelo.bin", tsr.ExternalFile)
	}
}

func TestTensorSemDadosDaErro(t *testing.T) {
	tsr := &Tensor{Name: "vazio", DataType: Float, Dims: []int64{10}}
	if _, err := tsr.Floats(); err == nil {
		t.Error("tensor que declara 10 elementos e nao traz dados deveria dar erro")
	}

	// Ja um tensor de zero elementos e legitimo.
	vazio := &Tensor{Name: "escalar-vazio", DataType: Float, Dims: []int64{0}}
	if _, err := vazio.Floats(); err != nil {
		t.Errorf("tensor de zero elementos nao deveria dar erro: %v", err)
	}
}

func TestTensorTipoNaoSuportado(t *testing.T) {
	tsr := &Tensor{Name: "t", DataType: Complex64, RawData: []byte{1, 2, 3, 4}, Dims: []int64{1}}
	if _, err := tsr.Floats(); err == nil {
		t.Error("complex64 deveria dar erro em vez de devolver lixo")
	}
}

func TestCampoDesconhecidoEIgnorado(t *testing.T) {
	// Um .onnx gerado por uma versao mais nova do formato tem campos que a
	// ERA nao conhece. Eles precisam ser pulados, nao derrubar a leitura.
	g := protowire.NewWriter()
	g.StringField(fGraphName, "rede")
	g.StringField(999, "campo do futuro")
	g.Int64Field(998, 12345)

	m := protowire.NewWriter()
	m.Int64Field(fModelIRVersion, 9)
	m.StringField(997, "outro campo do futuro")
	m.MessageField(fModelGraph, g)

	mod, err := Parse(m.Bytes())
	if err != nil {
		t.Fatalf("campos desconhecidos deveriam ser ignorados: %v", err)
	}
	if mod.Graph.Name != "rede" {
		t.Errorf("Graph.Name = %q, quero rede", mod.Graph.Name)
	}
}

func TestModeloSemGrafoDaErro(t *testing.T) {
	m := protowire.NewWriter()
	m.Int64Field(fModelIRVersion, 8)

	if _, err := Parse(m.Bytes()); err == nil {
		t.Error("modelo sem grafo deveria dar erro")
	}
}

// TestArquivoInvalidoNaoEntraEmPanico: a ERA e uma biblioteca. Um arquivo
// corrompido -- ou um arquivo que nem e um .onnx -- nao pode derrubar o
// processo de quem chama.
func TestArquivoInvalidoNaoEntraEmPanico(t *testing.T) {
	completo := modeloSintetico()

	casos := [][]byte{
		nil,
		{},
		{0xff, 0xff, 0xff},
		[]byte("isto nao e um onnx, e um texto qualquer"),
		completo[:len(completo)/2],                                // truncado no meio
		completo[:10],                                             // truncado no comeco
		append([]byte{0x0a, 0xff, 0xff, 0xff, 0x7f}, completo...), // tamanho absurdo
	}

	for i, buf := range casos {
		func() {
			defer func() {
				if p := recover(); p != nil {
					t.Errorf("caso %d entrou em panico: %v", i, p)
				}
			}()
			Parse(buf) // erro e aceitavel; panico nao
		}()
	}
}

func TestLoadArquivo(t *testing.T) {
	dir := t.TempDir()
	caminho := filepath.Join(dir, "modelo.onnx")

	if err := os.WriteFile(caminho, modeloSintetico(), 0o644); err != nil {
		t.Fatal(err)
	}

	m, err := Load(caminho)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if m.Graph.Name != "rede-de-teste" {
		t.Errorf("Graph.Name = %q, quero rede-de-teste", m.Graph.Name)
	}

	if _, err := Load(filepath.Join(dir, "nao_existe.onnx")); err == nil {
		t.Error("Load de arquivo inexistente deveria dar erro")
	}
}

func TestNodeString(t *testing.T) {
	n := &Node{OpType: "Conv", Name: "conv1",
		Inputs: []string{"x", "w"}, Outputs: []string{"y"}}
	if got := n.String(); !strings.Contains(got, "Conv") || !strings.Contains(got, "conv1") {
		t.Errorf("String = %q, deveria citar o tipo e o nome", got)
	}

	semNome := &Node{OpType: "Relu", Outputs: []string{"y"}}
	if got := semNome.String(); !strings.Contains(got, "sem nome") {
		t.Errorf("String = %q, deveria marcar que o no nao tem nome", got)
	}
}

func TestDataTypeString(t *testing.T) {
	if got := Float.String(); got != "float32" {
		t.Errorf("Float.String() = %q, quero float32", got)
	}
	if got := DataType(99).String(); !strings.Contains(got, "99") {
		t.Errorf("tipo desconhecido = %q, deveria citar o numero", got)
	}
}

func TestOpsetSemDeclaracao(t *testing.T) {
	g := protowire.NewWriter()
	m := protowire.NewWriter()
	m.MessageField(fModelGraph, g)

	mod, err := Parse(m.Bytes())
	if err != nil {
		t.Fatal(err)
	}
	if mod.Opset() != 0 {
		t.Errorf("Opset sem declaracao = %d, quero 0", mod.Opset())
	}
}
