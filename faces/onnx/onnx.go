// Package onnx le modelos no formato ONNX.
//
// Um arquivo .onnx e uma mensagem Protobuf chamada ModelProto. Dentro dela
// ha um grafo: uma lista de nos (as operacoes), uma lista de tensores
// iniciais (os pesos treinados) e a declaracao de quais valores entram e
// saem.
//
// Este pacote so LE. Ele nao executa nada e nao sabe o que e uma convolucao
// -- devolve a estrutura crua, e quem executa e o pacote graph.
//
// A separacao importa: erro de leitura de arquivo e erro de execucao de rede
// sao problemas diferentes, e misturar os dois torna as duas coisas mais
// dificeis de testar.
//
// # O que nao e suportado
//
// Pesos em arquivo externo (data_location = EXTERNAL). Modelos grandes as
// vezes guardam os pesos fora do .onnx. Isso e detectado e reportado como
// erro claro, em vez de devolver tensores vazios em silencio.
package onnx

import "fmt"

// DataType e o tipo dos elementos de um tensor, conforme
// TensorProto.DataType do ONNX.
type DataType int32

// Os tipos de dado do ONNX. A ERA usa float32 na pratica; os demais existem
// para que o parser reconheca o que encontrar e possa reclamar com clareza.
const (
	Undefined  DataType = 0
	Float      DataType = 1
	Uint8      DataType = 2
	Int8       DataType = 3
	Uint16     DataType = 4
	Int16      DataType = 5
	Int32      DataType = 6
	Int64      DataType = 7
	String     DataType = 8
	Bool       DataType = 9
	Float16    DataType = 10
	Double     DataType = 11
	Uint32     DataType = 12
	Uint64     DataType = 13
	Complex64  DataType = 14
	Complex128 DataType = 15
	BFloat16   DataType = 16
)

// String descreve o tipo.
func (d DataType) String() string {
	switch d {
	case Float:
		return "float32"
	case Uint8:
		return "uint8"
	case Int8:
		return "int8"
	case Uint16:
		return "uint16"
	case Int16:
		return "int16"
	case Int32:
		return "int32"
	case Int64:
		return "int64"
	case String:
		return "string"
	case Bool:
		return "bool"
	case Float16:
		return "float16"
	case Double:
		return "float64"
	case Uint32:
		return "uint32"
	case Uint64:
		return "uint64"
	case Complex64:
		return "complex64"
	case Complex128:
		return "complex128"
	case BFloat16:
		return "bfloat16"
	}
	return fmt.Sprintf("tipo(%d)", int32(d))
}

// AttributeType e o tipo de um atributo de no, conforme
// AttributeProto.AttributeType.
type AttributeType int32

// Os tipos de atributo do ONNX.
const (
	AttrUndefined AttributeType = 0
	AttrFloat     AttributeType = 1
	AttrInt       AttributeType = 2
	AttrString    AttributeType = 3
	AttrTensor    AttributeType = 4
	AttrGraph     AttributeType = 5
	AttrFloats    AttributeType = 6
	AttrInts      AttributeType = 7
	AttrStrings   AttributeType = 8
	AttrTensors   AttributeType = 9
	AttrGraphs    AttributeType = 10
)

// Model e um arquivo .onnx lido.
type Model struct {
	IRVersion       int64
	ProducerName    string
	ProducerVersion string
	Domain          string
	ModelVersion    int64
	DocString       string
	OpsetImports    []OpsetID
	Graph           *Graph
}

// OpsetID diz qual versao do conjunto de operadores o modelo usa.
//
// Importa porque operadores mudam de comportamento entre versoes. Um mesmo
// nome de operacao pode ter semantica diferente em opsets diferentes.
type OpsetID struct {
	Domain  string
	Version int64
}

// Opset devolve a versao do conjunto de operadores padrao (dominio vazio,
// que o ONNX chama de ai.onnx). Devolve 0 se o modelo nao declarar.
func (m *Model) Opset() int64 {
	for _, o := range m.OpsetImports {
		if o.Domain == "" || o.Domain == "ai.onnx" {
			return o.Version
		}
	}
	return 0
}

// Graph e o grafo de computacao.
type Graph struct {
	Name string

	// Nodes sao as operacoes. O ONNX exige que venham em ordem topologica,
	// mas a ERA nao confia nisso -- o pacote graph reordena.
	Nodes []*Node

	// Initializers sao os pesos treinados, indexados por nome.
	Initializers []*Tensor

	Inputs     []*ValueInfo
	Outputs    []*ValueInfo
	ValueInfos []*ValueInfo
}

// Initializer procura um peso pelo nome.
func (g *Graph) Initializer(name string) *Tensor {
	for _, t := range g.Initializers {
		if t.Name == name {
			return t
		}
	}
	return nil
}

// Node e uma operacao do grafo.
//
// Entradas e saidas sao NOMES, nao ponteiros. E assim que o ONNX liga os nos:
// a saida "conv1_out" de um no e a entrada "conv1_out" do seguinte. Montar o
// grafo de verdade e resolver esses nomes, e e trabalho do pacote graph.
type Node struct {
	Name       string
	OpType     string
	Domain     string
	Inputs     []string
	Outputs    []string
	Attributes []*Attribute
}

// Attr procura um atributo pelo nome, devolvendo nil se nao houver.
func (n *Node) Attr(name string) *Attribute {
	for _, a := range n.Attributes {
		if a.Name == name {
			return a
		}
	}
	return nil
}

// AttrInts devolve um atributo de lista de inteiros, ou o padrao se ele nao
// existir.
//
// Atributos opcionais com padrao sao a regra no ONNX -- "strides" ausente
// significa passo 1, nao erro. Por isso a busca devolve o padrao em vez de
// falhar.
func (n *Node) AttrInts(name string, padrao []int64) []int64 {
	if a := n.Attr(name); a != nil && len(a.Ints) > 0 {
		return a.Ints
	}
	return padrao
}

// AttrInt devolve um atributo inteiro, ou o padrao.
func (n *Node) AttrInt(name string, padrao int64) int64 {
	if a := n.Attr(name); a != nil {
		return a.I
	}
	return padrao
}

// AttrFloat devolve um atributo float, ou o padrao.
func (n *Node) AttrFloat(name string, padrao float32) float32 {
	if a := n.Attr(name); a != nil {
		return a.F
	}
	return padrao
}

// AttrString devolve um atributo de texto, ou o padrao.
func (n *Node) AttrString(name string, padrao string) string {
	if a := n.Attr(name); a != nil && a.S != nil {
		return string(a.S)
	}
	return padrao
}

// String descreve o no de forma legivel.
func (n *Node) String() string {
	nome := n.Name
	if nome == "" {
		nome = "(sem nome)"
	}
	return fmt.Sprintf("%s %s: %v -> %v", n.OpType, nome, n.Inputs, n.Outputs)
}

// Attribute e um parametro de um no -- o tamanho do kernel, o passo, o
// padding.
type Attribute struct {
	Name string
	Type AttributeType

	F float32
	I int64
	S []byte
	T *Tensor

	Floats  []float32
	Ints    []int64
	Strings [][]byte
	Tensors []*Tensor
}

// ValueInfo declara o nome e o tipo de um valor que entra ou sai do grafo.
type ValueInfo struct {
	Name     string
	ElemType DataType
	Shape    []Dim
}

// Dim e uma dimensao, que pode ser um numero ou um simbolo.
//
// Modelos costumam declarar a primeira dimensao como simbolica -- "N",
// "batch_size" -- para aceitar qualquer tamanho de lote. Nesse caso Value e
// zero e Param traz o nome.
type Dim struct {
	Value int64
	Param string
}

// Simbolica informa se a dimensao e um simbolo em vez de um numero fixo.
func (d Dim) Simbolica() bool { return d.Param != "" }

// String descreve a dimensao.
func (d Dim) String() string {
	if d.Simbolica() {
		return d.Param
	}
	return fmt.Sprintf("%d", d.Value)
}
