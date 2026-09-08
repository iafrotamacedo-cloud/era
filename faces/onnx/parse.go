package onnx

import (
	"fmt"
	"io"
	"os"

	"github.com/iafrotamacedo-cloud/era/internal/protowire"
)

// Os numeros de campo do esquema ONNX (onnx.proto).
//
// Estao escritos como constantes porque numero solto no meio de um switch e
// impossivel de conferir contra a especificacao. A numeracao nao e sequencial
// -- campos foram acrescentados ao longo dos anos e receberam os numeros
// livres do momento.
const (
	// ModelProto
	fModelIRVersion       = 1
	fModelProducerName    = 2
	fModelProducerVersion = 3
	fModelDomain          = 4
	fModelVersion         = 5
	fModelDocString       = 6
	fModelGraph           = 7
	fModelOpsetImport     = 8

	// GraphProto
	fGraphNode        = 1
	fGraphName        = 2
	fGraphInitializer = 5
	fGraphInput       = 11
	fGraphOutput      = 12
	fGraphValueInfo   = 13

	// NodeProto
	fNodeInput     = 1
	fNodeOutput    = 2
	fNodeName      = 3
	fNodeOpType    = 4
	fNodeAttribute = 5
	fNodeDomain    = 7

	// AttributeProto
	fAttrName    = 1
	fAttrF       = 2
	fAttrI       = 3
	fAttrS       = 4
	fAttrT       = 5
	fAttrFloats  = 7
	fAttrInts    = 8
	fAttrStrings = 9
	fAttrTensors = 10
	fAttrType    = 20

	// TensorProto
	fTensorDims         = 1
	fTensorDataType     = 2
	fTensorFloatData    = 4
	fTensorInt32Data    = 5
	fTensorInt64Data    = 7
	fTensorName         = 8
	fTensorRawData      = 9
	fTensorDoubleData   = 10
	fTensorExternalData = 13
	fTensorDataLocation = 14

	// ValueInfoProto
	fValueInfoName = 1
	fValueInfoType = 2

	// TypeProto / TypeProto.Tensor
	fTypeTensorType   = 1
	fTensorTypeElem   = 1
	fTensorTypeShape  = 2
	fShapeDim         = 1
	fDimValue         = 1
	fDimParam         = 2
	fOpsetIDDomain    = 1
	fOpsetIDVersion   = 2
	fStringEntryKey   = 1
	fStringEntryValue = 2
)

// tamanhoMaximo limita o arquivo que Load aceita.
//
// Existe porque Load le tudo na memoria: sem limite, apontar para o arquivo
// errado -- um dump, um video -- consumiria a RAM da maquina antes de
// qualquer verificacao de formato. 2 GB e folgado para qualquer modelo de
// reconhecimento facial; o maior que avaliamos tem 249 MB.
const tamanhoMaximo = 2 << 30

// Load le um modelo de um arquivo .onnx.
func Load(caminho string) (*Model, error) {
	f, err := os.Open(caminho)
	if err != nil {
		return nil, fmt.Errorf("onnx: %w", err)
	}
	defer f.Close()

	info, err := f.Stat()
	if err != nil {
		return nil, fmt.Errorf("onnx: %w", err)
	}
	if info.Size() > tamanhoMaximo {
		return nil, fmt.Errorf("onnx: %s tem %d bytes, acima do limite de %d", caminho, info.Size(), tamanhoMaximo)
	}

	buf, err := io.ReadAll(f)
	if err != nil {
		return nil, fmt.Errorf("onnx: lendo %s: %w", caminho, err)
	}

	m, err := Parse(buf)
	if err != nil {
		return nil, fmt.Errorf("onnx: %s: %w", caminho, err)
	}
	return m, nil
}

// Parse le um modelo a partir dos bytes de um .onnx.
//
// Os slices do modelo apontam para buf. Guardar o modelo mantem buf vivo, o
// que e desejavel: sao os pesos, e copia-los dobraria a memoria a toa.
func Parse(buf []byte) (*Model, error) {
	m := &Model{}
	r := protowire.New(buf)

	for !r.Done() {
		field, typ, err := r.Tag()
		if err != nil {
			return nil, err
		}

		switch field {
		case fModelIRVersion:
			m.IRVersion, err = r.Int64()
		case fModelProducerName:
			m.ProducerName, err = r.String()
		case fModelProducerVersion:
			m.ProducerVersion, err = r.String()
		case fModelDomain:
			m.Domain, err = r.String()
		case fModelVersion:
			m.ModelVersion, err = r.Int64()
		case fModelDocString:
			m.DocString, err = r.String()

		case fModelGraph:
			var sub *protowire.Reader
			if sub, err = r.Message(); err == nil {
				m.Graph, err = parseGraph(sub)
			}

		case fModelOpsetImport:
			var sub *protowire.Reader
			if sub, err = r.Message(); err == nil {
				var o OpsetID
				if o, err = parseOpsetID(sub); err == nil {
					m.OpsetImports = append(m.OpsetImports, o)
				}
			}

		default:
			err = r.Skip(typ)
		}

		if err != nil {
			return nil, fmt.Errorf("ModelProto campo %d: %w", field, err)
		}
	}

	if m.Graph == nil {
		return nil, fmt.Errorf("onnx: o modelo nao tem grafo")
	}
	return m, nil
}

func parseOpsetID(r *protowire.Reader) (OpsetID, error) {
	var o OpsetID
	for !r.Done() {
		field, typ, err := r.Tag()
		if err != nil {
			return o, err
		}
		switch field {
		case fOpsetIDDomain:
			o.Domain, err = r.String()
		case fOpsetIDVersion:
			o.Version, err = r.Int64()
		default:
			err = r.Skip(typ)
		}
		if err != nil {
			return o, err
		}
	}
	return o, nil
}

func parseGraph(r *protowire.Reader) (*Graph, error) {
	g := &Graph{}

	for !r.Done() {
		field, typ, err := r.Tag()
		if err != nil {
			return nil, err
		}

		switch field {
		case fGraphName:
			g.Name, err = r.String()

		case fGraphNode:
			var sub *protowire.Reader
			if sub, err = r.Message(); err == nil {
				var n *Node
				if n, err = parseNode(sub); err == nil {
					g.Nodes = append(g.Nodes, n)
				}
			}

		case fGraphInitializer:
			var sub *protowire.Reader
			if sub, err = r.Message(); err == nil {
				var t *Tensor
				if t, err = parseTensor(sub); err == nil {
					g.Initializers = append(g.Initializers, t)
				}
			}

		case fGraphInput, fGraphOutput, fGraphValueInfo:
			var sub *protowire.Reader
			if sub, err = r.Message(); err == nil {
				var v *ValueInfo
				if v, err = parseValueInfo(sub); err == nil {
					switch field {
					case fGraphInput:
						g.Inputs = append(g.Inputs, v)
					case fGraphOutput:
						g.Outputs = append(g.Outputs, v)
					default:
						g.ValueInfos = append(g.ValueInfos, v)
					}
				}
			}

		default:
			err = r.Skip(typ)
		}

		if err != nil {
			return nil, fmt.Errorf("GraphProto campo %d: %w", field, err)
		}
	}

	return g, nil
}

func parseNode(r *protowire.Reader) (*Node, error) {
	n := &Node{}

	for !r.Done() {
		field, typ, err := r.Tag()
		if err != nil {
			return nil, err
		}

		switch field {
		case fNodeInput:
			var s string
			if s, err = r.String(); err == nil {
				n.Inputs = append(n.Inputs, s)
			}
		case fNodeOutput:
			var s string
			if s, err = r.String(); err == nil {
				n.Outputs = append(n.Outputs, s)
			}
		case fNodeName:
			n.Name, err = r.String()
		case fNodeOpType:
			n.OpType, err = r.String()
		case fNodeDomain:
			n.Domain, err = r.String()

		case fNodeAttribute:
			var sub *protowire.Reader
			if sub, err = r.Message(); err == nil {
				var a *Attribute
				if a, err = parseAttribute(sub); err == nil {
					n.Attributes = append(n.Attributes, a)
				}
			}

		default:
			err = r.Skip(typ)
		}

		if err != nil {
			return nil, fmt.Errorf("NodeProto campo %d: %w", field, err)
		}
	}

	return n, nil
}

func parseAttribute(r *protowire.Reader) (*Attribute, error) {
	a := &Attribute{}

	for !r.Done() {
		field, typ, err := r.Tag()
		if err != nil {
			return nil, err
		}

		switch field {
		case fAttrName:
			a.Name, err = r.String()
		case fAttrType:
			var v int32
			if v, err = r.Int32(); err == nil {
				a.Type = AttributeType(v)
			}
		case fAttrF:
			a.F, err = r.Float()
		case fAttrI:
			a.I, err = r.Int64()
		case fAttrS:
			a.S, err = r.Bytes()

		case fAttrT:
			var sub *protowire.Reader
			if sub, err = r.Message(); err == nil {
				a.T, err = parseTensor(sub)
			}

		case fAttrFloats:
			// Repetidos podem vir empacotados ou um a um, e um exportador
			// pode escolher qualquer um dos dois. O tipo na tag diz qual e.
			if typ == protowire.Bytes {
				a.Floats, err = r.PackedFloats(a.Floats)
			} else {
				var v float32
				if v, err = r.Float(); err == nil {
					a.Floats = append(a.Floats, v)
				}
			}

		case fAttrInts:
			if typ == protowire.Bytes {
				a.Ints, err = r.PackedInt64s(a.Ints)
			} else {
				var v int64
				if v, err = r.Int64(); err == nil {
					a.Ints = append(a.Ints, v)
				}
			}

		case fAttrStrings:
			var b []byte
			if b, err = r.Bytes(); err == nil {
				a.Strings = append(a.Strings, b)
			}

		case fAttrTensors:
			var sub *protowire.Reader
			if sub, err = r.Message(); err == nil {
				var t *Tensor
				if t, err = parseTensor(sub); err == nil {
					a.Tensors = append(a.Tensors, t)
				}
			}

		default:
			err = r.Skip(typ)
		}

		if err != nil {
			return nil, fmt.Errorf("AttributeProto campo %d: %w", field, err)
		}
	}

	return a, nil
}

func parseTensor(r *protowire.Reader) (*Tensor, error) {
	t := &Tensor{}

	for !r.Done() {
		field, typ, err := r.Tag()
		if err != nil {
			return nil, err
		}

		switch field {
		case fTensorName:
			t.Name, err = r.String()

		case fTensorDataType:
			var v int32
			if v, err = r.Int32(); err == nil {
				t.DataType = DataType(v)
			}

		case fTensorDims:
			if typ == protowire.Bytes {
				t.Dims, err = r.PackedInt64s(t.Dims)
			} else {
				var v int64
				if v, err = r.Int64(); err == nil {
					t.Dims = append(t.Dims, v)
				}
			}

		case fTensorRawData:
			t.RawData, err = r.Bytes()

		case fTensorFloatData:
			if typ == protowire.Bytes {
				t.FloatData, err = r.PackedFloats(t.FloatData)
			} else {
				var v float32
				if v, err = r.Float(); err == nil {
					t.FloatData = append(t.FloatData, v)
				}
			}

		case fTensorInt64Data:
			if typ == protowire.Bytes {
				t.Int64Data, err = r.PackedInt64s(t.Int64Data)
			} else {
				var v int64
				if v, err = r.Int64(); err == nil {
					t.Int64Data = append(t.Int64Data, v)
				}
			}

		case fTensorInt32Data:
			if typ == protowire.Bytes {
				t.Int32Data, err = r.PackedInt32s(t.Int32Data)
			} else {
				var v int32
				if v, err = r.Int32(); err == nil {
					t.Int32Data = append(t.Int32Data, v)
				}
			}

		case fTensorDoubleData:
			if typ == protowire.Bytes {
				t.DoubleData, err = r.PackedDoubles(t.DoubleData)
			} else {
				var v float64
				if v, err = r.Double(); err == nil {
					t.DoubleData = append(t.DoubleData, v)
				}
			}

		case fTensorDataLocation:
			var v int32
			if v, err = r.Int32(); err == nil {
				t.DataLocation = DataLocation(v)
			}

		case fTensorExternalData:
			// Lista de pares chave/valor; a chave "location" traz o caminho
			// do arquivo com os pesos. Guardamos so para a mensagem de erro.
			var sub *protowire.Reader
			if sub, err = r.Message(); err == nil {
				var chave, valor string
				for !sub.Done() && err == nil {
					var f2 int32
					var t2 protowire.Type
					if f2, t2, err = sub.Tag(); err != nil {
						break
					}
					switch f2 {
					case fStringEntryKey:
						chave, err = sub.String()
					case fStringEntryValue:
						valor, err = sub.String()
					default:
						err = sub.Skip(t2)
					}
				}
				if err == nil && chave == "location" {
					t.ExternalFile = valor
				}
			}

		default:
			err = r.Skip(typ)
		}

		if err != nil {
			return nil, fmt.Errorf("TensorProto campo %d: %w", field, err)
		}
	}

	return t, nil
}

func parseValueInfo(r *protowire.Reader) (*ValueInfo, error) {
	v := &ValueInfo{}

	for !r.Done() {
		field, typ, err := r.Tag()
		if err != nil {
			return nil, err
		}

		switch field {
		case fValueInfoName:
			v.Name, err = r.String()

		case fValueInfoType:
			var sub *protowire.Reader
			if sub, err = r.Message(); err == nil {
				err = parseTypeInto(sub, v)
			}

		default:
			err = r.Skip(typ)
		}

		if err != nil {
			return nil, fmt.Errorf("ValueInfoProto campo %d: %w", field, err)
		}
	}

	return v, nil
}

// parseTypeInto le um TypeProto. So o caso de tensor interessa: sequencias e
// mapas nao aparecem em modelos de visao.
func parseTypeInto(r *protowire.Reader, v *ValueInfo) error {
	for !r.Done() {
		field, typ, err := r.Tag()
		if err != nil {
			return err
		}

		if field != fTypeTensorType {
			if err := r.Skip(typ); err != nil {
				return err
			}
			continue
		}

		sub, err := r.Message()
		if err != nil {
			return err
		}

		for !sub.Done() {
			f2, t2, err := sub.Tag()
			if err != nil {
				return err
			}
			switch f2 {
			case fTensorTypeElem:
				var e int32
				if e, err = sub.Int32(); err == nil {
					v.ElemType = DataType(e)
				}
			case fTensorTypeShape:
				var sh *protowire.Reader
				if sh, err = sub.Message(); err == nil {
					v.Shape, err = parseShape(sh)
				}
			default:
				err = sub.Skip(t2)
			}
			if err != nil {
				return err
			}
		}
	}
	return nil
}

func parseShape(r *protowire.Reader) ([]Dim, error) {
	var dims []Dim

	for !r.Done() {
		field, typ, err := r.Tag()
		if err != nil {
			return nil, err
		}

		if field != fShapeDim {
			if err := r.Skip(typ); err != nil {
				return nil, err
			}
			continue
		}

		sub, err := r.Message()
		if err != nil {
			return nil, err
		}

		var d Dim
		for !sub.Done() {
			f2, t2, err := sub.Tag()
			if err != nil {
				return nil, err
			}
			switch f2 {
			case fDimValue:
				d.Value, err = sub.Int64()
			case fDimParam:
				d.Param, err = sub.String()
			default:
				err = sub.Skip(t2)
			}
			if err != nil {
				return nil, err
			}
		}
		dims = append(dims, d)
	}

	return dims, nil
}
