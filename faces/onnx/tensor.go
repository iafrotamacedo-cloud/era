package onnx

import (
	"encoding/binary"
	"fmt"
	"math"
)

// DataLocation diz onde os dados do tensor estao.
type DataLocation int32

// Onde os dados de um tensor podem estar.
const (
	LocationDefault  DataLocation = 0 // dentro do proprio .onnx
	LocationExternal DataLocation = 1 // num arquivo separado
)

// Tensor e um TensorProto: os pesos treinados, ou uma constante do grafo.
//
// O ONNX guarda os numeros de duas formas concorrentes. Ou em campos tipados
// (FloatData, Int64Data...), ou em RawData -- um bloco de bytes little-endian
// sem tipo. Na pratica, todo exportador serio usa RawData, porque e menor e
// mais rapido de gravar. As duas sao suportadas.
type Tensor struct {
	Name     string
	DataType DataType
	Dims     []int64

	FloatData  []float32
	Int32Data  []int32
	Int64Data  []int64
	DoubleData []float64
	RawData    []byte

	DataLocation DataLocation
	ExternalFile string
}

// Shape devolve as dimensoes como []int.
func (t *Tensor) Shape() []int {
	s := make([]int, len(t.Dims))
	for i, d := range t.Dims {
		s[i] = int(d)
	}
	return s
}

// Count devolve quantos elementos o tensor tem, pelas dimensoes declaradas.
func (t *Tensor) Count() int {
	n := 1
	for _, d := range t.Dims {
		n *= int(d)
	}
	return n
}

// Floats devolve os dados como []float32, convertendo do que houver.
//
// Aceita float32, float16, float64, int32 e int64 -- os tipos que aparecem em
// modelos de reconhecimento facial. Os demais viram erro explicito, porque
// devolver zeros silenciosamente produziria uma rede que roda e da resultado
// errado, que e o pior desfecho possivel.
func (t *Tensor) Floats() ([]float32, error) {
	if t.DataLocation == LocationExternal {
		return nil, fmt.Errorf("onnx: tensor %q guarda os pesos no arquivo externo %q, o que ainda nao e suportado",
			t.Name, t.ExternalFile)
	}

	// Campos tipados, quando o exportador os usou.
	switch {
	case len(t.FloatData) > 0:
		return t.FloatData, nil

	case len(t.Int64Data) > 0:
		out := make([]float32, len(t.Int64Data))
		for i, v := range t.Int64Data {
			out[i] = float32(v)
		}
		return out, nil

	case len(t.Int32Data) > 0:
		out := make([]float32, len(t.Int32Data))
		for i, v := range t.Int32Data {
			out[i] = float32(v)
		}
		return out, nil

	case len(t.DoubleData) > 0:
		out := make([]float32, len(t.DoubleData))
		for i, v := range t.DoubleData {
			out[i] = float32(v)
		}
		return out, nil
	}

	if len(t.RawData) == 0 {
		// Tensor sem dado nenhum e valido apenas se ele nao tiver elementos.
		if t.Count() == 0 {
			return nil, nil
		}
		return nil, fmt.Errorf("onnx: tensor %q declara %d elementos mas nao traz dados", t.Name, t.Count())
	}

	return t.rawFloats()
}

// rawFloats converte RawData conforme o tipo declarado. Os bytes sao sempre
// little-endian, independentemente da maquina que gravou.
func (t *Tensor) rawFloats() ([]float32, error) {
	b := t.RawData

	tamanho := map[DataType]int{
		Float: 4, Float16: 2, BFloat16: 2, Double: 8,
		Int32: 4, Int64: 8, Uint8: 1, Int8: 1, Bool: 1,
	}[t.DataType]

	if tamanho == 0 {
		return nil, fmt.Errorf("onnx: tensor %q e do tipo %v, que a ERA ainda nao converte", t.Name, t.DataType)
	}
	if len(b)%tamanho != 0 {
		return nil, fmt.Errorf("onnx: tensor %q tem %d bytes, que nao dividem em elementos de %v",
			t.Name, len(b), t.DataType)
	}

	n := len(b) / tamanho
	out := make([]float32, n)

	switch t.DataType {
	case Float:
		for i := 0; i < n; i++ {
			out[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[i*4:]))
		}

	case Float16:
		for i := 0; i < n; i++ {
			out[i] = float16ToFloat32(binary.LittleEndian.Uint16(b[i*2:]))
		}

	case BFloat16:
		// bfloat16 e float32 com a mantissa truncada: basta recolocar os
		// 16 bits no alto e zerar o resto.
		for i := 0; i < n; i++ {
			out[i] = math.Float32frombits(uint32(binary.LittleEndian.Uint16(b[i*2:])) << 16)
		}

	case Double:
		for i := 0; i < n; i++ {
			out[i] = float32(math.Float64frombits(binary.LittleEndian.Uint64(b[i*8:])))
		}

	case Int32:
		for i := 0; i < n; i++ {
			out[i] = float32(int32(binary.LittleEndian.Uint32(b[i*4:])))
		}

	case Int64:
		for i := 0; i < n; i++ {
			out[i] = float32(int64(binary.LittleEndian.Uint64(b[i*8:])))
		}

	case Uint8, Bool:
		for i := 0; i < n; i++ {
			out[i] = float32(b[i])
		}

	case Int8:
		for i := 0; i < n; i++ {
			out[i] = float32(int8(b[i]))
		}
	}

	return out, nil
}

// Ints devolve os dados como []int64.
//
// Alguns operadores levam os parametros como tensor em vez de atributo --
// o formato de um Reshape, por exemplo, e um tensor de int64.
func (t *Tensor) Ints() ([]int64, error) {
	if t.DataLocation == LocationExternal {
		return nil, fmt.Errorf("onnx: tensor %q guarda os dados no arquivo externo %q, o que ainda nao e suportado",
			t.Name, t.ExternalFile)
	}

	if len(t.Int64Data) > 0 {
		return t.Int64Data, nil
	}
	if len(t.Int32Data) > 0 {
		out := make([]int64, len(t.Int32Data))
		for i, v := range t.Int32Data {
			out[i] = int64(v)
		}
		return out, nil
	}

	if len(t.RawData) == 0 {
		if t.Count() == 0 {
			return nil, nil
		}
		return nil, fmt.Errorf("onnx: tensor %q declara %d elementos mas nao traz dados", t.Name, t.Count())
	}

	switch t.DataType {
	case Int64:
		if len(t.RawData)%8 != 0 {
			return nil, fmt.Errorf("onnx: tensor %q tem %d bytes, que nao dividem em int64", t.Name, len(t.RawData))
		}
		out := make([]int64, len(t.RawData)/8)
		for i := range out {
			out[i] = int64(binary.LittleEndian.Uint64(t.RawData[i*8:]))
		}
		return out, nil

	case Int32:
		if len(t.RawData)%4 != 0 {
			return nil, fmt.Errorf("onnx: tensor %q tem %d bytes, que nao dividem em int32", t.Name, len(t.RawData))
		}
		out := make([]int64, len(t.RawData)/4)
		for i := range out {
			out[i] = int64(int32(binary.LittleEndian.Uint32(t.RawData[i*4:])))
		}
		return out, nil
	}

	return nil, fmt.Errorf("onnx: tensor %q e do tipo %v, que nao converte para inteiro", t.Name, t.DataType)
}

// float16ToFloat32 converte meia precisao IEEE 754 para precisao simples.
//
//	float16:  1 sinal |  5 expoente | 10 mantissa   (vies 15)
//	float32:  1 sinal |  8 expoente | 23 mantissa   (vies 127)
//
// Todo float16 cabe num float32, entao a conversao nunca perde nada. Os tres
// casos que exigem cuidado sao o zero, os subnormais e os infinitos/NaN --
// que tem expoente todo zero ou todo um, e nao seguem a regra geral.
func float16ToFloat32(h uint16) float32 {
	sinal := uint32(h&0x8000) << 16
	exp := uint32(h>>10) & 0x1f
	mant := uint32(h & 0x03ff)

	switch exp {
	case 0:
		if mant == 0 {
			return math.Float32frombits(sinal) // zero, com sinal
		}
		// Subnormal em float16 vira normal em float32: normaliza deslocando
		// a mantissa ate o bit implicito aparecer, descontando do expoente.
		exp = 1
		for mant&0x0400 == 0 {
			mant <<= 1
			exp--
		}
		mant &= 0x03ff
		return math.Float32frombits(sinal | (exp+127-15)<<23 | mant<<13)

	case 0x1f:
		// Infinito ou NaN: expoente todo um nos dois formatos.
		return math.Float32frombits(sinal | 0xff<<23 | mant<<13)
	}

	return math.Float32frombits(sinal | (exp+127-15)<<23 | mant<<13)
}
