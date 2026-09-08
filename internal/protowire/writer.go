package protowire

import (
	"encoding/binary"
	"math"
)

// Writer monta uma mensagem protobuf.
//
// A ERA nao precisa gravar protobuf em producao -- ela so le modelos e mapas.
// O escritor existe para os testes: sem ele, testar o parser de .onnx exigiria
// baixar um modelo de verdade, e o repositorio nao guarda pesos.
//
// Com ele, um teste monta a mensagem que quiser -- inclusive as malformadas,
// que sao justamente as dificeis de obter no mundo real.
type Writer struct {
	buf []byte
}

// NewWriter cria um escritor vazio.
func NewWriter() *Writer { return &Writer{} }

// Bytes devolve a mensagem montada.
func (w *Writer) Bytes() []byte { return w.buf }

// Len devolve o tamanho da mensagem ate agora.
func (w *Writer) Len() int { return len(w.buf) }

// Varint grava um inteiro de tamanho variavel, sem tag.
func (w *Writer) Varint(v uint64) {
	for v >= 0x80 {
		w.buf = append(w.buf, byte(v)|0x80)
		v >>= 7
	}
	w.buf = append(w.buf, byte(v))
}

// Tag grava a tag de um campo.
func (w *Writer) Tag(field int32, typ Type) {
	w.Varint(uint64(field)<<3 | uint64(typ))
}

// Raw acrescenta bytes crus, sem tag nem tamanho. Serve para montar
// mensagens propositalmente malformadas nos testes.
func (w *Writer) Raw(b ...byte) { w.buf = append(w.buf, b...) }

// Int64Field grava um campo varint com sinal.
func (w *Writer) Int64Field(field int32, v int64) {
	w.Tag(field, Varint)
	w.Varint(uint64(v))
}

// Int32Field grava um campo varint de 32 bits.
func (w *Writer) Int32Field(field int32, v int32) {
	w.Tag(field, Varint)
	w.Varint(uint64(int64(v)))
}

// BoolField grava um campo booleano.
func (w *Writer) BoolField(field int32, v bool) {
	var b uint64
	if v {
		b = 1
	}
	w.Tag(field, Varint)
	w.Varint(b)
}

// SInt64Field grava um campo varint zigzag.
func (w *Writer) SInt64Field(field int32, v int64) {
	w.Tag(field, Varint)
	w.Varint(uint64(v<<1) ^ uint64(v>>63))
}

// FloatField grava um campo float de 32 bits.
func (w *Writer) FloatField(field int32, v float32) {
	w.Tag(field, Fixed32)
	w.buf = binary.LittleEndian.AppendUint32(w.buf, math.Float32bits(v))
}

// DoubleField grava um campo float de 64 bits.
func (w *Writer) DoubleField(field int32, v float64) {
	w.Tag(field, Fixed64)
	w.buf = binary.LittleEndian.AppendUint64(w.buf, math.Float64bits(v))
}

// BytesField grava um campo delimitado por tamanho.
func (w *Writer) BytesField(field int32, b []byte) {
	w.Tag(field, Bytes)
	w.Varint(uint64(len(b)))
	w.buf = append(w.buf, b...)
}

// StringField grava um campo de texto.
func (w *Writer) StringField(field int32, s string) {
	w.BytesField(field, []byte(s))
}

// MessageField grava uma mensagem aninhada.
func (w *Writer) MessageField(field int32, m *Writer) {
	w.BytesField(field, m.Bytes())
}

// PackedFloatsField grava uma sequencia empacotada de float32.
func (w *Writer) PackedFloatsField(field int32, vals []float32) {
	b := make([]byte, 0, len(vals)*4)
	for _, v := range vals {
		b = binary.LittleEndian.AppendUint32(b, math.Float32bits(v))
	}
	w.BytesField(field, b)
}

// PackedInt64sField grava uma sequencia empacotada de varints.
func (w *Writer) PackedInt64sField(field int32, vals []int64) {
	inner := NewWriter()
	for _, v := range vals {
		inner.Varint(uint64(v))
	}
	w.BytesField(field, inner.Bytes())
}

// PackedInt32sField grava uma sequencia empacotada de varints de 32 bits.
func (w *Writer) PackedInt32sField(field int32, vals []int32) {
	inner := NewWriter()
	for _, v := range vals {
		inner.Varint(uint64(int64(v)))
	}
	w.BytesField(field, inner.Bytes())
}

// PackedSInt64sField grava uma sequencia empacotada de varints zigzag.
func (w *Writer) PackedSInt64sField(field int32, vals []int64) {
	inner := NewWriter()
	for _, v := range vals {
		inner.Varint(uint64(v<<1) ^ uint64(v>>63))
	}
	w.BytesField(field, inner.Bytes())
}

// RawFloats devolve os bytes little-endian de uma sequencia de float32, do
// jeito que o campo raw_data do TensorProto guarda os pesos.
func RawFloats(vals []float32) []byte {
	b := make([]byte, 0, len(vals)*4)
	for _, v := range vals {
		b = binary.LittleEndian.AppendUint32(b, math.Float32bits(v))
	}
	return b
}
