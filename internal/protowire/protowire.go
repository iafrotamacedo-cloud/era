// Package protowire le o formato binario do Protocol Buffers.
//
// Nao gera codigo, nao conhece esquema, nao depende de nada. Entrega os
// campos crus -- numero, tipo e valor -- e quem chama decide o que fazer com
// cada um. Um parser de mensagem vira um switch sobre o numero do campo.
//
// Existe porque dois motores da ERA leem protobuf: o faces le modelos .onnx e
// o maps le extratos .osm.pbf. Sao esquemas completamente diferentes sobre o
// mesmo formato binario, e o formato e simples o bastante para caber aqui
// inteiro.
//
// # O formato
//
// Uma mensagem e uma sequencia de campos. Cada campo comeca por uma tag --
// um varint que empacota o numero do campo e o tipo de codificacao:
//
//	tag = (numero << 3) | tipo
//
// Sao seis tipos, dos quais quatro ainda se usam:
//
//	0  Varint    inteiro de tamanho variavel
//	1  Fixed64   8 bytes, little-endian
//	2  Bytes     varint com o tamanho, seguido dos bytes
//	5  Fixed32   4 bytes, little-endian
//
// Os tipos 3 e 4 (grupos) foram descontinuados ha mais de uma decada. Sao
// reconhecidos apenas para que Skip consiga pula-los.
//
// # Campos desconhecidos
//
// A tag traz o tipo, entao da para pular um campo sem saber o que ele e.
// E isso que permite ler um .onnx gerado por uma versao mais nova do formato
// sem quebrar: o que nao se conhece, se ignora.
package protowire

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
)

// Type e o tipo de codificacao de um campo.
type Type uint8

// Os tipos de codificacao do formato.
const (
	Varint     Type = 0
	Fixed64    Type = 1
	Bytes      Type = 2
	StartGroup Type = 3 // descontinuado
	EndGroup   Type = 4 // descontinuado
	Fixed32    Type = 5
)

// String descreve o tipo.
func (t Type) String() string {
	switch t {
	case Varint:
		return "varint"
	case Fixed64:
		return "fixed64"
	case Bytes:
		return "bytes"
	case StartGroup:
		return "start-group"
	case EndGroup:
		return "end-group"
	case Fixed32:
		return "fixed32"
	}
	return fmt.Sprintf("tipo-desconhecido(%d)", uint8(t))
}

// ErrTruncado indica que os dados acabaram no meio de um campo.
var ErrTruncado = errors.New("protowire: dados truncados")

// ErrMalformado indica codificacao invalida.
var ErrMalformado = errors.New("protowire: dados malformados")

// varintMaxBytes e o limite de um varint de 64 bits: 10 grupos de 7 bits
// cobrem os 64. Um varint mais longo que isso e lixo, e sem esse limite um
// arquivo corrompido levaria o laco a varrer o buffer inteiro.
const varintMaxBytes = 10

// Reader percorre uma mensagem protobuf.
//
// Nao copia nada: os slices devolvidos por Bytes apontam para o buffer
// original. Isso torna a leitura barata, e significa que o buffer precisa
// continuar vivo enquanto os dados forem usados.
type Reader struct {
	buf []byte
	pos int
}

// New cria um leitor sobre os bytes de uma mensagem.
func New(buf []byte) *Reader { return &Reader{buf: buf} }

// Done informa se a mensagem acabou.
func (r *Reader) Done() bool { return r.pos >= len(r.buf) }

// Pos devolve a posicao atual, util em mensagens de erro.
func (r *Reader) Pos() int { return r.pos }

// Len devolve quantos bytes ainda restam.
func (r *Reader) Len() int { return len(r.buf) - r.pos }

// Tag le a tag do proximo campo, devolvendo o numero e o tipo.
func (r *Reader) Tag() (field int32, typ Type, err error) {
	v, err := r.Varint()
	if err != nil {
		return 0, 0, err
	}

	num := v >> 3
	if num == 0 || num > 536870911 { // limite do protobuf: 2^29 - 1
		return 0, 0, fmt.Errorf("%w: numero de campo invalido (%d)", ErrMalformado, num)
	}
	return int32(num), Type(v & 7), nil
}

// Varint le um inteiro de tamanho variavel.
func (r *Reader) Varint() (uint64, error) {
	var v uint64
	var shift uint

	for i := 0; i < varintMaxBytes; i++ {
		if r.pos >= len(r.buf) {
			return 0, ErrTruncado
		}
		b := r.buf[r.pos]
		r.pos++

		v |= uint64(b&0x7f) << shift
		if b < 0x80 {
			return v, nil
		}
		shift += 7
	}
	return 0, fmt.Errorf("%w: varint longo demais", ErrMalformado)
}

// Int64 le um varint como inteiro com sinal.
func (r *Reader) Int64() (int64, error) {
	v, err := r.Varint()
	return int64(v), err
}

// Int32 le um varint como inteiro de 32 bits com sinal.
//
// Valores negativos sao gravados pelo protobuf como varints de 10 bytes com
// extensao de sinal em 64 bits, por isso a conversao passa por int64.
func (r *Reader) Int32() (int32, error) {
	v, err := r.Varint()
	return int32(int64(v)), err
}

// Bool le um varint como booleano.
func (r *Reader) Bool() (bool, error) {
	v, err := r.Varint()
	return v != 0, err
}

// SInt64 le um varint com codificacao zigzag.
//
// Zigzag intercala positivos e negativos (0, -1, 1, -2, 2, ...) para que
// numeros pequenos de qualquer sinal ocupem poucos bytes. Sem ela, -1 viraria
// um varint de 10 bytes. E o que o formato .osm.pbf usa para gravar as
// diferencas entre coordenadas consecutivas.
func (r *Reader) SInt64() (int64, error) {
	v, err := r.Varint()
	if err != nil {
		return 0, err
	}
	return int64(v>>1) ^ -int64(v&1), nil
}

// SInt32 le um varint zigzag de 32 bits.
func (r *Reader) SInt32() (int32, error) {
	v, err := r.SInt64()
	return int32(v), err
}

// Fixed32 le 4 bytes little-endian.
func (r *Reader) Fixed32() (uint32, error) {
	if r.pos+4 > len(r.buf) {
		return 0, ErrTruncado
	}
	v := binary.LittleEndian.Uint32(r.buf[r.pos:])
	r.pos += 4
	return v, nil
}

// Fixed64 le 8 bytes little-endian.
func (r *Reader) Fixed64() (uint64, error) {
	if r.pos+8 > len(r.buf) {
		return 0, ErrTruncado
	}
	v := binary.LittleEndian.Uint64(r.buf[r.pos:])
	r.pos += 8
	return v, nil
}

// Float le um float de 32 bits.
func (r *Reader) Float() (float32, error) {
	v, err := r.Fixed32()
	if err != nil {
		return 0, err
	}
	return math.Float32frombits(v), nil
}

// Double le um float de 64 bits.
func (r *Reader) Double() (float64, error) {
	v, err := r.Fixed64()
	if err != nil {
		return 0, err
	}
	return math.Float64frombits(v), nil
}

// Bytes le um campo delimitado por tamanho.
//
// O slice devolvido aponta para o buffer original -- nao ha copia. Nao
// escreva nele, e nao o guarde alem da vida do buffer.
func (r *Reader) Bytes() ([]byte, error) {
	n, err := r.Varint()
	if err != nil {
		return nil, err
	}
	// A conversao para int precisa ser conferida: num arquivo corrompido, um
	// tamanho absurdo estouraria o int em 32 bits e viraria negativo.
	if n > uint64(len(r.buf)-r.pos) {
		return nil, fmt.Errorf("%w: campo diz ter %d bytes, restam %d", ErrTruncado, n, len(r.buf)-r.pos)
	}
	b := r.buf[r.pos : r.pos+int(n)]
	r.pos += int(n)
	return b, nil
}

// String le um campo delimitado como texto. Copia, porque string em Go e
// imutavel e nao pode apontar para um buffer que muda.
func (r *Reader) String() (string, error) {
	b, err := r.Bytes()
	if err != nil {
		return "", err
	}
	return string(b), nil
}

// Message le um campo delimitado e devolve um leitor sobre ele. E assim que
// se desce numa mensagem aninhada.
func (r *Reader) Message() (*Reader, error) {
	b, err := r.Bytes()
	if err != nil {
		return nil, err
	}
	return New(b), nil
}

// Skip pula o campo cujo tipo foi dado, para depois da leitura da tag.
//
// E o que torna o leitor tolerante a versoes futuras do esquema: campos
// desconhecidos sao pulados sem que seja preciso saber o que significam.
func (r *Reader) Skip(typ Type) error {
	switch typ {
	case Varint:
		_, err := r.Varint()
		return err

	case Fixed64:
		_, err := r.Fixed64()
		return err

	case Bytes:
		_, err := r.Bytes()
		return err

	case Fixed32:
		_, err := r.Fixed32()
		return err

	case StartGroup:
		// Grupos sao descontinuados desde o proto2, mas pular um custa pouco:
		// le-se ate o EndGroup correspondente, respeitando o aninhamento.
		for {
			_, t, err := r.Tag()
			if err != nil {
				return err
			}
			if t == EndGroup {
				return nil
			}
			if err := r.Skip(t); err != nil {
				return err
			}
		}

	case EndGroup:
		return fmt.Errorf("%w: EndGroup sem StartGroup", ErrMalformado)
	}

	return fmt.Errorf("%w: tipo de campo desconhecido (%d)", ErrMalformado, uint8(typ))
}

// PackedFloats le um campo delimitado como sequencia de float32.
//
// Campos numericos repetidos vem "empacotados": um unico campo delimitado
// com todos os valores em sequencia, sem tag entre eles. Anexa a dst, que
// pode vir nil.
func (r *Reader) PackedFloats(dst []float32) ([]float32, error) {
	b, err := r.Bytes()
	if err != nil {
		return dst, err
	}
	if len(b)%4 != 0 {
		return dst, fmt.Errorf("%w: %d bytes nao dividem em float32", ErrMalformado, len(b))
	}

	for i := 0; i < len(b); i += 4 {
		dst = append(dst, math.Float32frombits(binary.LittleEndian.Uint32(b[i:])))
	}
	return dst, nil
}

// PackedDoubles le um campo delimitado como sequencia de float64.
func (r *Reader) PackedDoubles(dst []float64) ([]float64, error) {
	b, err := r.Bytes()
	if err != nil {
		return dst, err
	}
	if len(b)%8 != 0 {
		return dst, fmt.Errorf("%w: %d bytes nao dividem em float64", ErrMalformado, len(b))
	}

	for i := 0; i < len(b); i += 8 {
		dst = append(dst, math.Float64frombits(binary.LittleEndian.Uint64(b[i:])))
	}
	return dst, nil
}

// PackedVarints le um campo delimitado como sequencia de varints.
func (r *Reader) PackedVarints(dst []uint64) ([]uint64, error) {
	b, err := r.Bytes()
	if err != nil {
		return dst, err
	}

	sub := New(b)
	for !sub.Done() {
		v, err := sub.Varint()
		if err != nil {
			return dst, err
		}
		dst = append(dst, v)
	}
	return dst, nil
}

// PackedInt64s le um campo delimitado como sequencia de int64.
func (r *Reader) PackedInt64s(dst []int64) ([]int64, error) {
	b, err := r.Bytes()
	if err != nil {
		return dst, err
	}

	sub := New(b)
	for !sub.Done() {
		v, err := sub.Varint()
		if err != nil {
			return dst, err
		}
		dst = append(dst, int64(v))
	}
	return dst, nil
}

// PackedSInt64s le um campo delimitado como sequencia de varints zigzag.
func (r *Reader) PackedSInt64s(dst []int64) ([]int64, error) {
	b, err := r.Bytes()
	if err != nil {
		return dst, err
	}

	sub := New(b)
	for !sub.Done() {
		v, err := sub.SInt64()
		if err != nil {
			return dst, err
		}
		dst = append(dst, v)
	}
	return dst, nil
}

// PackedInt32s le um campo delimitado como sequencia de int32.
func (r *Reader) PackedInt32s(dst []int32) ([]int32, error) {
	b, err := r.Bytes()
	if err != nil {
		return dst, err
	}

	sub := New(b)
	for !sub.Done() {
		v, err := sub.Varint()
		if err != nil {
			return dst, err
		}
		dst = append(dst, int32(int64(v)))
	}
	return dst, nil
}
