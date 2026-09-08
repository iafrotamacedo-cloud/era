package protowire

import (
	"errors"
	"math"
	"reflect"
	"testing"
)

func TestVarintIdaEVolta(t *testing.T) {
	valores := []uint64{
		0, 1, 2, 127, 128, 129, 255, 256,
		16383, 16384, 1 << 20, 1 << 31, 1 << 32,
		math.MaxUint32, math.MaxInt64, math.MaxUint64,
	}

	for _, v := range valores {
		w := NewWriter()
		w.Varint(v)

		got, err := New(w.Bytes()).Varint()
		if err != nil {
			t.Fatalf("Varint(%d): %v", v, err)
		}
		if got != v {
			t.Errorf("gravei %d, li %d (%d bytes)", v, got, w.Len())
		}
	}
}

func TestVarintTamanhoEsperado(t *testing.T) {
	// O tamanho do varint e a razao de ele existir: numeros pequenos
	// ocupam pouco.
	casos := []struct {
		v     uint64
		bytes int
	}{
		{0, 1}, {1, 1}, {127, 1},
		{128, 2}, {16383, 2},
		{16384, 3},
		{math.MaxUint64, 10},
	}

	for _, c := range casos {
		w := NewWriter()
		w.Varint(c.v)
		if w.Len() != c.bytes {
			t.Errorf("%d ocupou %d bytes, quero %d", c.v, w.Len(), c.bytes)
		}
	}
}

func TestZigzag(t *testing.T) {
	// Zigzag existe para que numeros negativos pequenos nao virem varints
	// de 10 bytes. E o que o .osm.pbf usa nas diferencas de coordenadas.
	valores := []int64{0, -1, 1, -2, 2, -1000, 1000, math.MinInt64, math.MaxInt64}

	for _, v := range valores {
		w := NewWriter()
		w.SInt64Field(1, v)

		r := New(w.Bytes())
		if _, _, err := r.Tag(); err != nil {
			t.Fatal(err)
		}
		got, err := r.SInt64()
		if err != nil {
			t.Fatalf("SInt64(%d): %v", v, err)
		}
		if got != v {
			t.Errorf("gravei %d, li %d", v, got)
		}
	}

	// -1 tem que caber em 1 byte de payload, contra 10 sem zigzag.
	w := NewWriter()
	w.SInt64Field(1, -1)
	if w.Len() != 2 { // 1 tag + 1 payload
		t.Errorf("-1 em zigzag ocupou %d bytes, quero 2", w.Len())
	}
}

func TestTag(t *testing.T) {
	w := NewWriter()
	w.Tag(1, Varint)
	w.Tag(15, Bytes)
	w.Tag(16, Fixed32)
	w.Tag(2000, Fixed64)

	r := New(w.Bytes())
	esperado := []struct {
		field int32
		typ   Type
	}{
		{1, Varint}, {15, Bytes}, {16, Fixed32}, {2000, Fixed64},
	}

	for _, e := range esperado {
		f, tp, err := r.Tag()
		if err != nil {
			t.Fatal(err)
		}
		if f != e.field || tp != e.typ {
			t.Errorf("li campo %d tipo %v, quero %d %v", f, tp, e.field, e.typ)
		}
	}
}

func TestTiposEscalares(t *testing.T) {
	w := NewWriter()
	w.Int64Field(1, -42)
	w.Int32Field(2, -7)
	w.BoolField(3, true)
	w.FloatField(4, 3.5)
	w.DoubleField(5, -2.25)
	w.StringField(6, "arcface")

	r := New(w.Bytes())

	next := func() {
		if _, _, err := r.Tag(); err != nil {
			t.Fatal(err)
		}
	}

	next()
	if v, _ := r.Int64(); v != -42 {
		t.Errorf("Int64 = %d, quero -42", v)
	}
	next()
	if v, _ := r.Int32(); v != -7 {
		t.Errorf("Int32 = %d, quero -7", v)
	}
	next()
	if v, _ := r.Bool(); !v {
		t.Error("Bool = false, quero true")
	}
	next()
	if v, _ := r.Float(); v != 3.5 {
		t.Errorf("Float = %v, quero 3.5", v)
	}
	next()
	if v, _ := r.Double(); v != -2.25 {
		t.Errorf("Double = %v, quero -2.25", v)
	}
	next()
	if v, _ := r.String(); v != "arcface" {
		t.Errorf("String = %q, quero arcface", v)
	}

	if !r.Done() {
		t.Errorf("sobraram %d bytes", r.Len())
	}
}

func TestMensagemAninhada(t *testing.T) {
	interna := NewWriter()
	interna.StringField(1, "conv1")
	interna.Int64Field(2, 64)

	externa := NewWriter()
	externa.MessageField(3, interna)

	r := New(externa.Bytes())
	f, tp, err := r.Tag()
	if err != nil {
		t.Fatal(err)
	}
	if f != 3 || tp != Bytes {
		t.Fatalf("campo %d tipo %v, quero 3 bytes", f, tp)
	}

	sub, err := r.Message()
	if err != nil {
		t.Fatal(err)
	}

	sub.Tag()
	if s, _ := sub.String(); s != "conv1" {
		t.Errorf("nome = %q, quero conv1", s)
	}
	sub.Tag()
	if v, _ := sub.Int64(); v != 64 {
		t.Errorf("valor = %d, quero 64", v)
	}
}

func TestPackedFloats(t *testing.T) {
	vals := []float32{1, -2.5, 3.25, 0, 1e10, -1e-10}

	w := NewWriter()
	w.PackedFloatsField(1, vals)

	r := New(w.Bytes())
	r.Tag()

	got, err := r.PackedFloats(nil)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, vals) {
		t.Errorf("li %v, quero %v", got, vals)
	}
}

func TestPackedInteiros(t *testing.T) {
	vals := []int64{1, 2, 3, 128, 100000, -1}

	w := NewWriter()
	w.PackedInt64sField(1, vals)

	r := New(w.Bytes())
	r.Tag()

	got, err := r.PackedInt64s(nil)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, vals) {
		t.Errorf("li %v, quero %v", got, vals)
	}
}

func TestPackedSInt64s(t *testing.T) {
	vals := []int64{0, -1, 1, -100000, 100000}

	w := NewWriter()
	w.PackedSInt64sField(1, vals)

	r := New(w.Bytes())
	r.Tag()

	got, err := r.PackedSInt64s(nil)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, vals) {
		t.Errorf("li %v, quero %v", got, vals)
	}
}

// TestSkipCampoDesconhecido cobre a propriedade que permite ler um .onnx
// gerado por uma versao mais nova do formato: o que nao se conhece, se pula.
func TestSkipCampoDesconhecido(t *testing.T) {
	w := NewWriter()
	w.Int64Field(1, 111)     // desconhecido
	w.StringField(2, "lixo") // desconhecido
	w.FloatField(3, 1.5)     // desconhecido
	w.DoubleField(4, 2.5)    // desconhecido
	w.Int64Field(99, 42)     // o que interessa

	r := New(w.Bytes())
	var achado int64

	for !r.Done() {
		f, tp, err := r.Tag()
		if err != nil {
			t.Fatal(err)
		}
		if f == 99 {
			achado, err = r.Int64()
			if err != nil {
				t.Fatal(err)
			}
			continue
		}
		if err := r.Skip(tp); err != nil {
			t.Fatalf("Skip(%v): %v", tp, err)
		}
	}

	if achado != 42 {
		t.Errorf("achado = %d, quero 42", achado)
	}
}

func TestSkipGrupoDescontinuado(t *testing.T) {
	// Grupos sao descontinuados, mas um arquivo antigo pode traze-los.
	w := NewWriter()
	w.Tag(1, StartGroup)
	w.Int64Field(2, 7)
	w.StringField(3, "dentro")
	w.Tag(1, EndGroup)
	w.Int64Field(4, 99)

	r := New(w.Bytes())

	_, tp, err := r.Tag()
	if err != nil {
		t.Fatal(err)
	}
	if err := r.Skip(tp); err != nil {
		t.Fatalf("Skip(grupo): %v", err)
	}

	if _, _, err := r.Tag(); err != nil {
		t.Fatal(err)
	}
	if v, _ := r.Int64(); v != 99 {
		t.Errorf("depois do grupo li %d, quero 99", v)
	}
}

// TestDadosMalformados e o teste que mais importa numa biblioteca: um arquivo
// corrompido nao pode derrubar o processo de quem chama. Erro, sempre; panico,
// nunca.
func TestDadosMalformados(t *testing.T) {
	casos := []struct {
		nome string
		buf  []byte
	}{
		{"vazio no meio do varint", []byte{0x80}},
		{"varint sem fim", []byte{0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80}},
		{"tamanho maior que o buffer", []byte{0x0a, 0xff, 0x01}},
		{"fixed32 truncado", []byte{0x0d, 0x01, 0x02}},
		{"fixed64 truncado", []byte{0x09, 0x01, 0x02, 0x03}},
		{"campo zero", []byte{0x00}},
		{"tamanho absurdo", []byte{0x0a, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x7f}},
	}

	for _, c := range casos {
		t.Run(c.nome, func(t *testing.T) {
			defer func() {
				if p := recover(); p != nil {
					t.Fatalf("entrou em panico com dados invalidos: %v", p)
				}
			}()

			r := New(c.buf)
			var err error
			for !r.Done() && err == nil {
				var tp Type
				_, tp, err = r.Tag()
				if err != nil {
					break
				}
				err = r.Skip(tp)
			}
			if err == nil {
				t.Error("dados invalidos deveriam produzir erro")
			}
		})
	}
}

func TestErrosSaoIdentificaveis(t *testing.T) {
	// Quem chama precisa distinguir "arquivo cortado" de "arquivo corrompido".
	_, err := New([]byte{0x80}).Varint()
	if !errors.Is(err, ErrTruncado) {
		t.Errorf("erro = %v, quero ErrTruncado", err)
	}

	_, _, err = New([]byte{0x00}).Tag()
	if !errors.Is(err, ErrMalformado) {
		t.Errorf("erro = %v, quero ErrMalformado", err)
	}
}

func TestBytesNaoCopia(t *testing.T) {
	// Bytes devolve uma fatia do buffer original. E uma decisao de
	// desempenho com consequencia: quem chamar precisa saber.
	w := NewWriter()
	w.BytesField(1, []byte{1, 2, 3})
	buf := w.Bytes()

	r := New(buf)
	r.Tag()
	b, err := r.Bytes()
	if err != nil {
		t.Fatal(err)
	}

	b[0] = 99
	if buf[2] != 99 {
		t.Error("Bytes copiou; a documentacao promete que nao copia")
	}
}

func TestReaderVazio(t *testing.T) {
	r := New(nil)
	if !r.Done() {
		t.Error("leitor vazio deveria estar Done")
	}
	if r.Len() != 0 {
		t.Errorf("Len = %d, quero 0", r.Len())
	}
}

func TestTypeString(t *testing.T) {
	casos := map[Type]string{
		Varint: "varint", Fixed64: "fixed64", Bytes: "bytes",
		StartGroup: "start-group", EndGroup: "end-group", Fixed32: "fixed32",
	}
	for tp, quero := range casos {
		if got := tp.String(); got != quero {
			t.Errorf("Type(%d).String() = %q, quero %q", uint8(tp), got, quero)
		}
	}
	if got := Type(7).String(); got == "" {
		t.Error("tipo desconhecido deveria ter descricao")
	}
}
