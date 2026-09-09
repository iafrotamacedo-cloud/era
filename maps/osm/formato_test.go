package osm

import (
	"bytes"
	"compress/zlib"
	"errors"
	"math"
	"testing"

	"github.com/iafrotamacedo-cloud/era/internal/protowire"
	"github.com/iafrotamacedo-cloud/era/maps/geo"
)

// Este arquivo cobre os caminhos do formato que um extrato do Geofabrik nunca
// exercitaria -- e que por isso quebrariam em silencio no dia em que
// aparecesse um arquivo gerado por outra ferramenta.

// blocoNosAvulsos escreve nos no formato antigo, um por mensagem, em vez de
// DenseNodes.
func (a *arquivo) blocoNosAvulsos(nos []Node) {
	t := novaTabela()
	grupo := protowire.NewWriter()

	for _, n := range nos {
		var chaves, valores []int32
		for _, tag := range n.Tags {
			chaves = append(chaves, t.id(tag.Key))
			valores = append(valores, t.id(tag.Value))
		}

		no := protowire.NewWriter()
		no.SInt64Field(1, n.ID)
		no.PackedInt32sField(2, chaves)
		no.PackedInt32sField(3, valores)
		no.SInt64Field(8, paraInteiro(n.Point.Lat))
		no.SInt64Field(9, paraInteiro(n.Point.Lon))
		grupo.MessageField(1, no)
	}

	w := protowire.NewWriter()
	t.escrever(w)
	w.MessageField(2, grupo)
	a.blob(blocoDados, w.Bytes())
}

// Praticamente todo extrato publicado usa DenseNodes, mas o formato admite
// nos avulsos e ha ferramentas que os geram.
func TestNosAvulsos(t *testing.T) {
	nos := []Node{
		{ID: 1, Point: fortaleza, Tags: Tags{{Key: "amenity", Value: "fuel"}}},
		{ID: 2, Point: geo.Point{Lat: -3.8, Lon: -38.6}},
		{ID: 3, Point: geo.Point{Lat: -3.9, Lon: -38.7}, Tags: Tags{
			{Key: "highway", Value: "bus_stop"}, {Key: "name", Value: "Parada Central"},
		}},
	}

	a := arquivoPadrao()
	a.blocoNosAvulsos(nos)
	c := coletar(t, a.bytes(), 2)

	if len(c.nos) != len(nos) {
		t.Fatalf("leu %d nos, escreveu %d", len(c.nos), len(nos))
	}
	for i, n := range nos {
		if c.nos[i].ID != n.ID || !pontosIguais(c.nos[i].Point, n.Point) {
			t.Errorf("no %d: %+v, esperado %+v", i, c.nos[i], n)
		}
		if len(c.nos[i].Tags) != len(n.Tags) {
			t.Errorf("no %d: %d etiquetas, esperado %d", i, len(c.nos[i].Tags), len(n.Tags))
		}
	}
}

// blocoViasRepetidas escreve as referencias como campos repetidos em vez de
// empacotados. O protobuf admite as duas formas e arquivos antigos usam esta.
func (a *arquivo) blocoViasRepetidas(vias []Way) {
	t := novaTabela()
	grupo := protowire.NewWriter()

	for _, v := range vias {
		way := protowire.NewWriter()
		way.Int64Field(1, v.ID)
		for _, tag := range v.Tags {
			way.Varint(uint64(2)<<3 | uint64(protowire.Varint))
			way.Varint(uint64(t.id(tag.Key)))
		}
		for _, tag := range v.Tags {
			way.Varint(uint64(3)<<3 | uint64(protowire.Varint))
			way.Varint(uint64(t.id(tag.Value)))
		}
		var ant int64
		for _, r := range v.Refs {
			way.SInt64Field(8, r-ant)
			ant = r
		}
		grupo.MessageField(3, way)
	}

	w := protowire.NewWriter()
	t.escrever(w)
	w.MessageField(2, grupo)
	a.blob(blocoDados, w.Bytes())
}

func TestListasRepetidasEmVezDeEmpacotadas(t *testing.T) {
	vias := []Way{
		{ID: 1, Refs: []int64{100, 101, 105, 200}, Tags: Tags{{Key: "highway", Value: "track"}}},
		{ID: 2, Refs: []int64{7, 8}},
	}

	a := arquivoPadrao()
	a.blocoViasRepetidas(vias)
	c := coletar(t, a.bytes(), 1)

	if len(c.vias) != len(vias) {
		t.Fatalf("leu %d vias, escreveu %d", len(c.vias), len(vias))
	}
	for i, v := range vias {
		if c.vias[i].ID != v.ID {
			t.Errorf("via %d: id %d, esperado %d", i, c.vias[i].ID, v.ID)
		}
		if len(c.vias[i].Refs) != len(v.Refs) {
			t.Fatalf("via %d: %d refs, esperado %d", i, len(c.vias[i].Refs), len(v.Refs))
		}
		for j := range v.Refs {
			if c.vias[i].Refs[j] != v.Refs[j] {
				t.Errorf("via %d, ref %d: %d, esperado %d", i, j, c.vias[i].Refs[j], v.Refs[j])
			}
		}
	}
}

// blocoComEscala escreve um bloco com granularidade e deslocamentos fora do
// padrao. Extratos regionais costumam trazer deslocamento para que as
// diferencas de coordenada fiquem ainda menores.
func (a *arquivo) blocoComEscala(nos []Node, granularidade int32, latOffset, lonOffset int64) {
	t := novaTabela()

	var ids, lats, lons []int64
	var antID, antLat, antLon int64
	for _, n := range nos {
		lat := int64(math.Round((n.Point.Lat*1e9 - float64(latOffset)) / float64(granularidade)))
		lon := int64(math.Round((n.Point.Lon*1e9 - float64(lonOffset)) / float64(granularidade)))
		ids = append(ids, n.ID-antID)
		lats = append(lats, lat-antLat)
		lons = append(lons, lon-antLon)
		antID, antLat, antLon = n.ID, lat, lon
	}

	dense := protowire.NewWriter()
	dense.PackedSInt64sField(1, ids)
	dense.PackedSInt64sField(8, lats)
	dense.PackedSInt64sField(9, lons)

	grupo := protowire.NewWriter()
	grupo.MessageField(2, dense)

	w := protowire.NewWriter()
	t.escrever(w)
	w.MessageField(2, grupo)
	w.Int32Field(17, granularidade)
	w.Int64Field(19, latOffset)
	w.Int64Field(20, lonOffset)

	a.blob(blocoDados, w.Bytes())
}

func TestGranularidadeEDeslocamentoNaoPadrao(t *testing.T) {
	nos := []Node{
		{ID: 1, Point: fortaleza},
		{ID: 2, Point: geo.Point{Lat: -3.7400, Lon: -38.5300}},
		{ID: 3, Point: geo.Point{Lat: -3.7500, Lon: -38.5400}},
	}

	a := arquivoPadrao()
	a.blocoComEscala(nos, 10, -3_700_000_000, -38_500_000_000)
	c := coletar(t, a.bytes(), 1)

	if len(c.nos) != len(nos) {
		t.Fatalf("leu %d nos, escreveu %d", len(c.nos), len(nos))
	}
	for i, n := range nos {
		if !pontosIguais(c.nos[i].Point, n.Point) {
			t.Errorf("no %d: %v, esperado %v", i, c.nos[i].Point, n.Point)
		}
	}
}

// Granularidade zero zeraria toda coordenada do bloco. Melhor recusar o
// arquivo do que devolver um mapa inteiro em cima da ilha de Null.
func TestGranularidadeZeroFalha(t *testing.T) {
	a := arquivoPadrao()

	dense := protowire.NewWriter()
	dense.PackedSInt64sField(1, []int64{1})
	dense.PackedSInt64sField(8, []int64{1})
	dense.PackedSInt64sField(9, []int64{1})
	grupo := protowire.NewWriter()
	grupo.MessageField(2, dense)

	w := protowire.NewWriter()
	novaTabela().escrever(w)
	w.MessageField(2, grupo)
	w.Int32Field(17, 0)
	a.blob(blocoDados, w.Bytes())

	err := Scan(bytes.NewReader(a.bytes()), Handler{Node: func(Node) error { return nil }})
	if !errors.Is(err, ErrFormato) {
		t.Errorf("erro = %v, esperado ErrFormato", err)
	}
}

// Uma etiqueta que aponta para fora da tabela de strings e arquivo
// corrompido. Sem a checagem, seria um panic de indice em vez de um erro.
func TestIndiceDeTabelaForaDaFaixa(t *testing.T) {
	a := arquivoPadrao()

	dense := protowire.NewWriter()
	dense.PackedSInt64sField(1, []int64{1})
	dense.PackedSInt64sField(8, []int64{paraInteiro(-3.73)})
	dense.PackedSInt64sField(9, []int64{paraInteiro(-38.52)})
	dense.PackedInt32sField(10, []int32{999, 1000, 0}) // tabela so tem 1 entrada
	grupo := protowire.NewWriter()
	grupo.MessageField(2, dense)

	w := protowire.NewWriter()
	novaTabela().escrever(w)
	w.MessageField(2, grupo)
	a.blob(blocoDados, w.Bytes())

	err := Scan(bytes.NewReader(a.bytes()), Handler{Node: func(Node) error { return nil }})
	if !errors.Is(err, ErrFormato) {
		t.Errorf("erro = %v, esperado ErrFormato", err)
	}
}

// Listas de tamanhos diferentes -- 3 ids para 2 latitudes -- sao arquivo
// corrompido, e o leitor precisa dizer isso em vez de ler pela metade.
func TestListasDesalinhadasFalham(t *testing.T) {
	a := arquivoPadrao()

	dense := protowire.NewWriter()
	dense.PackedSInt64sField(1, []int64{1, 2, 3})
	dense.PackedSInt64sField(8, []int64{1, 2})
	dense.PackedSInt64sField(9, []int64{1, 2, 3})
	grupo := protowire.NewWriter()
	grupo.MessageField(2, dense)

	w := protowire.NewWriter()
	novaTabela().escrever(w)
	w.MessageField(2, grupo)
	a.blob(blocoDados, w.Bytes())

	err := Scan(bytes.NewReader(a.bytes()), Handler{Node: func(Node) error { return nil }})
	if !errors.Is(err, ErrFormato) {
		t.Errorf("erro = %v, esperado ErrFormato", err)
	}
}

// A ERA nao tem dependencias, e lzma, lz4 e zstd exigiriam uma. Encontrar um
// deles precisa ser erro nomeado, para quem receber o arquivo saber
// recomprimi-lo em vez de achar que o arquivo esta corrompido.
func TestCompressaoNaoSuportada(t *testing.T) {
	for _, caso := range []struct {
		campo int32
		nome  string
	}{{4, "lzma"}, {6, "lz4"}, {7, "zstd"}} {
		var buf bytes.Buffer

		corpo := protowire.NewWriter()
		corpo.Int64Field(2, 100)
		corpo.BytesField(caso.campo, []byte("dados comprimidos de outro jeito"))

		cab := protowire.NewWriter()
		cab.StringField(1, blocoCabecalho)
		cab.Int32Field(3, int32(len(corpo.Bytes())))

		var tam [4]byte
		tam[0], tam[1] = 0, 0
		tam[2] = byte(len(cab.Bytes()) >> 8)
		tam[3] = byte(len(cab.Bytes()))
		buf.Write(tam[:])
		buf.Write(cab.Bytes())
		buf.Write(corpo.Bytes())

		err := Scan(bytes.NewReader(buf.Bytes()), Handler{Node: func(Node) error { return nil }})
		if !errors.Is(err, ErrNaoSuportado) {
			t.Errorf("%s: erro = %v, esperado ErrNaoSuportado", caso.nome, err)
		}
	}
}

// Um bloco que promete mais bytes descomprimidos do que entrega e arquivo
// corrompido -- e sem a checagem viraria leitura de lixo.
func TestTamanhoDescomprimidoMentiroso(t *testing.T) {
	a := arquivoPadrao()
	dados := a.bytes()

	// Monta a mao um blob cujo raw_size nao corresponde ao conteudo.
	var buf bytes.Buffer
	buf.Write(dados) // cabecalho valido primeiro

	corpo := protowire.NewWriter()
	corpo.Int64Field(2, 1_000_000) // promete 1 MB
	var z bytes.Buffer
	zw := zlib.NewWriter(&z)
	if _, err := zw.Write([]byte("poucos bytes")); err != nil {
		t.Fatal(err)
	}
	if err := zw.Close(); err != nil {
		t.Fatal(err)
	}
	corpo.BytesField(3, z.Bytes())

	cab := protowire.NewWriter()
	cab.StringField(1, blocoDados)
	cab.Int32Field(3, int32(len(corpo.Bytes())))

	var tam [4]byte
	tam[2] = byte(len(cab.Bytes()) >> 8)
	tam[3] = byte(len(cab.Bytes()))
	buf.Write(tam[:])
	buf.Write(cab.Bytes())
	buf.Write(corpo.Bytes())

	err := Scan(bytes.NewReader(buf.Bytes()), Handler{Node: func(Node) error { return nil }})
	if !errors.Is(err, ErrFormato) {
		t.Errorf("erro = %v, esperado ErrFormato", err)
	}
}

// Tipo de bloco desconhecido e ignorado de proposito: o formato preve que
// versoes futuras acrescentem tipos, e um leitor que quebra por isso
// envelhece mal.
func TestTipoDeBlocoDesconhecidoEhIgnorado(t *testing.T) {
	a := arquivoPadrao()
	a.blob("OSMAlgoQueAindaNaoExiste", []byte("conteudo que este leitor nao entende"))
	a.blocoDenso(nosDeTeste(10))

	c := coletar(t, a.bytes(), 2)
	if len(c.nos) != 10 {
		t.Errorf("leu %d nos, esperava 10 -- o bloco desconhecido atrapalhou", len(c.nos))
	}
}

// Um bloco com nos, vias e relacoes juntos: o formato permite, e o leitor
// precisa entregar os tres na ordem certa.
func TestBlocoMisto(t *testing.T) {
	t2 := novaTabela()
	grupo := protowire.NewWriter()

	dense := protowire.NewWriter()
	dense.PackedSInt64sField(1, []int64{1, 1})
	dense.PackedSInt64sField(8, []int64{paraInteiro(-3.73), 100})
	dense.PackedSInt64sField(9, []int64{paraInteiro(-38.52), 100})
	grupo.MessageField(2, dense)

	way := protowire.NewWriter()
	way.Int64Field(1, 50)
	way.PackedSInt64sField(8, []int64{1, 1})
	way.PackedInt32sField(2, []int32{t2.id("highway")})
	way.PackedInt32sField(3, []int32{t2.id("service")})
	grupo.MessageField(3, way)

	rel := protowire.NewWriter()
	rel.Int64Field(1, 900)
	rel.PackedInt32sField(8, []int32{t2.id("outer")})
	rel.PackedSInt64sField(9, []int64{50})
	rel.PackedInt32sField(10, []int32{int32(MemberWay)})
	grupo.MessageField(4, rel)

	w := protowire.NewWriter()
	t2.escrever(w)
	w.MessageField(2, grupo)

	a := arquivoPadrao()
	a.blob(blocoDados, w.Bytes())

	c := coletar(t, a.bytes(), 1)
	if len(c.nos) != 2 || len(c.vias) != 1 || len(c.relacoes) != 1 {
		t.Fatalf("leu %d nos, %d vias, %d relacoes; esperava 2, 1, 1",
			len(c.nos), len(c.vias), len(c.relacoes))
	}
	if c.vias[0].ID != 50 || len(c.vias[0].Refs) != 2 || c.vias[0].Refs[1] != 2 {
		t.Errorf("via = %+v", c.vias[0])
	}
	if c.relacoes[0].Members[0].Type != MemberWay {
		t.Errorf("membro = %+v", c.relacoes[0].Members[0])
	}
}

// Tipo de membro fora dos tres que existem e arquivo corrompido.
func TestTipoDeMembroInvalido(t *testing.T) {
	t2 := novaTabela()
	rel := protowire.NewWriter()
	rel.Int64Field(1, 1)
	rel.PackedInt32sField(8, []int32{t2.id("outer")})
	rel.PackedSInt64sField(9, []int64{1})
	rel.PackedInt32sField(10, []int32{9}) // nao existe

	grupo := protowire.NewWriter()
	grupo.MessageField(4, rel)
	w := protowire.NewWriter()
	t2.escrever(w)
	w.MessageField(2, grupo)

	a := arquivoPadrao()
	a.blob(blocoDados, w.Bytes())

	err := Scan(bytes.NewReader(a.bytes()), Handler{Relation: func(Relation) error { return nil }})
	if !errors.Is(err, ErrFormato) {
		t.Errorf("erro = %v, esperado ErrFormato", err)
	}
}
