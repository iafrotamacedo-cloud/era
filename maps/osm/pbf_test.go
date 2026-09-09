package osm

import (
	"bytes"
	"compress/zlib"
	"encoding/binary"
	"math"
	"testing"

	"github.com/iafrotamacedo-cloud/era/internal/protowire"
	"github.com/iafrotamacedo-cloud/era/maps/geo"
)

// Este arquivo monta .osm.pbf validos para os testes lerem.
//
// Existe porque o repositorio nao guarda dados de mapa: um extrato, mesmo de
// um municipio, tem dezenas de MB e licenca propria. Sem um escritor, testar
// o leitor exigiria baixar arquivo da internet no CI -- o que torna o teste
// lento, dependente de rede e quebravel por decisao de terceiro.
//
// E o mesmo movimento que o faces fez: o protowire ganhou um escritor para
// que o parser de .onnx pudesse ser testado sem baixar um modelo.
//
// A vantagem escondida e outra: com um escritor da para produzir arquivos que
// nao existiriam naturalmente -- truncados, com compressao exotica, com
// indice de tabela apontando para o vazio -- e conferir que o leitor recusa
// cada um deles com erro claro.

// arquivo monta um .osm.pbf bloco a bloco.
type arquivo struct {
	buf        bytes.Buffer
	comprimido bool
}

func novoArquivo() *arquivo { return &arquivo{comprimido: true} }

func (a *arquivo) bytes() []byte { return a.buf.Bytes() }

// blob acrescenta um bloco ja serializado, com o envelope do formato:
// tamanho do cabecalho em 4 bytes big-endian, cabecalho, corpo.
func (a *arquivo) blob(tipo string, corpo []byte) {
	b := protowire.NewWriter()
	if a.comprimido {
		var z bytes.Buffer
		zw := zlib.NewWriter(&z)
		if _, err := zw.Write(corpo); err != nil {
			panic(err)
		}
		if err := zw.Close(); err != nil {
			panic(err)
		}
		b.Int64Field(2, int64(len(corpo))) // raw_size
		b.BytesField(3, z.Bytes())         // zlib_data
	} else {
		b.BytesField(1, corpo) // raw
	}
	corpoSerializado := b.Bytes()

	cab := protowire.NewWriter()
	cab.StringField(1, tipo)
	cab.Int32Field(3, int32(len(corpoSerializado)))

	var tam [4]byte
	binary.BigEndian.PutUint32(tam[:], uint32(len(cab.Bytes())))
	a.buf.Write(tam[:])
	a.buf.Write(cab.Bytes())
	a.buf.Write(corpoSerializado)
}

// cabecalho escreve o bloco OSMHeader.
func (a *arquivo) cabecalho(recursos []string, bbox *geo.Box) {
	w := protowire.NewWriter()

	if bbox != nil {
		bb := protowire.NewWriter()
		bb.SInt64Field(1, int64(math.Round(bbox.MinLon*1e9))) // left
		bb.SInt64Field(2, int64(math.Round(bbox.MaxLon*1e9))) // right
		bb.SInt64Field(3, int64(math.Round(bbox.MaxLat*1e9))) // top
		bb.SInt64Field(4, int64(math.Round(bbox.MinLat*1e9))) // bottom
		w.MessageField(1, bb)
	}
	for _, r := range recursos {
		w.StringField(4, r)
	}
	w.StringField(16, "era-mapas-teste")

	a.blob(blocoCabecalho, w.Bytes())
}

// tabela monta a tabela de strings de um bloco.
//
// O indice 0 e sempre vazio: e o valor que separa um no do proximo na lista
// achatada de etiquetas dos DenseNodes.
type tabela struct {
	strs   []string
	indice map[string]int32
}

func novaTabela() *tabela {
	return &tabela{strs: []string{""}, indice: map[string]int32{"": 0}}
}

func (t *tabela) id(s string) int32 {
	if i, ok := t.indice[s]; ok {
		return i
	}
	i := int32(len(t.strs))
	t.strs = append(t.strs, s)
	t.indice[s] = i
	return i
}

func (t *tabela) escrever(w *protowire.Writer) {
	st := protowire.NewWriter()
	for _, s := range t.strs {
		st.StringField(1, s)
	}
	w.MessageField(1, st)
}

// coordenadas convertidas para o inteiro que o formato guarda, com a
// granularidade padrao de 100 nanograus.
func paraInteiro(graus float64) int64 { return int64(math.Round(graus * 1e7)) }

// blocoDenso escreve um OSMData com os nos em DenseNodes, que e como todo
// extrato publicado guarda seus nos.
func (a *arquivo) blocoDenso(nos []Node) {
	t := novaTabela()

	var ids, lats, lons []int64
	var chavesValores []int32
	var antID, antLat, antLon int64

	for _, n := range nos {
		id := n.ID
		lat := paraInteiro(n.Point.Lat)
		lon := paraInteiro(n.Point.Lon)

		ids = append(ids, id-antID)
		lats = append(lats, lat-antLat)
		lons = append(lons, lon-antLon)
		antID, antLat, antLon = id, lat, lon

		for _, tag := range n.Tags {
			chavesValores = append(chavesValores, t.id(tag.Key), t.id(tag.Value))
		}
		chavesValores = append(chavesValores, 0) // separador
	}

	dense := protowire.NewWriter()
	dense.PackedSInt64sField(1, ids)
	dense.PackedSInt64sField(8, lats)
	dense.PackedSInt64sField(9, lons)
	dense.PackedInt32sField(10, chavesValores)

	grupo := protowire.NewWriter()
	grupo.MessageField(2, dense)

	w := protowire.NewWriter()
	t.escrever(w)
	w.MessageField(2, grupo)

	a.blob(blocoDados, w.Bytes())
}

// blocoVias escreve um OSMData com vias.
func (a *arquivo) blocoVias(vias []Way) {
	t := novaTabela()
	grupo := protowire.NewWriter()

	for _, v := range vias {
		var chaves, valores []int32
		for _, tag := range v.Tags {
			chaves = append(chaves, t.id(tag.Key))
			valores = append(valores, t.id(tag.Value))
		}

		var refs []int64
		var ant int64
		for _, r := range v.Refs {
			refs = append(refs, r-ant)
			ant = r
		}

		way := protowire.NewWriter()
		way.Int64Field(1, v.ID)
		way.PackedInt32sField(2, chaves)
		way.PackedInt32sField(3, valores)
		way.PackedSInt64sField(8, refs)
		grupo.MessageField(3, way)
	}

	w := protowire.NewWriter()
	t.escrever(w)
	w.MessageField(2, grupo)

	a.blob(blocoDados, w.Bytes())
}

// blocoRelacoes escreve um OSMData com relacoes.
func (a *arquivo) blocoRelacoes(rels []Relation) {
	t := novaTabela()
	grupo := protowire.NewWriter()

	for _, rel := range rels {
		var chaves, valores []int32
		for _, tag := range rel.Tags {
			chaves = append(chaves, t.id(tag.Key))
			valores = append(valores, t.id(tag.Value))
		}

		var papeis []int32
		var memids []int64
		var tipos []int32
		var ant int64
		for _, m := range rel.Members {
			papeis = append(papeis, t.id(m.Role))
			memids = append(memids, m.ID-ant)
			ant = m.ID
			tipos = append(tipos, int32(m.Type))
		}

		r := protowire.NewWriter()
		r.Int64Field(1, rel.ID)
		r.PackedInt32sField(2, chaves)
		r.PackedInt32sField(3, valores)
		r.PackedInt32sField(8, papeis)
		r.PackedSInt64sField(9, memids)
		r.PackedInt32sField(10, tipos)
		grupo.MessageField(4, r)
	}

	w := protowire.NewWriter()
	t.escrever(w)
	w.MessageField(2, grupo)

	a.blob(blocoDados, w.Bytes())
}

// arquivoPadrao monta um .osm.pbf pequeno e completo, com o cabecalho que
// todo extrato real traz.
func arquivoPadrao() *arquivo {
	a := novoArquivo()
	a.cabecalho([]string{"OsmSchema-V0.6", "DenseNodes"}, nil)
	return a
}

// coletar le um arquivo inteiro e devolve o que encontrou, com as fatias
// copiadas -- o leitor recicla os buffers entre blocos, entao guardar as
// fatias originais seria um erro que o proprio teste cometeria.
type coleta struct {
	cabecalhos []Header
	nos        []Node
	vias       []Way
	relacoes   []Relation
}

func coletar(t *testing.T, dados []byte, paralelismo int) coleta {
	t.Helper()
	var c coleta
	err := ScanN(bytes.NewReader(dados), Handler{
		Header: func(h Header) error { c.cabecalhos = append(c.cabecalhos, h); return nil },
		Node: func(n Node) error {
			n.Tags = append(Tags(nil), n.Tags...)
			c.nos = append(c.nos, n)
			return nil
		},
		Way: func(w Way) error {
			w.Refs = append([]int64(nil), w.Refs...)
			w.Tags = append(Tags(nil), w.Tags...)
			c.vias = append(c.vias, w)
			return nil
		},
		Relation: func(r Relation) error {
			r.Members = append([]Member(nil), r.Members...)
			r.Tags = append(Tags(nil), r.Tags...)
			c.relacoes = append(c.relacoes, r)
			return nil
		},
	}, paralelismo)
	if err != nil {
		t.Fatalf("Scan: %v", err)
	}
	return c
}
