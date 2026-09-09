package ch

import (
	"bytes"
	"compress/zlib"
	"encoding/binary"
	"math"
	"sort"
	"testing"

	"github.com/iafrotamacedo-cloud/era/internal/protowire"
	"github.com/iafrotamacedo-cloud/era/maps/geo"
	"github.com/iafrotamacedo-cloud/era/maps/graph"
)

// Escreve .osm.pbf minusculos para os testes montarem grafos de verdade.
//
// O maps/graph so sabe construir a partir do formato, e nao expoe um
// construtor a mao -- de proposito: seria um caminho na API publica que
// producao nenhuma deveria usar. Entao o teste paga o preco de escrever o
// arquivo, que e o mesmo que o maps/graph paga nos testes dele.

type noPbf struct {
	id int64
	p  geo.Point
}

type viaPbf struct {
	id   int64
	refs []int64
	tags map[string]string
}

type arquivoPbf struct{ buf bytes.Buffer }

func (a *arquivoPbf) blob(tipo string, corpo []byte) {
	var z bytes.Buffer
	zw := zlib.NewWriter(&z)
	if _, err := zw.Write(corpo); err != nil {
		panic(err)
	}
	if err := zw.Close(); err != nil {
		panic(err)
	}

	b := protowire.NewWriter()
	b.Int64Field(2, int64(len(corpo)))
	b.BytesField(3, z.Bytes())

	cab := protowire.NewWriter()
	cab.StringField(1, tipo)
	cab.Int32Field(3, int32(len(b.Bytes())))

	var tam [4]byte
	binary.BigEndian.PutUint32(tam[:], uint32(len(cab.Bytes())))
	a.buf.Write(tam[:])
	a.buf.Write(cab.Bytes())
	a.buf.Write(b.Bytes())
}

type tabelaPbf struct {
	strs   []string
	indice map[string]int32
}

func novaTabelaPbf() *tabelaPbf {
	return &tabelaPbf{strs: []string{""}, indice: map[string]int32{"": 0}}
}

func (t *tabelaPbf) id(s string) int32 {
	if i, ok := t.indice[s]; ok {
		return i
	}
	i := int32(len(t.strs))
	t.strs = append(t.strs, s)
	t.indice[s] = i
	return i
}

func (t *tabelaPbf) escrever(w *protowire.Writer) {
	st := protowire.NewWriter()
	for _, s := range t.strs {
		st.StringField(1, s)
	}
	w.MessageField(1, st)
}

func montarGrafo(t testing.TB, nos []noPbf, vias []viaPbf) *graph.Graph {
	t.Helper()
	var a arquivoPbf

	cab := protowire.NewWriter()
	cab.StringField(4, "OsmSchema-V0.6")
	cab.StringField(4, "DenseNodes")
	a.blob("OSMHeader", cab.Bytes())

	// Os nos densos exigem identificadores crescentes, porque as diferencas
	// sao codificadas em zigzag e as ferramentas reais sempre os ordenam.
	ordenados := append([]noPbf(nil), nos...)
	sort.Slice(ordenados, func(i, j int) bool { return ordenados[i].id < ordenados[j].id })

	tn := novaTabelaPbf()
	var ids, lats, lons []int64
	var kv []int32
	var antID, antLat, antLon int64
	for _, n := range ordenados {
		lat := int64(math.Round(n.p.Lat * 1e7))
		lon := int64(math.Round(n.p.Lon * 1e7))
		ids = append(ids, n.id-antID)
		lats = append(lats, lat-antLat)
		lons = append(lons, lon-antLon)
		antID, antLat, antLon = n.id, lat, lon
		kv = append(kv, 0)
	}

	dense := protowire.NewWriter()
	dense.PackedSInt64sField(1, ids)
	dense.PackedSInt64sField(8, lats)
	dense.PackedSInt64sField(9, lons)
	dense.PackedInt32sField(10, kv)

	grupoNos := protowire.NewWriter()
	grupoNos.MessageField(2, dense)

	wn := protowire.NewWriter()
	tn.escrever(wn)
	wn.MessageField(2, grupoNos)
	a.blob("OSMData", wn.Bytes())

	tv := novaTabelaPbf()
	grupoVias := protowire.NewWriter()
	for _, v := range vias {
		chaves := make([]string, 0, len(v.tags))
		for k := range v.tags {
			chaves = append(chaves, k)
		}
		sort.Strings(chaves)

		var ks, vs []int32
		for _, k := range chaves {
			ks = append(ks, tv.id(k))
			vs = append(vs, tv.id(v.tags[k]))
		}

		var refs []int64
		var ant int64
		for _, r := range v.refs {
			refs = append(refs, r-ant)
			ant = r
		}

		w := protowire.NewWriter()
		w.Int64Field(1, v.id)
		w.PackedInt32sField(2, ks)
		w.PackedInt32sField(3, vs)
		w.PackedSInt64sField(8, refs)
		grupoVias.MessageField(3, w)
	}

	wv := protowire.NewWriter()
	tv.escrever(wv)
	wv.MessageField(2, grupoVias)
	a.blob("OSMData", wv.Bytes())

	g, err := graph.Build(bytes.NewReader(a.buf.Bytes()), graph.Car())
	if err != nil {
		t.Fatalf("graph.Build: %v", err)
	}
	return g
}

func pontoEm(lat, lon float64) geo.Point { return geo.Point{Lat: lat, Lon: lon} }
