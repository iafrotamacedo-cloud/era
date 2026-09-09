package osm

import (
	"bytes"
	"errors"
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/iafrotamacedo-cloud/era/maps/geo"
)

var fortaleza = geo.Point{Lat: -3.7319, Lon: -38.5267}

// A precisao do formato e de 100 nanograus, entao coordenada lida de volta
// nao bate bit a bit com a original -- bate ate a setima casa decimal, que e
// cerca de um centimetro.
const tolerancia = 1e-7

func pontosIguais(a, b geo.Point) bool {
	return math.Abs(a.Lat-b.Lat) < tolerancia && math.Abs(a.Lon-b.Lon) < tolerancia
}

func nosDeTeste(n int) []Node {
	r := rand.New(rand.NewSource(1))
	nos := make([]Node, n)
	for i := range nos {
		nos[i] = Node{
			ID:    int64(1000 + i*7), // ids crescentes, como num extrato real
			Point: geo.Destination(fortaleza, r.Float64()*360, r.Float64()*50_000),
		}
	}
	// Alguns com etiqueta, a maioria sem -- que e a proporcao real.
	if n > 3 {
		nos[1].Tags = Tags{{Key: "highway", Value: "traffic_signals"}}
		nos[3].Tags = Tags{
			{Key: "amenity", Value: "pharmacy"},
			{Key: "name", Value: "Farmacia do Bairro"},
		}
	}
	return nos
}

// O teste central: o que foi escrito volta igual.
func TestIdaEVoltaDeNos(t *testing.T) {
	nos := nosDeTeste(500)

	a := arquivoPadrao()
	a.blocoDenso(nos)

	c := coletar(t, a.bytes(), 4)

	if len(c.nos) != len(nos) {
		t.Fatalf("leu %d nos, escreveu %d", len(c.nos), len(nos))
	}
	for i, n := range nos {
		got := c.nos[i]
		if got.ID != n.ID {
			t.Fatalf("no %d: id %d, esperado %d", i, got.ID, n.ID)
		}
		if !pontosIguais(got.Point, n.Point) {
			t.Fatalf("no %d: %v, esperado %v", i, got.Point, n.Point)
		}
		if len(got.Tags) != len(n.Tags) {
			t.Fatalf("no %d: %d etiquetas, esperado %d", i, len(got.Tags), len(n.Tags))
		}
		for j, tag := range n.Tags {
			if got.Tags[j] != tag {
				t.Fatalf("no %d, etiqueta %d: %v, esperado %v", i, j, got.Tags[j], tag)
			}
		}
	}
}

// As coordenadas e os ids sao guardados como diferenca para o valor anterior.
// Se a soma acumulada estiver errada, o primeiro no sai certo e todos os
// seguintes saem deslocados -- um erro que so aparece com muitos nos.
func TestDiferencasAcumuladasNaoDerivam(t *testing.T) {
	const n = 5000
	nos := make([]Node, n)
	for i := range nos {
		nos[i] = Node{
			ID: int64(i + 1),
			// Uma linha reta, para o erro de deriva ficar obvio.
			Point: geo.Point{Lat: -3.0 - float64(i)*0.0001, Lon: -38.0 + float64(i)*0.0001},
		}
	}

	a := arquivoPadrao()
	a.blocoDenso(nos)
	c := coletar(t, a.bytes(), 2)

	if len(c.nos) != n {
		t.Fatalf("leu %d nos, escreveu %d", len(c.nos), n)
	}
	// O ultimo e o que mais sofreria com deriva acumulada.
	ultimo := c.nos[n-1]
	if ultimo.ID != int64(n) {
		t.Errorf("ultimo id = %d, esperado %d", ultimo.ID, n)
	}
	if !pontosIguais(ultimo.Point, nos[n-1].Point) {
		t.Errorf("ultimo ponto = %v, esperado %v", ultimo.Point, nos[n-1].Point)
	}
}

func TestIdaEVoltaDeVias(t *testing.T) {
	vias := []Way{
		{
			ID:   1,
			Refs: []int64{10, 11, 12, 13},
			Tags: Tags{{Key: "highway", Value: "residential"}, {Key: "name", Value: "Rua A"}},
		},
		{
			ID:   2,
			Refs: []int64{13, 500, 501},
			Tags: Tags{{Key: "highway", Value: "primary"}, {Key: "oneway", Value: "yes"}},
		},
		{ID: 3, Refs: []int64{999}}, // via degenerada, sem etiqueta
	}

	a := arquivoPadrao()
	a.blocoVias(vias)
	c := coletar(t, a.bytes(), 3)

	if len(c.vias) != len(vias) {
		t.Fatalf("leu %d vias, escreveu %d", len(c.vias), len(vias))
	}
	for i, v := range vias {
		got := c.vias[i]
		if got.ID != v.ID {
			t.Errorf("via %d: id %d, esperado %d", i, got.ID, v.ID)
		}
		if fmt.Sprint(got.Refs) != fmt.Sprint(v.Refs) {
			t.Errorf("via %d: refs %v, esperado %v", i, got.Refs, v.Refs)
		}
		if fmt.Sprint(got.Tags) != fmt.Sprint(v.Tags) {
			t.Errorf("via %d: etiquetas %v, esperado %v", i, got.Tags, v.Tags)
		}
	}
}

func TestIdaEVoltaDeRelacoes(t *testing.T) {
	rels := []Relation{
		{
			ID: 100,
			Members: []Member{
				{ID: 1, Type: MemberWay, Role: "from"},
				{ID: 2, Type: MemberNode, Role: "via"},
				{ID: 3, Type: MemberWay, Role: "to"},
			},
			Tags: Tags{
				{Key: "type", Value: "restriction"},
				{Key: "restriction", Value: "no_left_turn"},
			},
		},
		{
			ID:      101,
			Members: []Member{{ID: 100, Type: MemberRelation, Role: "subarea"}},
			Tags:    Tags{{Key: "type", Value: "boundary"}},
		},
	}

	a := arquivoPadrao()
	a.blocoRelacoes(rels)
	c := coletar(t, a.bytes(), 2)

	if len(c.relacoes) != len(rels) {
		t.Fatalf("leu %d relacoes, escreveu %d", len(c.relacoes), len(rels))
	}
	for i, rel := range rels {
		got := c.relacoes[i]
		if got.ID != rel.ID {
			t.Errorf("relacao %d: id %d, esperado %d", i, got.ID, rel.ID)
		}
		if fmt.Sprint(got.Members) != fmt.Sprint(rel.Members) {
			t.Errorf("relacao %d: membros %v, esperado %v", i, got.Members, rel.Members)
		}
		if fmt.Sprint(got.Tags) != fmt.Sprint(rel.Tags) {
			t.Errorf("relacao %d: etiquetas %v, esperado %v", i, got.Tags, rel.Tags)
		}
	}
}

// A restricao de conversao e o que a roteirizacao precisa das relacoes:
// "quem vem por esta via nao pode entrar naquela".
func TestRestricaoDeConversaoSobrevive(t *testing.T) {
	a := arquivoPadrao()
	a.blocoRelacoes([]Relation{{
		ID: 7,
		Members: []Member{
			{ID: 42, Type: MemberWay, Role: "from"},
			{ID: 99, Type: MemberNode, Role: "via"},
			{ID: 43, Type: MemberWay, Role: "to"},
		},
		Tags: Tags{{Key: "type", Value: "restriction"}, {Key: "restriction", Value: "no_left_turn"}},
	}})

	c := coletar(t, a.bytes(), 1)
	rel := c.relacoes[0]

	if tipo, ok := rel.Tags.Get("type"); !ok || tipo != "restriction" {
		t.Errorf("type = %q, %v", tipo, ok)
	}
	var de, via, para int64
	for _, m := range rel.Members {
		switch m.Role {
		case "from":
			de = m.ID
		case "via":
			via = m.ID
		case "to":
			para = m.ID
		}
	}
	if de != 42 || via != 99 || para != 43 {
		t.Errorf("de=%d via=%d para=%d, esperado 42/99/43", de, via, para)
	}
}

// Campo nil no Handler quer dizer "nao decodifique". E o que faz a primeira
// passada de quem so quer vias custar quase nada num extrato de dezenas de
// milhoes de nos.
func TestHandlerNilPulaADecodificacao(t *testing.T) {
	a := arquivoPadrao()
	a.blocoDenso(nosDeTeste(1000))
	a.blocoVias([]Way{{ID: 1, Refs: []int64{1, 2, 3}, Tags: Tags{{Key: "highway", Value: "residential"}}}})

	var vias int
	err := Scan(bytes.NewReader(a.bytes()), Handler{
		// Node deliberadamente nil.
		Way: func(w Way) error { vias++; return nil },
	})
	if err != nil {
		t.Fatal(err)
	}
	if vias != 1 {
		t.Errorf("leu %d vias, esperava 1", vias)
	}
}

// Um handler inteiramente vazio percorre o arquivo sem decodificar nada. E o
// caso extremo do teste acima, e tem de terminar sem erro.
func TestHandlerVazioPercorreSemQuebrar(t *testing.T) {
	a := arquivoPadrao()
	a.blocoDenso(nosDeTeste(200))
	a.blocoVias([]Way{{ID: 1, Refs: []int64{1, 2}}})
	a.blocoRelacoes([]Relation{{ID: 1, Members: []Member{{ID: 1, Type: MemberWay, Role: "outer"}}}})

	if err := Scan(bytes.NewReader(a.bytes()), Handler{}); err != nil {
		t.Fatal(err)
	}
}

func TestErrPararInterrompeSemErro(t *testing.T) {
	a := arquivoPadrao()
	for i := 0; i < 20; i++ {
		a.blocoDenso(nosDeTeste(100))
	}

	lidos := 0
	err := Scan(bytes.NewReader(a.bytes()), Handler{
		Node: func(n Node) error {
			lidos++
			if lidos == 150 {
				return ErrParar
			}
			return nil
		},
	})
	if err != nil {
		t.Fatalf("ErrParar deveria terminar sem erro, veio %v", err)
	}
	if lidos != 150 {
		t.Errorf("parou em %d nos, esperado 150", lidos)
	}
}

func TestErroDoHandlerChegaAoChamador(t *testing.T) {
	a := arquivoPadrao()
	for i := 0; i < 20; i++ {
		a.blocoDenso(nosDeTeste(100))
	}

	querido := errors.New("o handler desistiu")
	err := Scan(bytes.NewReader(a.bytes()), Handler{
		Node: func(n Node) error { return querido },
	})
	if !errors.Is(err, querido) {
		t.Errorf("erro = %v, esperado %v", err, querido)
	}
}

// A decodificacao e paralela, a entrega e ordenada. Duas leituras do mesmo
// arquivo, com paralelismos diferentes, tem de produzir a mesma sequencia.
func TestOrdemNaoDependeDoParalelismo(t *testing.T) {
	a := arquivoPadrao()
	for i := 0; i < 30; i++ {
		nos := make([]Node, 50)
		for j := range nos {
			nos[j] = Node{ID: int64(i*1000 + j), Point: geo.Point{Lat: -3 - float64(j)*0.001, Lon: -38}}
		}
		a.blocoDenso(nos)
	}
	dados := a.bytes()

	referencia := coletar(t, dados, 1)
	for _, p := range []int{2, 3, 8, 16} {
		got := coletar(t, dados, p)
		if len(got.nos) != len(referencia.nos) {
			t.Fatalf("paralelismo %d: %d nos, esperado %d", p, len(got.nos), len(referencia.nos))
		}
		for i := range got.nos {
			if got.nos[i].ID != referencia.nos[i].ID {
				t.Fatalf("paralelismo %d: no %d tem id %d, esperado %d",
					p, i, got.nos[i].ID, referencia.nos[i].ID)
			}
		}
	}
}

func TestCabecalho(t *testing.T) {
	bbox := geo.Box{MinLat: -8, MinLon: -42, MaxLat: -2, MaxLon: -37}
	a := novoArquivo()
	a.cabecalho([]string{"OsmSchema-V0.6", "DenseNodes"}, &bbox)
	a.blocoDenso(nosDeTeste(10))

	c := coletar(t, a.bytes(), 2)
	if len(c.cabecalhos) != 1 {
		t.Fatalf("leu %d cabecalhos, esperava 1", len(c.cabecalhos))
	}
	h := c.cabecalhos[0]

	if !h.HasBBox {
		t.Fatal("o retangulo nao foi lido")
	}
	for _, par := range [][2]float64{
		{h.BBox.MinLat, bbox.MinLat}, {h.BBox.MinLon, bbox.MinLon},
		{h.BBox.MaxLat, bbox.MaxLat}, {h.BBox.MaxLon, bbox.MaxLon},
	} {
		if math.Abs(par[0]-par[1]) > 1e-7 {
			t.Errorf("retangulo: %v, esperado %v", h.BBox, bbox)
			break
		}
	}
	if h.WritingProgram != "era-mapas-teste" {
		t.Errorf("programa = %q", h.WritingProgram)
	}
	if len(h.RequiredFeatures) != 2 {
		t.Errorf("recursos exigidos = %v", h.RequiredFeatures)
	}
}

// Um arquivo que exige capacidade que este leitor nao tem precisa falhar
// alto. Ler pela metade em silencio produziria um mapa com buracos.
func TestRecursoExigidoDesconhecidoFalha(t *testing.T) {
	a := novoArquivo()
	a.cabecalho([]string{"OsmSchema-V0.6", "DenseNodes", "HistoricalInformation"}, nil)
	a.blocoDenso(nosDeTeste(10))

	err := Scan(bytes.NewReader(a.bytes()), Handler{Node: func(Node) error { return nil }})
	if !errors.Is(err, ErrNaoSuportado) {
		t.Errorf("erro = %v, esperado ErrNaoSuportado", err)
	}
}

func TestArquivoSemCabecalhoFalha(t *testing.T) {
	a := novoArquivo()
	a.blocoDenso(nosDeTeste(10)) // dados sem OSMHeader antes

	err := Scan(bytes.NewReader(a.bytes()), Handler{Node: func(Node) error { return nil }})
	if !errors.Is(err, ErrFormato) {
		t.Errorf("erro = %v, esperado ErrFormato", err)
	}
}

// Blob sem compressao: o formato admite, e um leitor que so entende zlib
// quebra em arquivo gerado por ferramenta menos comum.
func TestBlobSemCompressao(t *testing.T) {
	a := novoArquivo()
	a.comprimido = false
	a.cabecalho([]string{"OsmSchema-V0.6", "DenseNodes"}, nil)
	a.blocoDenso(nosDeTeste(100))

	c := coletar(t, a.bytes(), 2)
	if len(c.nos) != 100 {
		t.Errorf("leu %d nos, esperava 100", len(c.nos))
	}
}

// Arquivo cortado no meio -- download interrompido, disco cheio -- tem de
// virar erro, nao mapa pela metade que ninguem percebe que esta incompleto.
func TestArquivoTruncadoFalha(t *testing.T) {
	a := arquivoPadrao()
	a.blocoDenso(nosDeTeste(300))
	a.blocoVias([]Way{{ID: 1, Refs: []int64{1, 2, 3}}})
	inteiro := a.bytes()

	// Corta em varios pontos: no tamanho, no cabecalho, no corpo.
	for _, corte := range []int{2, 6, 20, len(inteiro) / 3, len(inteiro) / 2, len(inteiro) - 1} {
		err := Scan(bytes.NewReader(inteiro[:corte]), Handler{
			Node: func(Node) error { return nil },
			Way:  func(Way) error { return nil },
		})
		if err == nil {
			t.Errorf("cortado em %d bytes: leu sem reclamar", corte)
		} else if !errors.Is(err, ErrFormato) {
			t.Errorf("cortado em %d bytes: erro = %v, esperado ErrFormato", corte, err)
		}
	}
}

func TestLixoNaoEhLidoComoMapa(t *testing.T) {
	for _, lixo := range [][]byte{
		nil,
		[]byte("nao sou um osm.pbf"),
		bytes.Repeat([]byte{0xff}, 1000),
		{0x7f, 0xff, 0xff, 0xff}, // tamanho de cabecalho absurdo
	} {
		err := Scan(bytes.NewReader(lixo), Handler{Node: func(Node) error { return nil }})
		if err == nil {
			t.Errorf("lixo de %d bytes foi lido como mapa", len(lixo))
		}
	}
}

func TestTags(t *testing.T) {
	tags := Tags{
		{Key: "highway", Value: "residential"},
		{Key: "name", Value: "Rua das Flores"},
		{Key: "oneway", Value: "yes"},
	}

	if v, ok := tags.Get("name"); !ok || v != "Rua das Flores" {
		t.Errorf("Get(name) = %q, %v", v, ok)
	}
	if _, ok := tags.Get("maxspeed"); ok {
		t.Error("Get de chave inexistente devolveu ok")
	}
	if !tags.Has("oneway") || tags.Has("bridge") {
		t.Error("Has errou")
	}
	if v, ok := Tags(nil).Get("qualquer"); ok || v != "" {
		t.Error("Get em Tags nil deveria devolver vazio")
	}
}
