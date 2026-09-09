package graph

import (
	"bytes"
	"compress/zlib"
	"encoding/binary"
	"math"
	"testing"

	"github.com/iafrotamacedo-cloud/era/internal/protowire"
	"github.com/iafrotamacedo-cloud/era/maps/geo"
)

// Um .osm.pbf minimo, montado a mao, para os testes de construcao.
//
// O repositorio nao guarda extratos de mapa -- eles pesam centenas de MB e
// tem licenca propria --, entao o jeito de testar a construcao e escrever o
// arquivo. E o mesmo movimento que o maps/osm faz nos testes dele.
//
// So o que a construcao consome: cabecalho, nos densos e vias com etiquetas.

type noTeste struct {
	id  int64
	p   geo.Point
	tag [2]string
}

type viaTeste struct {
	id   int64
	refs []int64
	tags map[string]string
}

type arquivo struct {
	buf bytes.Buffer
}

func (a *arquivo) blob(tipo string, corpo []byte) {
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

func (a *arquivo) cabecalho() {
	w := protowire.NewWriter()
	w.StringField(4, "OsmSchema-V0.6")
	w.StringField(4, "DenseNodes")
	a.blob("OSMHeader", w.Bytes())
}

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

func paraInteiro(graus float64) int64 { return int64(math.Round(graus * 1e7)) }

func (a *arquivo) nos(nos []noTeste) {
	t := novaTabela()

	var ids, lats, lons []int64
	var kv []int32
	var antID, antLat, antLon int64

	for _, n := range nos {
		lat, lon := paraInteiro(n.p.Lat), paraInteiro(n.p.Lon)
		ids = append(ids, n.id-antID)
		lats = append(lats, lat-antLat)
		lons = append(lons, lon-antLon)
		antID, antLat, antLon = n.id, lat, lon

		if n.tag[0] != "" {
			kv = append(kv, t.id(n.tag[0]), t.id(n.tag[1]))
		}
		kv = append(kv, 0)
	}

	dense := protowire.NewWriter()
	dense.PackedSInt64sField(1, ids)
	dense.PackedSInt64sField(8, lats)
	dense.PackedSInt64sField(9, lons)
	dense.PackedInt32sField(10, kv)

	grupo := protowire.NewWriter()
	grupo.MessageField(2, dense)

	w := protowire.NewWriter()
	t.escrever(w)
	w.MessageField(2, grupo)
	a.blob("OSMData", w.Bytes())
}

func (a *arquivo) vias(vias []viaTeste) {
	t := novaTabela()
	grupo := protowire.NewWriter()

	for _, v := range vias {
		// Ordem estavel das etiquetas, para o arquivo nao mudar entre
		// execucoes por causa da iteracao aleatoria de map.
		var chaves, valores []int32
		for _, k := range chavesOrdenadas(v.tags) {
			chaves = append(chaves, t.id(k))
			valores = append(valores, t.id(v.tags[k]))
		}

		var refs []int64
		var ant int64
		for _, r := range v.refs {
			refs = append(refs, r-ant)
			ant = r
		}

		w := protowire.NewWriter()
		w.Int64Field(1, v.id)
		w.PackedInt32sField(2, chaves)
		w.PackedInt32sField(3, valores)
		w.PackedSInt64sField(8, refs)
		grupo.MessageField(3, w)
	}

	w := protowire.NewWriter()
	t.escrever(w)
	w.MessageField(2, grupo)
	a.blob("OSMData", w.Bytes())
}

func chavesOrdenadas(m map[string]string) []string {
	ks := make([]string, 0, len(m))
	for k := range m {
		ks = append(ks, k)
	}
	for i := 1; i < len(ks); i++ {
		for j := i; j > 0 && ks[j] < ks[j-1]; j-- {
			ks[j], ks[j-1] = ks[j-1], ks[j]
		}
	}
	return ks
}

// ruaReta devolve uma via de n nos em linha, espacados de 100 m.
func ruaReta(idBase int64, n int, lat float64) ([]noTeste, []int64) {
	var nos []noTeste
	var refs []int64
	for i := 0; i < n; i++ {
		id := idBase + int64(i)
		nos = append(nos, noTeste{id: id, p: geo.Point{
			Lat: lat,
			Lon: -38.5 + float64(i)*0.0009, // ~100 m por passo nesta latitude
		}})
		refs = append(refs, id)
	}
	return nos, refs
}

func montar(t *testing.T, nos []noTeste, vias []viaTeste) *Graph {
	t.Helper()
	return montarCom(t, nos, vias, Car())
}

func montarCom(t *testing.T, nos []noTeste, vias []viaTeste, p Profile) *Graph {
	t.Helper()
	var a arquivo
	a.cabecalho()
	a.nos(nos)
	a.vias(vias)

	g, err := Build(bytes.NewReader(a.buf.Bytes()), p)
	if err != nil {
		t.Fatalf("Build: %v", err)
	}
	return g
}

// A decisao central do pacote: os nos que so desenham a curva da rua nao
// viram no do grafo.
func TestContracaoDeNosDeGrau2(t *testing.T) {
	// Uma rua de 11 nos, ou seja 10 segmentos de 100 m. Nenhum cruzamento no
	// meio: o grafo tem de ficar com 2 nos e 1 trecho de 1 km.
	nos, refs := ruaReta(1, 11, -3.7)
	g := montar(t, nos, []viaTeste{
		{id: 1, refs: refs, tags: map[string]string{"highway": "residential"}},
	})

	if g.Len() != 2 {
		t.Fatalf("grafo com %d nos, esperava 2 -- os 9 do meio deveriam ter sido contraidos", g.Len())
	}
	if g.Edges() != 2 { // um trecho, guardado nas duas pontas
		t.Fatalf("grafo com %d arestas, esperava 2", g.Edges())
	}

	st := g.Stats()
	t.Log(st)
	if st.NosDeVia != 11 || st.Cruzamentos != 2 {
		t.Errorf("stats = %+v", st)
	}
	if e := st.Encolhimento(); math.Abs(e-9.0/11.0) > 1e-9 {
		t.Errorf("encolhimento = %.4f, esperado %.4f", e, 9.0/11.0)
	}

	// O comprimento tem de ser a soma dos dez pedacos, nao a linha reta entre
	// as pontas -- que aqui calha de ser igual, porque a rua e reta. O teste
	// da curva abaixo cobre a diferenca.
	p, ok := g.Route(0, 1, Distance)
	if !ok {
		t.Fatal("sem rota entre as duas pontas")
	}
	if math.Abs(p.Meters-1000) > 20 {
		t.Errorf("comprimento = %.1f m, esperado ~1000", p.Meters)
	}
}

// O comprimento do trecho contraido e a soma dos pedacos, e nao a linha reta
// entre os cruzamentos. Numa rua em L a diferenca e grande, e e isso que
// prova que a geometria intermediaria foi levada em conta antes de ser
// descartada.
func TestComprimentoSegueACurva(t *testing.T) {
	// Um L: 1 km para o leste, depois 1 km para o sul.
	nos := []noTeste{
		{id: 1, p: geo.Point{Lat: -3.7, Lon: -38.5}},
		{id: 2, p: geo.Point{Lat: -3.7, Lon: -38.491}},
		{id: 3, p: geo.Point{Lat: -3.709, Lon: -38.491}},
	}
	g := montar(t, nos, []viaTeste{
		{id: 1, refs: []int64{1, 2, 3}, tags: map[string]string{"highway": "residential"}},
	})

	if g.Len() != 2 {
		t.Fatalf("grafo com %d nos, esperava 2", g.Len())
	}

	p, ok := g.Route(0, 1, Distance)
	if !ok {
		t.Fatal("sem rota")
	}

	pelaCurva := geo.Haversine(nos[0].p, nos[1].p) + geo.Haversine(nos[1].p, nos[2].p)
	emLinhaReta := geo.Haversine(nos[0].p, nos[2].p)

	if math.Abs(p.Meters-pelaCurva) > 1 {
		t.Errorf("comprimento = %.1f m, esperado %.1f (a soma dos pedacos)", p.Meters, pelaCurva)
	}
	if math.Abs(p.Meters-emLinhaReta) < 100 {
		t.Errorf("o comprimento %.1f esta perto demais da linha reta %.1f; o teste nao esta provando nada",
			p.Meters, emLinhaReta)
	}
}

// Um no compartilhado por duas ruas e cruzamento e nao pode ser contraido.
func TestNoCompartilhadoViraCruzamento(t *testing.T) {
	// Duas ruas em cruz, encontrando-se no no 3.
	nos := []noTeste{
		{id: 1, p: geo.Point{Lat: -3.700, Lon: -38.500}},
		{id: 2, p: geo.Point{Lat: -3.700, Lon: -38.495}},
		{id: 3, p: geo.Point{Lat: -3.700, Lon: -38.490}}, // o cruzamento
		{id: 4, p: geo.Point{Lat: -3.700, Lon: -38.485}},
		{id: 5, p: geo.Point{Lat: -3.695, Lon: -38.490}},
		{id: 6, p: geo.Point{Lat: -3.705, Lon: -38.490}},
	}
	g := montar(t, nos, []viaTeste{
		{id: 1, refs: []int64{1, 2, 3, 4}, tags: map[string]string{"highway": "residential"}},
		{id: 2, refs: []int64{5, 3, 6}, tags: map[string]string{"highway": "residential"}},
	})

	// Cruzamentos: 1, 3, 4, 5, 6. O no 2 e ponto de curva e some.
	if g.Len() != 5 {
		t.Fatalf("grafo com %d nos, esperava 5 (o no 2 deveria sumir)", g.Len())
	}

	// Da ponta de uma rua ate a ponta da outra, passando pelo cruzamento.
	de, _, _ := g.Nearest(nos[0].p)
	para, _, _ := g.Nearest(nos[4].p)
	p, ok := g.Route(de, para, Distance)
	if !ok {
		t.Fatal("as duas ruas deveriam estar ligadas pelo cruzamento")
	}
	if len(p.Nodes) != 3 {
		t.Errorf("caminho com %d nos, esperava 3 (ponta, cruzamento, ponta)", len(p.Nodes))
	}
}

func TestPerfilFiltraOQueNaoEhEstrada(t *testing.T) {
	nosA, refsA := ruaReta(1, 3, -3.70)
	nosB, refsB := ruaReta(101, 3, -3.71)
	nosC, refsC := ruaReta(201, 3, -3.72)

	g := montar(t, append(append(nosA, nosB...), nosC...), []viaTeste{
		{id: 1, refs: refsA, tags: map[string]string{"highway": "residential"}},
		{id: 2, refs: refsB, tags: map[string]string{"highway": "footway"}},
		{id: 3, refs: refsC, tags: map[string]string{"building": "yes"}},
	})

	st := g.Stats()
	if st.ViasLidas != 3 || st.ViasUsadas != 1 {
		t.Errorf("stats = %+v; so a residential deveria contar", st)
	}
	if g.Len() != 2 {
		t.Errorf("grafo com %d nos, esperava 2", g.Len())
	}
}

func TestMaoUnicaVemDaEtiqueta(t *testing.T) {
	casos := []struct {
		tags         map[string]string
		frente, tras bool
	}{
		{map[string]string{"highway": "residential"}, true, true},
		{map[string]string{"highway": "residential", "oneway": "yes"}, true, false},
		{map[string]string{"highway": "residential", "oneway": "-1"}, false, true},
		{map[string]string{"highway": "motorway"}, true, false},                // implicita
		{map[string]string{"highway": "motorway", "oneway": "no"}, true, true}, // explicita vence
		{map[string]string{"highway": "residential", "junction": "roundabout"}, true, false},
	}

	for i, caso := range casos {
		nos, refs := ruaReta(1, 3, -3.7)
		g := montar(t, nos, []viaTeste{{id: 1, refs: refs, tags: caso.tags}})
		if g.Len() != 2 {
			t.Fatalf("caso %d: grafo com %d nos", i, g.Len())
		}

		_, ida := g.Route(0, 1, Distance)
		_, volta := g.Route(1, 0, Distance)
		if ida != caso.frente || volta != caso.tras {
			t.Errorf("caso %d %v: ida=%v volta=%v, esperado ida=%v volta=%v",
				i, caso.tags, ida, volta, caso.frente, caso.tras)
		}
	}
}

// Uma via que cita nos que o extrato nao trouxe -- rua cortada na fronteira
// do recorte. Nao pode quebrar, e nao pode inventar distancia atraves do
// buraco.
func TestNoAusenteInterrompeOTrecho(t *testing.T) {
	// Duas ruas que se encontram no no 4. O no 3, ponto de curva da primeira,
	// nao veio no arquivo.
	//
	//   rua 1: 1 - 2 - [3 ausente] - 4
	//   rua 2: 4 - 5 - 6
	//
	// A rua 1 fica sem nenhum trecho: entre o cruzamento 1 e o cruzamento 4 o
	// caminho passa pelo buraco, e medir atraves dele seria inventar. A rua 2
	// sobrevive inteira.
	nos := []noTeste{
		{id: 1, p: geo.Point{Lat: -3.700, Lon: -38.500}},
		{id: 2, p: geo.Point{Lat: -3.700, Lon: -38.499}},
		{id: 4, p: geo.Point{Lat: -3.700, Lon: -38.497}},
		{id: 5, p: geo.Point{Lat: -3.700, Lon: -38.496}},
		{id: 6, p: geo.Point{Lat: -3.700, Lon: -38.495}},
	}
	g := montar(t, nos, []viaTeste{
		{id: 1, refs: []int64{1, 2, 3, 4}, tags: map[string]string{"highway": "residential"}},
		{id: 2, refs: []int64{4, 5, 6}, tags: map[string]string{"highway": "residential"}},
	})

	st := g.Stats()
	if st.NosSemCoordenada != 1 {
		t.Errorf("nos sem coordenada = %d, esperava 1 (stats: %+v)", st.NosSemCoordenada, st)
	}
	if st.Trechos != 1 {
		t.Errorf("trechos = %d, esperava 1 -- so a rua 2 sobrevive", st.Trechos)
	}
	if g.Len() != 2 {
		t.Fatalf("grafo com %d nos, esperava 2 (os cruzamentos 4 e 6)", g.Len())
	}

	// O no 1 nao pode ter entrado no grafo: ele so era alcancavel pelo trecho
	// que o buraco cortou.
	for n := NodeID(0); int(n) < g.Len(); n++ {
		if g.OSMID(n) == 1 {
			t.Errorf("o no 1 entrou no grafo apesar de estar do outro lado do buraco")
		}
	}
}

// Duas construcoes do mesmo arquivo tem de dar o mesmo grafo. Os indices dos
// nos saem da ordem do arquivo, e nao da iteracao do mapa, que em Go e
// deliberadamente aleatoria.
func TestConstrucaoEhDeterministica(t *testing.T) {
	var nos []noTeste
	var vias []viaTeste
	for r := 0; r < 20; r++ {
		n, refs := ruaReta(int64(r*100+1), 6, -3.7-float64(r)*0.002)
		nos = append(nos, n...)
		vias = append(vias, viaTeste{
			id:   int64(r + 1),
			refs: refs,
			tags: map[string]string{"highway": "residential"},
		})
	}

	var a arquivo
	a.cabecalho()
	a.nos(nos)
	a.vias(vias)
	dados := a.buf.Bytes()

	primeiro, err := Build(bytes.NewReader(dados), Car())
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 5; i++ {
		outro, err := Build(bytes.NewReader(dados), Car())
		if err != nil {
			t.Fatal(err)
		}
		if outro.Len() != primeiro.Len() || outro.Edges() != primeiro.Edges() {
			t.Fatalf("construcao %d: %d nos e %d arestas, esperado %d e %d",
				i, outro.Len(), outro.Edges(), primeiro.Len(), primeiro.Edges())
		}
		for n := NodeID(0); int(n) < primeiro.Len(); n++ {
			if outro.OSMID(n) != primeiro.OSMID(n) {
				t.Fatalf("construcao %d: no %d e o osm %d, esperado %d",
					i, n, outro.OSMID(n), primeiro.OSMID(n))
			}
		}
	}
}

func TestNearest(t *testing.T) {
	nos, refs := ruaReta(1, 11, -3.7)
	g := montar(t, nos, []viaTeste{
		{id: 1, refs: refs, tags: map[string]string{"highway": "residential"}},
	})

	// Um ponto logo acima da primeira ponta.
	perto := geo.Point{Lat: nos[0].p.Lat + 0.0002, Lon: nos[0].p.Lon}
	n, metros, ok := g.Nearest(perto)
	if !ok {
		t.Fatal("Nearest nao achou nada")
	}
	if g.OSMID(n) != 1 {
		t.Errorf("achou o osm %d, esperava 1", g.OSMID(n))
	}
	if metros < 10 || metros > 40 {
		t.Errorf("distancia = %.1f m, esperava uns 22", metros)
	}

	if _, _, ok := g.Nearest(geo.Point{Lat: 200, Lon: 0}); ok {
		t.Error("coordenada invalida deveria ser recusada")
	}
	vazio := &Graph{}
	if _, _, ok := vazio.Nearest(geo.Point{}); ok {
		t.Error("grafo vazio nao tem no mais proximo")
	}
}

func TestBuildRecusaPerfilVazio(t *testing.T) {
	var a arquivo
	a.cabecalho()
	if _, err := Build(bytes.NewReader(a.buf.Bytes()), Profile{}); err == nil {
		t.Error("perfil sem tipos de via deveria ser recusado")
	}
}

func TestBuildPropagaErroDeLeitura(t *testing.T) {
	if _, err := Build(bytes.NewReader([]byte("nao sou um osm.pbf")), Car()); err == nil {
		t.Error("arquivo invalido deveria virar erro")
	}
}

// A superficie reduz a velocidade e nao mexe na distancia. Barro nao encurta
// nem alonga a estrada -- so muda quanto tempo ela custa.
func TestSuperficieMudaOTempoENaoADistancia(t *testing.T) {
	// unclassified anda a 40 km/h. O teto so morde quando e menor que isso --
	// e por isso que compacted, cujo teto e 50, nao muda nada. Um teto que
	// virasse multiplicador quebraria esse caso.
	casos := []struct {
		nome string
		tags map[string]string
		kmh  float64
	}{
		{"sem etiqueta", map[string]string{"highway": "unclassified"}, 40},
		{"asfalto", map[string]string{"highway": "unclassified", "surface": "asphalt"}, 40},
		{"compactada, teto acima da classe", map[string]string{"highway": "unclassified", "surface": "compacted"}, 40},
		{"terra", map[string]string{"highway": "unclassified", "surface": "dirt"}, 25},
		{"areia", map[string]string{"highway": "unclassified", "surface": "sand"}, 15},
		{"paralelepipedo", map[string]string{"highway": "unclassified", "surface": "cobblestone"}, 30},
		{"valor desconhecido", map[string]string{"highway": "unclassified", "surface": "marte"}, 40},
		{"tracktype no lugar de surface", map[string]string{"highway": "unclassified", "tracktype": "grade4"}, 25},
		{"surface vence tracktype", map[string]string{"highway": "unclassified", "surface": "asphalt", "tracktype": "grade5"}, 40},
	}

	var referencia Path
	for i, caso := range casos {
		nos, refs := ruaReta(1, 3, -3.7)
		g := montar(t, nos, []viaTeste{{id: 1, refs: refs, tags: caso.tags}})

		p, ok := g.Route(0, 1, Distance)
		if !ok {
			t.Fatalf("%s: sem rota", caso.nome)
		}
		if i == 0 {
			referencia = p
			continue
		}

		if math.Abs(p.Meters-referencia.Meters) > 1e-6 {
			t.Errorf("%s: a distancia mudou, %.3f contra %.3f", caso.nome, p.Meters, referencia.Meters)
		}

		// Menor velocidade e mais tempo, na proporcao das duas velocidades.
		querido := referencia.Seconds * (40 / caso.kmh)
		if math.Abs(p.Seconds-querido) > 1e-3*querido {
			t.Errorf("%s: %.2f s, esperado %.2f s (teto %.0f km/h)", caso.nome, p.Seconds, querido, caso.kmh)
		}
	}
}

// O motivo pelo qual isto existe: com a superficie no perfil, a rota mais
// rapida passa a preferir o desvio pelo asfalto -- que e o que o motorista
// faz.
func TestAsfaltoGanhaDoBarroNoTempo(t *testing.T) {
	// Dois caminhos entre os nos 1 e 4, ambos unclassified: o unico que muda
	// entre eles e a superficie.
	//
	//   curto, de terra:  1 - 2 - 4
	//   longo, asfaltado: 1 - 3 - 4, com um desvio para o norte
	//
	// A classe da via precisa ser a mesma nos dois, senao a velocidade base ja
	// decide sozinha e o teste nao prova nada sobre superficie.
	nos := []noTeste{
		{id: 1, p: geo.Point{Lat: -3.700, Lon: -38.500}},
		{id: 2, p: geo.Point{Lat: -3.700, Lon: -38.490}},
		{id: 3, p: geo.Point{Lat: -3.690, Lon: -38.495}},
		{id: 4, p: geo.Point{Lat: -3.700, Lon: -38.480}},
	}
	vias := []viaTeste{
		{id: 1, refs: []int64{1, 2, 4}, tags: map[string]string{"highway": "unclassified", "surface": "dirt"}},
		{id: 2, refs: []int64{1, 3}, tags: map[string]string{"highway": "unclassified", "surface": "asphalt"}},
		{id: 3, refs: []int64{3, 4}, tags: map[string]string{"highway": "unclassified", "surface": "asphalt"}},
	}
	g := montar(t, nos, vias)

	de, _, _ := g.Nearest(nos[0].p)
	para, _, _ := g.Nearest(nos[3].p)

	curto, ok := g.Route(de, para, Distance)
	if !ok {
		t.Fatal("sem rota por distancia")
	}
	rapido, ok := g.Route(de, para, Time)
	if !ok {
		t.Fatal("sem rota por tempo")
	}

	if curto.Meters >= rapido.Meters {
		t.Errorf("a rota de terra deveria ser a mais curta: %.0f m contra %.0f m",
			curto.Meters, rapido.Meters)
	}
	if rapido.Seconds >= curto.Seconds {
		t.Errorf("o asfalto deveria ser mais rapido: %.1f s contra %.1f s",
			rapido.Seconds, curto.Seconds)
	}

	// Sem penalidade de superficie o barro venceria tambem no tempo, porque e
	// mais curto. E o teste que prova que a mudanca faz alguma coisa.
	semSuperficie := Car()
	semSuperficie.SurfaceMaxKmh = nil
	g2 := montarCom(t, nos, vias, semSuperficie)
	de2, _, _ := g2.Nearest(nos[0].p)
	para2, _, _ := g2.Nearest(nos[3].p)
	antes, ok := g2.Route(de2, para2, Time)
	if !ok {
		t.Fatal("sem rota sem superficie")
	}
	if antes.Meters >= rapido.Meters {
		t.Error("sem penalidade de superficie a rota mais rapida deveria ser a de terra")
	}
}
