package ch

import (
	"bytes"
	"errors"
	"math"
	"math/rand"
	"path/filepath"
	"testing"

	"github.com/iafrotamacedo-cloud/era/maps/geo"
	"github.com/iafrotamacedo-cloud/era/maps/graph"
)

func prepararParaFormato(t *testing.T) *CH {
	t.Helper()
	r := rand.New(rand.NewSource(20))
	g := grafoAleatorio(t, r, 60, 200)
	c, err := Prepare(g, graph.Time)
	if err != nil {
		t.Fatal(err)
	}
	return c
}

// O teste central do formato: a hierarquia que volta do arquivo responde
// exatamente o mesmo que a que foi gravada, em todo par.
//
// Comparar campo a campo nao bastaria. O que importa nao e a estrutura ser
// igual, e sim as respostas serem -- e e por isso que a checagem passa pela
// consulta.
func TestIdaEVoltaResponderIgual(t *testing.T) {
	original := prepararParaFormato(t)

	var buf bytes.Buffer
	if err := original.Save(&buf); err != nil {
		t.Fatal(err)
	}

	// O tamanho e previsivel. Se deixar de ser, o formato mudou sem que
	// ninguem avisasse.
	querido := 9 + 24 + original.Len()*tamanhoNo + (original.Len()+1)*4 + original.Arcs()*tamanhoArco
	if buf.Len() != querido {
		t.Errorf("arquivo com %d bytes, esperado %d", buf.Len(), querido)
	}

	lida, err := Load(bytes.NewReader(buf.Bytes()))
	if err != nil {
		t.Fatal(err)
	}

	if lida.Len() != original.Len() || lida.Arcs() != original.Arcs() {
		t.Fatalf("carregou %d nos e %d arcos, gravou %d e %d",
			lida.Len(), lida.Arcs(), original.Len(), original.Arcs())
	}
	if lida.Metric() != original.Metric() {
		t.Errorf("metrica = %v, esperada %v", lida.Metric(), original.Metric())
	}

	qo, ql := original.NewQuery(), lida.NewQuery()
	for de := graph.NodeID(0); int(de) < original.Len(); de++ {
		for para := graph.NodeID(0); int(para) < original.Len(); para++ {
			co, oko := qo.Cost(de, para)
			cl, okl := ql.Cost(de, para)
			if oko != okl {
				t.Fatalf("%d->%d: original achou=%v, carregada achou=%v", de, para, oko, okl)
			}
			if oko && co != cl {
				t.Fatalf("%d->%d: original %.9f, carregada %.9f", de, para, co, cl)
			}
		}
	}
}

// Coordenadas e identificadores tem de voltar bit a bit: sao float64 no
// arquivo justamente para isso.
func TestNosVoltamExatos(t *testing.T) {
	original := prepararParaFormato(t)

	var buf bytes.Buffer
	if err := original.Save(&buf); err != nil {
		t.Fatal(err)
	}
	lida, err := Load(&buf)
	if err != nil {
		t.Fatal(err)
	}

	for n := graph.NodeID(0); int(n) < original.Len(); n++ {
		if lida.Point(n) != original.Point(n) {
			t.Fatalf("no %d: %v, esperado %v", n, lida.Point(n), original.Point(n))
		}
		if lida.OSMID(n) != original.OSMID(n) {
			t.Fatalf("no %d: osm %d, esperado %d", n, lida.OSMID(n), original.OSMID(n))
		}
		if lida.Rank(n) != original.Rank(n) {
			t.Fatalf("no %d: posicao %d, esperada %d", n, lida.Rank(n), original.Rank(n))
		}
	}
}

// O caminho desempacotado tambem tem de sobreviver: e ele que depende dos nos
// do meio guardados em cada arco.
func TestCaminhosSobrevivemAoArquivo(t *testing.T) {
	original := prepararParaFormato(t)

	var buf bytes.Buffer
	if err := original.Save(&buf); err != nil {
		t.Fatal(err)
	}
	lida, err := Load(&buf)
	if err != nil {
		t.Fatal(err)
	}

	qo, ql := original.NewQuery(), lida.NewQuery()
	conferidos := 0
	for de := graph.NodeID(0); int(de) < original.Len(); de += 3 {
		for para := graph.NodeID(0); int(para) < original.Len(); para += 3 {
			po, oko := qo.Route(de, para)
			pl, okl := ql.Route(de, para)
			if oko != okl {
				t.Fatalf("%d->%d: achou=%v contra %v", de, para, oko, okl)
			}
			if !oko {
				continue
			}
			conferidos++

			if len(po.Nodes) != len(pl.Nodes) {
				t.Fatalf("%d->%d: caminho com %d nos, esperado %d", de, para, len(pl.Nodes), len(po.Nodes))
			}
			for i := range po.Nodes {
				if po.Nodes[i] != pl.Nodes[i] {
					t.Fatalf("%d->%d: no %d do caminho e %d, esperado %d",
						de, para, i, pl.Nodes[i], po.Nodes[i])
				}
			}
			if po.Meters != pl.Meters || po.Seconds != pl.Seconds {
				t.Fatalf("%d->%d: %.6f m %.6f s, esperado %.6f m %.6f s",
					de, para, pl.Meters, pl.Seconds, po.Meters, po.Seconds)
			}
		}
	}
	if conferidos < 20 {
		t.Errorf("so %d caminhos conferidos", conferidos)
	}
}

// Nearest funciona depois de carregar. E o ponto do formato: a hierarquia nao
// precisa mais do .osm.pbf que a gerou.
func TestNearestFuncionaSemOGrafoOriginal(t *testing.T) {
	original := prepararParaFormato(t)

	var buf bytes.Buffer
	if err := original.Save(&buf); err != nil {
		t.Fatal(err)
	}
	lida, err := Load(&buf)
	if err != nil {
		t.Fatal(err)
	}

	for n := graph.NodeID(0); int(n) < lida.Len(); n += 7 {
		p := lida.Point(n)
		achado, metros, ok := lida.Nearest(p)
		if !ok {
			t.Fatalf("Nearest nao achou nada para o proprio no %d", n)
		}
		if metros > 1 {
			t.Errorf("no %d: o mais proximo dele mesmo esta a %.2f m", n, metros)
		}
		if lida.Point(achado) != p {
			t.Errorf("no %d: achou %d, que esta em outro lugar", n, achado)
		}
	}

	if _, _, ok := lida.Nearest(geo.Point{Lat: 200, Lon: 0}); ok {
		t.Error("coordenada invalida deveria ser recusada")
	}
}

func TestSaveFileELoadFile(t *testing.T) {
	original := prepararParaFormato(t)
	caminho := filepath.Join(t.TempDir(), "teste.eramap")

	if err := original.SaveFile(caminho); err != nil {
		t.Fatal(err)
	}
	lida, err := LoadFile(caminho)
	if err != nil {
		t.Fatal(err)
	}
	if lida.Len() != original.Len() || lida.Arcs() != original.Arcs() {
		t.Errorf("carregou %d/%d, gravou %d/%d", lida.Len(), lida.Arcs(), original.Len(), original.Arcs())
	}

	if _, err := LoadFile(filepath.Join(t.TempDir(), "nao-existe.eramap")); err == nil {
		t.Error("arquivo inexistente deveria dar erro")
	}
}

func TestFormatoRejeitaArquivoErrado(t *testing.T) {
	for _, lixo := range [][]byte{
		nil,
		[]byte("oi"),
		[]byte("nao sou uma hierarquia da era, mas tenho tamanho"),
		bytes.Repeat([]byte{0xff}, 500),
	} {
		if _, err := Load(bytes.NewReader(lixo)); !errors.Is(err, ErrFormato) {
			t.Errorf("lixo de %d bytes: erro = %v, esperado ErrFormato", len(lixo), err)
		}
	}
}

func TestFormatoRejeitaVersaoDesconhecida(t *testing.T) {
	c := prepararParaFormato(t)
	var buf bytes.Buffer
	if err := c.Save(&buf); err != nil {
		t.Fatal(err)
	}
	bs := buf.Bytes()
	bs[7] = versaoAtual + 1

	if _, err := Load(bytes.NewReader(bs)); !errors.Is(err, ErrVersao) {
		t.Errorf("erro = %v, esperado ErrVersao", err)
	}
}

// Arquivo cortado -- disco cheio, copia interrompida -- tem de virar erro, e
// nao uma hierarquia pela metade que responde rota errada.
func TestFormatoRejeitaArquivoTruncado(t *testing.T) {
	c := prepararParaFormato(t)
	var buf bytes.Buffer
	if err := c.Save(&buf); err != nil {
		t.Fatal(err)
	}
	inteiro := buf.Bytes()

	for _, corte := range []int{5, 10, 40, len(inteiro) / 3, len(inteiro) / 2, len(inteiro) - 1} {
		if _, err := Load(bytes.NewReader(inteiro[:corte])); !errors.Is(err, ErrFormato) {
			t.Errorf("cortado em %d bytes: erro = %v, esperado ErrFormato", corte, err)
		}
	}
}

// Um arquivo do tamanho certo mas com conteudo corrompido e o caso perigoso:
// passaria pela checagem de tamanho e quebraria no meio de uma consulta. As
// invariantes existem para pega-lo na abertura.
func TestFormatoRejeitaInvarianteQuebrada(t *testing.T) {
	c := prepararParaFormato(t)
	var buf bytes.Buffer
	if err := c.Save(&buf); err != nil {
		t.Fatal(err)
	}
	base := buf.Bytes()

	corromper := func(desloca int, valor uint32) []byte {
		copia := append([]byte(nil), base...)
		copia[desloca] = byte(valor)
		copia[desloca+1] = byte(valor >> 8)
		copia[desloca+2] = byte(valor >> 16)
		copia[desloca+3] = byte(valor >> 24)
		return copia
	}

	inicioNos := 9 + 24
	// Duas posicoes iguais: a regra de "so sobe" perderia o sentido.
	duplicada := corromper(inicioNos+24, uint32(c.Rank(1)))
	if _, err := Load(bytes.NewReader(duplicada)); !errors.Is(err, ErrFormato) {
		t.Errorf("posicao duplicada: erro = %v, esperado ErrFormato", err)
	}

	// Posicao fora da faixa.
	fora := corromper(inicioNos+24, 999999)
	if _, err := Load(bytes.NewReader(fora)); !errors.Is(err, ErrFormato) {
		t.Errorf("posicao fora da faixa: erro = %v, esperado ErrFormato", err)
	}

	// Um arco apontando para fora do grafo.
	inicioArcos := inicioNos + c.Len()*tamanhoNo + (c.Len()+1)*4
	alvoInvalido := corromper(inicioArcos, 888888)
	if _, err := Load(bytes.NewReader(alvoInvalido)); !errors.Is(err, ErrFormato) {
		t.Errorf("arco para fora do grafo: erro = %v, esperado ErrFormato", err)
	}

	// Um indice de inicio maior que o seguinte.
	inicioOffsets := inicioNos + c.Len()*tamanhoNo
	desordenado := corromper(inicioOffsets+4, math.MaxUint32-1)
	if _, err := Load(bytes.NewReader(desordenado)); !errors.Is(err, ErrFormato) {
		t.Errorf("indices fora de ordem: erro = %v, esperado ErrFormato", err)
	}
}

// A hierarquia vazia nao existe -- Prepare recusa grafo vazio --, mas o
// formato precisa aguentar a menor possivel sem caso especial.
func TestHierarquiaMinima(t *testing.T) {
	nos := []noPbf{
		{id: 1, p: pontoEm(-3.70, -38.50)},
		{id: 2, p: pontoEm(-3.70, -38.49)},
	}
	g := montarGrafo(t, nos, []viaPbf{
		{id: 1, refs: []int64{1, 2}, tags: map[string]string{"highway": "residential"}},
	})
	c, err := Prepare(g, graph.Distance)
	if err != nil {
		t.Fatal(err)
	}

	var buf bytes.Buffer
	if err := c.Save(&buf); err != nil {
		t.Fatal(err)
	}
	lida, err := Load(&buf)
	if err != nil {
		t.Fatal(err)
	}

	antes, _ := c.Cost(0, 1)
	depois, ok := lida.Cost(0, 1)
	if !ok || antes != depois {
		t.Errorf("custo %v depois do arquivo, era %v (ok=%v)", depois, antes, ok)
	}
}
