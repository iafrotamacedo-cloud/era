package layout

import (
	"math/rand"
	"testing"

	"github.com/iafrotamacedo-cloud/era/documents/geom"
)

// caixa constroi um retangulo axis-aligned como poligono de 4 vertices --
// o suficiente para Box.Bounds() extrair a faixa vertical e horizontal
// certa, que e tudo que GroupLines usa.
func caixa(x0, y0, x1, y1 float64) geom.Polygon {
	return geom.Polygon{
		{X: x0, Y: y0}, {X: x1, Y: y0}, {X: x1, Y: y1}, {X: x0, Y: y1},
	}
}

func palavra(texto string, x0, y0, x1, y1 float64) Word {
	return Word{Box: caixa(x0, y0, x1, y1), Text: texto}
}

func TestGroupLinesVazio(t *testing.T) {
	if got := GroupLines(nil); got != nil {
		t.Errorf("GroupLines(nil) = %v, quero nil", got)
	}
}

func TestGroupLinesUmaPalavra(t *testing.T) {
	w := palavra("so", 0, 0, 10, 10)
	linhas := GroupLines([]Word{w})
	if len(linhas) != 1 || len(linhas[0].Words) != 1 {
		t.Fatalf("GroupLines = %+v, quero 1 linha com 1 palavra", linhas)
	}
	if linhas[0].Text() != "so" {
		t.Errorf("Text() = %q, quero %q", linhas[0].Text(), "so")
	}
}

// TestGroupLinesDuasLinhasEmbaralhadas e o teste principal: duas linhas
// bem separadas, tres palavras cada, entregues em ordem embaralhada.
// GroupLines tem que devolver as linhas de cima para baixo e, dentro de
// cada uma, as palavras da esquerda para a direita -- nunca a ordem em que
// foram passadas.
func TestGroupLinesDuasLinhasEmbaralhadas(t *testing.T) {
	entrada := []Word{
		palavra("123", 85, 10, 110, 20),
		palavra("R$", 45, 50, 55, 60),
		palavra("Nota", 0, 10, 30, 20),
		palavra("500,00", 60, 50, 110, 60),
		palavra("Fiscal", 35, 10, 80, 20),
		palavra("Total", 0, 50, 40, 60),
	}
	rnd := rand.New(rand.NewSource(5))
	rnd.Shuffle(len(entrada), func(i, j int) { entrada[i], entrada[j] = entrada[j], entrada[i] })

	linhas := GroupLines(entrada)
	if len(linhas) != 2 {
		t.Fatalf("GroupLines achou %d linhas, quero 2", len(linhas))
	}
	if got := linhas[0].Text(); got != "Nota Fiscal 123" {
		t.Errorf("linha 0 = %q, quero %q", got, "Nota Fiscal 123")
	}
	if got := linhas[1].Text(); got != "Total R$ 500,00" {
		t.Errorf("linha 1 = %q, quero %q", got, "Total R$ 500,00")
	}
}

// TestGroupLinesToleraJitterVertical confere que duas palavras com faixas
// verticais levemente diferentes (o normal numa deteccao real -- nem toda
// palavra da mesma linha tem exatamente a mesma altura) ainda caem juntas,
// desde que a sobreposicao passe do limiar.
func TestGroupLinesToleraJitterVertical(t *testing.T) {
	entrada := []Word{
		palavra("A", 0, 10, 20, 20),  // faixa [10,20], altura 10
		palavra("B", 25, 12, 45, 22), // faixa [12,22], altura 10 -- sobreposicao [12,20]=8, fracao 0.8
	}
	linhas := GroupLines(entrada)
	if len(linhas) != 1 {
		t.Fatalf("GroupLines achou %d linhas, quero 1 (jitter dentro do limiar)", len(linhas))
	}
	if got := linhas[0].Text(); got != "A B" {
		t.Errorf("linha = %q, quero %q", got, "A B")
	}
}

// TestGroupLinesSeparaSobreposicaoPequena e o caso oposto: sobreposicao
// abaixo do limiar tem que abrir duas linhas, nao uma.
func TestGroupLinesSeparaSobreposicaoPequena(t *testing.T) {
	entrada := []Word{
		palavra("A", 0, 10, 20, 20), // faixa [10,20], altura 10
		palavra("B", 0, 19, 20, 29), // faixa [19,29], altura 10 -- sobreposicao [19,20]=1, fracao 0.1
	}
	linhas := GroupLines(entrada)
	if len(linhas) != 2 {
		t.Fatalf("GroupLines achou %d linhas, quero 2 (sobreposicao abaixo do limiar)", len(linhas))
	}
}

func TestLineBounds(t *testing.T) {
	l := Line{Words: []Word{
		palavra("a", 0, 5, 10, 15),
		palavra("b", 20, 0, 30, 20),
	}}
	min, max := l.Bounds()
	if min != (geom.Point{X: 0, Y: 0}) {
		t.Errorf("min = %v, quero {0,0}", min)
	}
	if max != (geom.Point{X: 30, Y: 20}) {
		t.Errorf("max = %v, quero {30,20}", max)
	}
}
