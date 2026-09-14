// Package layout organiza palavras soltas (posição + texto) em linhas de
// texto, na ordem em que uma pessoa leria a página.
//
// Só usa geometria -- a posição das caixas -- nunca o conteúdo do texto.
// Isso cobre corretamente um bloco de texto de uma coluna só: a maioria
// dos campos de uma nota fiscal, boleto ou ordem de compra (cabeçalho,
// dados do fornecedor, totais). Multi-coluna e tabela ficam fora por ora
// -- ver a ressalva em GroupLines.
package layout

import (
	"strings"

	"github.com/iafrotamacedo-cloud/era/documents/geom"
)

// Word é uma palavra reconhecida: a caixa que a delimita e o texto que o
// reconhecedor devolveu para ela (fase 5 do roteiro, que ainda não existe
// -- este pacote só organiza o que ela vai produzir).
type Word struct {
	Box  geom.Polygon
	Text string
}

func (w Word) yRange() (min, max float64) {
	lo, hi := w.Box.Bounds()
	return lo.Y, hi.Y
}

func (w Word) xMin() float64 {
	lo, _ := w.Box.Bounds()
	return lo.X
}

// Line é uma sequência de palavras da mesma linha de texto, em ordem de
// leitura esquerda para direita.
type Line struct {
	Words []Word
}

// Bounds devolve o retângulo que envolve todas as palavras da linha.
func (l Line) Bounds() (min, max geom.Point) {
	if len(l.Words) == 0 {
		return geom.Point{}, geom.Point{}
	}
	min, max = l.Words[0].Box.Bounds()
	for _, w := range l.Words[1:] {
		wMin, wMax := w.Box.Bounds()
		if wMin.X < min.X {
			min.X = wMin.X
		}
		if wMin.Y < min.Y {
			min.Y = wMin.Y
		}
		if wMax.X > max.X {
			max.X = wMax.X
		}
		if wMax.Y > max.Y {
			max.Y = wMax.Y
		}
	}
	return min, max
}

// Text junta o texto das palavras da linha, separado por espaço, na ordem
// em que Words já está -- esquerda para direita.
func (l Line) Text() string {
	partes := make([]string, len(l.Words))
	for i, w := range l.Words {
		partes[i] = w.Text
	}
	return strings.Join(partes, " ")
}
