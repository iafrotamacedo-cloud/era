package layout

import (
	"math"
	"sort"
)

// OverlapFraction é o limiar mínimo de sobreposição vertical -- como
// fração da altura da menor das duas caixas -- para duas palavras
// contarem como da mesma linha.
const OverlapFraction = 0.5

// GroupLines agrupa palavras soltas em linhas de texto usando só a
// geometria (sobreposição vertical das caixas), nunca o conteúdo. Depois
// ordena as palavras de cada linha da esquerda para a direita, e as
// próprias linhas de cima para baixo -- a ordem de leitura de um bloco de
// texto de uma coluna só.
//
// Cada linha "abre" ancorada na faixa vertical [yMin,yMax] da primeira
// palavra atribuída a ela, e essa âncora não muda depois -- ao contrário
// de ir estendendo a faixa a cada palavra nova (o que deixaria a linha
// "derivar" verticalmente e engolir a linha vizinha de baixo aos poucos).
// Isso assume que as palavras de uma mesma linha têm altura razoavelmente
// uniforme; um título bem maior que o resto do parágrafo ao lado pode
// abrir a própria linha mesmo estando geometricamente colado.
//
// SÓ cobre layout de uma coluna: duas palavras na mesma faixa vertical
// caem na mesma linha mesmo que pertençam, na página de verdade, a colunas
// diferentes lado a lado (um cabeçalho "Item" e "Valor" de duas colunas
// distintas, por exemplo). Separar coluna de linha por geometria pura
// exige também olhar o espaçamento horizontal entre blocos de texto -- e
// esse limiar só dá para calibrar direito contra documento real, que este
// motor ainda não tem (a fase 3, detecção, não existe). Fica de fora agora
// em vez de arriscar um número chutado; ver documents/README.md.
func GroupLines(words []Word) []Line {
	if len(words) == 0 {
		return nil
	}

	ordenadas := append([]Word(nil), words...)
	sort.SliceStable(ordenadas, func(i, j int) bool {
		iMin, _ := ordenadas[i].yRange()
		jMin, _ := ordenadas[j].yRange()
		return iMin < jMin
	})

	type linhaAberta struct {
		ancoraMin, ancoraMax float64
		palavras             []Word
	}
	var abertas []linhaAberta

	for _, w := range ordenadas {
		wMin, wMax := w.yRange()

		melhor := -1
		melhorFracao := 0.0
		for i, l := range abertas {
			sobreposicao := overlap(wMin, wMax, l.ancoraMin, l.ancoraMax)
			menorAltura := math.Min(wMax-wMin, l.ancoraMax-l.ancoraMin)
			if menorAltura <= 0 {
				continue
			}
			fracao := sobreposicao / menorAltura
			if fracao > OverlapFraction && fracao > melhorFracao {
				melhor, melhorFracao = i, fracao
			}
		}

		if melhor == -1 {
			abertas = append(abertas, linhaAberta{ancoraMin: wMin, ancoraMax: wMax, palavras: []Word{w}})
		} else {
			abertas[melhor].palavras = append(abertas[melhor].palavras, w)
		}
	}

	linhas := make([]Line, len(abertas))
	for i, l := range abertas {
		palavras := append([]Word(nil), l.palavras...)
		sort.SliceStable(palavras, func(a, b int) bool {
			return palavras[a].xMin() < palavras[b].xMin()
		})
		linhas[i] = Line{Words: palavras}
	}
	sort.SliceStable(linhas, func(i, j int) bool {
		iMin, _ := linhas[i].Bounds()
		jMin, _ := linhas[j].Bounds()
		return iMin.Y < jMin.Y
	})
	return linhas
}

// overlap devolve o comprimento da interseção entre [aMin,aMax] e
// [bMin,bMax], ou 0 se não se tocarem.
func overlap(aMin, aMax, bMin, bMax float64) float64 {
	lo := math.Max(aMin, bMin)
	hi := math.Min(aMax, bMax)
	if hi <= lo {
		return 0
	}
	return hi - lo
}
