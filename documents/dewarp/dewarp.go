// Package dewarp mede o quanto uma pagina fotografada esta deformada e
// decide qual dos quatro niveis de retificacao ela precisa.
//
// A ideia central esta em documents/README.md: detectar antes de retificar.
// Um detector de texto devolve poligonos, e a forma desses poligonos ja e a
// medida da deformacao -- um quadrilatero de 4 vertices diz "esta linha e
// reta"; um poligono com mais vertices, tracando separadamente a borda de
// cima e a de baixo de um texto curvo, diz exatamente quanto ele curva. Este
// pacote so le essa geometria; nao detecta nada e nao mexe em pixel.
//
// Nao ha um passo global de otimizacao aqui: a decisao e por linha
// (MeasureLine, Classify) e depois agregada pagina inteira (PageLevel),
// pegando sempre o pior caso -- retificar so uma parte da pagina nao ajuda
// quem precisa ler a pagina inteira.
package dewarp

import "fmt"

// Level identifica qual nivel de retificacao uma linha (ou a pagina)
// precisa. A ordem crescente (N0 < N1 < N2 < N3) importa: PageLevel usa
// comparacao direta para achar o pior caso entre varias linhas.
type Level int

const (
	N0 Level = iota // ja plana e alinhada -- nada a fazer
	N1              // plana, mas torta ou em perspectiva -- homografia resolve
	N2              // curva suave -- retificacao por linha (curva) resolve
	N3              // curvatura irregular demais para uma curva suave -- fica para a rede (nao implementado ainda)
)

func (l Level) String() string {
	switch l {
	case N0:
		return "N0"
	case N1:
		return "N1"
	case N2:
		return "N2"
	case N3:
		return "N3"
	default:
		return fmt.Sprintf("Level(%d)", int(l))
	}
}
