// Package dist define como a ERA MAPAS mede a distancia entre dois lugares,
// e serve a primeira resposta util antes de existir grafo rodoviario.
//
// O pacote gira em torno de uma interface, Distancer, e da razao dela
// existir: medir distancia rodoviaria tem varias implementacoes possiveis,
// com custos e exatidoes muito diferentes, e o sistema que consome nao
// deveria saber qual esta rodando.
//
// Hoje ha uma implementacao, Estimator: distancia em linha reta multiplicada
// por um fator de desvio calibrado com o historico da operacao. Erra na casa
// de 10%, custa nanossegundos e ja resolve agrupamento por regiao, raio de
// atendimento e pre-filtragem de candidatos. Na Fase 5 entra a implementacao
// que percorre o grafo rodoviario de verdade, e o sistema chamador troca uma
// linha de configuracao.
//
// A interface nao e enfeite arquitetural: e o que permite entregar valor na
// primeira semana sem hipotecar a decisao seguinte.
package dist

import (
	"context"
	"fmt"

	"github.com/iafrotamacedo-cloud/era/maps/geo"
)

// Leg e o resultado de uma medicao entre dois pontos.
//
// Distancia e tempo andam juntos porque quem roteiriza precisa dos dois e
// porque a fonte deles e a mesma. Em logistica o tempo costuma pesar mais que
// a quilometragem na decisao -- janela de entrega, jornada do motorista --,
// entao ele nao e um extra opcional.
type Leg struct {
	Meters  float64
	Seconds float64
}

// Distancer mede distancia e tempo rodoviarios entre pontos.
//
// As implementacoes vao de uma multiplicacao sobre a linha reta ate uma busca
// no grafo rodoviario. O contrato e o mesmo; o que muda e quanto custa e
// quanto erra.
//
// O context existe para as implementacoes caras: uma matriz de 5.000 por
// 5.000 sao 25 milhoes de pares, e quem pediu precisa poder desistir. As
// implementacoes baratas o ignoram, exceto para checar cancelamento.
//
// Um erro em Distance significa que nao ha caminho conhecido -- uma ilha, um
// trecho desconectado do grafo --, nao que o calculo falhou.
type Distancer interface {
	Distance(ctx context.Context, from, to geo.Point) (Leg, error)
	Matrix(ctx context.Context, origins, destinations []geo.Point) (*Matrix, error)
}

// Matrix guarda a distancia e o tempo de cada par origem-destino.
//
// E a estrutura que a roteirizacao consome: dado o deposito e as paradas do
// dia, ela pergunta uma vez e depois consulta milhares de vezes enquanto
// testa ordens de visita.
//
// Os valores sao guardados em float32, nao float64. Uma matriz 5.000 x 5.000
// tem 25 milhoes de celulas: 200 MB em float32 contra 400 MB em float64. A
// precisao que se perde e irrelevante -- float32 distingue centimetros numa
// distancia de 1.000 km --, e a metade da memoria decide se a matriz cabe.
type Matrix struct {
	Origins      []geo.Point
	Destinations []geo.Point

	meters  []float32
	seconds []float32
}

// NewMatrix aloca uma matriz zerada para os pontos dados.
//
// As fatias de pontos sao guardadas por referencia, nao copiadas: quem chama
// nao deve altera-las depois.
func NewMatrix(origins, destinations []geo.Point) *Matrix {
	n := len(origins) * len(destinations)
	return &Matrix{
		Origins:      origins,
		Destinations: destinations,
		meters:       make([]float32, n),
		seconds:      make([]float32, n),
	}
}

// Rows e Cols devolvem as dimensoes da matriz.
func (m *Matrix) Rows() int { return len(m.Origins) }
func (m *Matrix) Cols() int { return len(m.Destinations) }

// At devolve a medicao da origem i ate o destino j.
func (m *Matrix) At(i, j int) Leg {
	k := m.index(i, j)
	return Leg{Meters: float64(m.meters[k]), Seconds: float64(m.seconds[k])}
}

// Set grava a medicao da origem i ate o destino j.
func (m *Matrix) Set(i, j int, l Leg) {
	k := m.index(i, j)
	m.meters[k] = float32(l.Meters)
	m.seconds[k] = float32(l.Seconds)
}

func (m *Matrix) index(i, j int) int {
	if i < 0 || i >= len(m.Origins) || j < 0 || j >= len(m.Destinations) {
		panic(fmt.Sprintf("dist: celula (%d,%d) fora de uma matriz %dx%d",
			i, j, len(m.Origins), len(m.Destinations)))
	}
	return i*len(m.Destinations) + j
}
