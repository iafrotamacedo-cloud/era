package dewarp

import (
	"fmt"

	"github.com/iafrotamacedo-cloud/era/documents/geom"
)

// ExtractBaseline devolve os pontos que aproximam a base do texto -- onde
// as letras se apoiam -- a partir do poligono que delimita uma linha
// detectada, da esquerda para a direita.
//
// Segue a convencao usual de deteccao de texto curvo (DBNet e os conjuntos
// que o treinam, TotalText e CTW1500): os vertices do poligono sao a borda
// de cima, da esquerda para a direita, seguidos pela borda de baixo, da
// direita para a esquerda -- as duas metades tem o mesmo tanto de pontos.
// A baseline e a segunda metade invertida.
//
// Um quadrilatero simples (4 vertices: superior-esquerdo, superior-direito,
// inferior-direito, inferior-esquerdo -- o caso comum de uma linha sem
// curvatura) cai no mesmo esquema: a segunda metade e [inferior-direito,
// inferior-esquerdo], invertida vira [inferior-esquerdo, inferior-direito].
// So 2 pontos de baseline nao bastam para medir curvatura -- e correto:
// um detector que devolveu so 4 vertices ja esta dizendo que considera a
// linha reta. MeasureLine trata esse caso sem erro.
func ExtractBaseline(p geom.Polygon) ([]geom.Point, error) {
	n := len(p)
	if n < 4 || n%2 != 0 {
		return nil, fmt.Errorf("dewarp: poligono de linha precisa de numero par de vertices >= 4, tem %d", n)
	}

	metade := n / 2
	borda := p[metade:]
	baseline := make([]geom.Point, metade)
	for i, pt := range borda {
		baseline[metade-1-i] = pt
	}
	return baseline, nil
}
