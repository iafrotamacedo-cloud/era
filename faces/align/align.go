// Package align recorta e endireita rostos.
//
// Entre detectar um rosto e gerar o vetor de identidade ha uma etapa que
// costuma ser subestimada: o alinhamento. A rede de reconhecimento foi
// treinada com rostos numa posicao canonica -- olhos numa altura fixa, boca
// noutra, tamanho padronizado. Alimentar essa rede com um rosto torto, ou
// maior, ou deslocado, degrada a acuracia muito mais do que a intuicao
// sugere.
//
// O alinhamento resolve isso com uma transformacao de similaridade: rotacao,
// escala uniforme e translacao, calculada a partir dos 5 pontos faciais de
// modo a aproximar ao maximo o gabarito canonico.
//
// Nao ha cisalhamento nem escala nao uniforme, de proposito. Uma
// transformacao afim completa conseguiria encaixar os 5 pontos exatamente,
// mas ao preco de distorcer o rosto -- e a distorcao mudaria a identidade
// que a rede enxerga.
package align

import (
	"fmt"
	"math"
)

// Point e uma coordenada em pixels.
//
// Usa float64 porque a transformacao passa por medias, produtos e uma
// divisao; acumular isso em float32 custaria precisao numa conta que roda
// uma vez por rosto e nao pesa no tempo total.
type Point struct {
	X, Y float64
}

// Landmarks sao os 5 pontos faciais, na ordem que o ArcFace e o YuNet usam.
//
//	0  olho direito da pessoa (a esquerda na imagem)
//	1  olho esquerdo da pessoa
//	2  ponta do nariz
//	3  canto direito da boca
//	4  canto esquerdo da boca
type Landmarks [5]Point

// Indices dos pontos, para quem preferir nomear.
const (
	OlhoDireito  = 0
	OlhoEsquerdo = 1
	Nariz        = 2
	BocaDireita  = 3
	BocaEsquerda = 4
)

// gabarito112 e a posicao canonica dos 5 pontos num recorte de 112x112.
//
// Sao os valores do ArcFace, usados por praticamente todo modelo de
// reconhecimento facial moderno -- inclusive os treinados a partir dele,
// como o SFace. Nao sao arbitrarios: vieram da media dos pontos sobre o
// conjunto de treino, e mudar um deles desalinha o rosto em relacao ao que a
// rede aprendeu.
var gabarito112 = Landmarks{
	{38.2946, 51.6963},
	{73.5318, 51.5014},
	{56.0252, 71.7366},
	{41.5493, 92.3655},
	{70.7299, 92.2041},
}

// Template devolve o gabarito canonico para um recorte quadrado de lado
// tamanho.
//
// O gabarito original e para 112x112; outros tamanhos sao obtidos por escala
// proporcional. Modelos de reconhecimento facial usam 112 quase sem excecao.
func Template(tamanho int) Landmarks {
	if tamanho == 112 {
		return gabarito112
	}

	k := float64(tamanho) / 112
	var out Landmarks
	for i, p := range gabarito112 {
		out[i] = Point{X: p.X * k, Y: p.Y * k}
	}
	return out
}

// Transform e uma transformacao de similaridade no plano:
//
//	x' = A*x - B*y + Tx
//	y' = B*x + A*y + Ty
//
// A matriz [A -B; B A] e uma rotacao multiplicada por uma escala uniforme.
// Essa forma -- e nao uma matriz 2x2 qualquer -- e o que garante que angulos
// sejam preservados e o rosto nao seja distorcido.
type Transform struct {
	A, B   float64
	Tx, Ty float64
}

// Identidade devolve a transformacao que nao muda nada.
func Identidade() Transform { return Transform{A: 1} }

// Apply aplica a transformacao a um ponto.
func (t Transform) Apply(p Point) Point {
	return Point{
		X: t.A*p.X - t.B*p.Y + t.Tx,
		Y: t.B*p.X + t.A*p.Y + t.Ty,
	}
}

// Escala devolve o fator de ampliacao da transformacao.
//
// Como [A -B; B A] e rotacao vezes escala, o fator e o modulo do numero
// complexo A + Bi.
func (t Transform) Escala() float64 { return math.Hypot(t.A, t.B) }

// Rotacao devolve o angulo de rotacao, em radianos.
func (t Transform) Rotacao() float64 { return math.Atan2(t.B, t.A) }

// Inverse devolve a transformacao inversa.
//
// E o que o recorte de fato usa: para cada pixel da saida, a inversa diz de
// onde na imagem original ele veio. Percorrer a saida e buscar na entrada --
// e nao o contrario -- garante que todo pixel do recorte receba valor, sem
// buracos.
func (t Transform) Inverse() (Transform, error) {
	// A inversa de z -> c*z + d, com c = A + Bi, e z -> (z - d)/c.
	den := t.A*t.A + t.B*t.B
	if den == 0 {
		return Transform{}, fmt.Errorf("align: transformacao degenerada (escala zero)")
	}

	a := t.A / den
	b := -t.B / den

	return Transform{
		A: a, B: b,
		Tx: -(a*t.Tx - b*t.Ty),
		Ty: -(b*t.Tx + a*t.Ty),
	}, nil
}

// String descreve a transformacao em termos legiveis.
func (t Transform) String() string {
	return fmt.Sprintf("escala %.3f, rotacao %.1f graus, translacao (%.1f, %.1f)",
		t.Escala(), t.Rotacao()*180/math.Pi, t.Tx, t.Ty)
}

// Similarity calcula a transformacao de similaridade que melhor leva src em
// dst, no sentido de minimizar a soma dos quadrados das distancias.
//
// # O truque dos numeros complexos
//
// Tratando cada ponto (x, y) como o numero complexo x + yi, uma
// transformacao de similaridade vira uma unica multiplicacao:
//
//	q = c*p + d
//
// onde c carrega rotacao e escala juntas, e d a translacao. Isso reduz o
// problema a um minimo quadrados de duas variaveis complexas, com solucao
// fechada:
//
//	c = soma(q_centrado * conjugado(p_centrado)) / soma(|p_centrado|^2)
//	d = media(q) - c * media(p)
//
// A alternativa usual -- decomposicao em valores singulares, pelo algoritmo
// de Umeyama -- da exatamente o mesmo resultado em duas dimensoes, com muito
// mais codigo. E a forma complexa tem uma vantagem de brinde: ela NAO
// consegue produzir reflexao, porque multiplicacao complexa e sempre rotacao
// pura. Um rosto espelhado seria uma solucao valida para o minimo quadrados,
// e uma catastrofe para o reconhecimento.
func Similarity(src, dst Landmarks) (Transform, error) {
	var mediaSrcX, mediaSrcY, mediaDstX, mediaDstY float64
	for i := range src {
		mediaSrcX += src[i].X
		mediaSrcY += src[i].Y
		mediaDstX += dst[i].X
		mediaDstY += dst[i].Y
	}
	n := float64(len(src))
	mediaSrcX /= n
	mediaSrcY /= n
	mediaDstX /= n
	mediaDstY /= n

	// numerador = soma(q_c * conj(p_c));  denominador = soma(|p_c|^2)
	var numRe, numIm, den float64
	for i := range src {
		px := src[i].X - mediaSrcX
		py := src[i].Y - mediaSrcY
		qx := dst[i].X - mediaDstX
		qy := dst[i].Y - mediaDstY

		// (qx + qy*i) * (px - py*i)
		numRe += qx*px + qy*py
		numIm += qy*px - qx*py

		den += px*px + py*py
	}

	if den == 0 {
		return Transform{}, fmt.Errorf("align: os 5 pontos de origem sao coincidentes")
	}

	a := numRe / den
	b := numIm / den

	return Transform{
		A: a, B: b,
		Tx: mediaDstX - (a*mediaSrcX - b*mediaSrcY),
		Ty: mediaDstY - (b*mediaSrcX + a*mediaSrcY),
	}, nil
}

// Para calcula a transformacao que leva os pontos detectados ao gabarito
// canonico de um recorte quadrado de lado tamanho.
//
// E o atalho para o uso normal: detectou os 5 pontos, quer o recorte.
func Para(pontos Landmarks, tamanho int) (Transform, error) {
	return Similarity(pontos, Template(tamanho))
}
