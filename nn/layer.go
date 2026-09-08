package nn

import (
	"fmt"
	"strings"

	"github.com/iafrotamacedo-cloud/era/tensor"
)

// Layer e uma etapa da rede.
//
// A convencao de formas segue a do ONNX:
//
//	imagens  [N, C, H, W]   lote, canais, altura, largura
//	vetores  [N, F]         lote, atributos
//
// N e o tamanho do lote. Reconhecimento facial normalmente processa um rosto
// por vez, mas uma foto com varias pessoas rende varios rostos de uma vez --
// e ai o lote paga.
type Layer interface {
	// Forward calcula a saida para x.
	//
	// O tensor devolvido costuma apontar para memoria do Workspace, e so e
	// valido ate o proximo Reset. Copie se precisar guardar.
	Forward(ws *Workspace, x *tensor.Tensor) (*tensor.Tensor, error)

	// OutputShape informa a forma da saida para uma entrada com a forma
	// dada, sem executar nada. Serve para validar uma rede antes de rodar.
	OutputShape(in []int) ([]int, error)

	// Name identifica a camada em mensagens de erro.
	Name() string
}

// checkRank confere o numero de dimensoes de um tensor.
func checkRank(layer string, x *tensor.Tensor, rank int) error {
	if x.Rank() != rank {
		return fmt.Errorf("nn: %s espera tensor de rank %d, recebeu %v", layer, rank, x.Shape)
	}
	return nil
}

// Sequential encadeia camadas, passando a saida de uma como entrada da
// seguinte.
//
// Cobre redes em linha reta. Topologias com desvio -- conexoes residuais,
// que o ArcFace usa -- precisam do executor de grafo da Fase 3. Sequential
// continua util depois disso para descrever os trechos lineares.
type Sequential struct {
	name   string
	Layers []Layer
}

// NewSequential monta uma sequencia de camadas.
func NewSequential(name string, layers ...Layer) *Sequential {
	return &Sequential{name: name, Layers: layers}
}

// Name identifica a sequencia.
func (s *Sequential) Name() string {
	if s.name == "" {
		return "Sequential"
	}
	return s.name
}

// Add acrescenta camadas ao fim da sequencia.
func (s *Sequential) Add(layers ...Layer) { s.Layers = append(s.Layers, layers...) }

// Forward passa x por todas as camadas, em ordem.
func (s *Sequential) Forward(ws *Workspace, x *tensor.Tensor) (*tensor.Tensor, error) {
	cur := x
	for i, l := range s.Layers {
		out, err := l.Forward(ws, cur)
		if err != nil {
			return nil, fmt.Errorf("nn: %s, camada %d (%s): %w", s.Name(), i, l.Name(), err)
		}
		cur = out
	}
	return cur, nil
}

// OutputShape percorre a sequencia calculando as formas, sem executar nada.
func (s *Sequential) OutputShape(in []int) ([]int, error) {
	cur := in
	for i, l := range s.Layers {
		out, err := l.OutputShape(cur)
		if err != nil {
			return nil, fmt.Errorf("nn: %s, camada %d (%s): %w", s.Name(), i, l.Name(), err)
		}
		cur = out
	}
	return cur, nil
}

// Fuse funde cada BatchNorm no Conv2D imediatamente anterior e remove a
// camada fundida da sequencia.
//
// Na inferencia, BatchNorm e apenas y = x*escala + deslocamento por canal, e
// os dois cabem dentro dos pesos da convolucao:
//
//	conv:  y = W*x + b
//	bn:    z = y*escala + deslocamento
//	       z = (W*escala)*x + (b*escala + deslocamento)
//
// A camada some do tempo de execucao sem mudar o resultado. Numa rede cheia
// de blocos conv-bn -- que e o caso das redes de reconhecimento facial --
// isso elimina uma passagem inteira sobre os dados por bloco.
//
// Devolve quantas camadas foram fundidas. Depois de fundir, os pesos das
// convolucoes mudaram: nao chame duas vezes.
func (s *Sequential) Fuse() (int, error) {
	var out []Layer
	fundidas := 0

	for _, l := range s.Layers {
		bn, ehBN := l.(*BatchNorm)
		if !ehBN || len(out) == 0 {
			out = append(out, l)
			continue
		}

		conv, ehConv := out[len(out)-1].(*Conv2D)
		if !ehConv {
			out = append(out, l)
			continue
		}

		if err := conv.FuseBatchNorm(bn); err != nil {
			return fundidas, fmt.Errorf("nn: %s: %w", s.Name(), err)
		}
		fundidas++
	}

	s.Layers = out
	return fundidas, nil
}

// String descreve a sequencia camada a camada.
func (s *Sequential) String() string {
	var b strings.Builder
	fmt.Fprintf(&b, "%s:\n", s.Name())
	for i, l := range s.Layers {
		fmt.Fprintf(&b, "  %2d  %s\n", i, l.Name())
	}
	return b.String()
}

// Flatten achata as dimensoes espaciais, transformando [N, C, H, W] em
// [N, C*H*W].
//
// E a transicao entre a parte convolucional e a camada linear final que
// produz o vetor de identidade.
type Flatten struct{}

// Name identifica a camada.
func (f *Flatten) Name() string { return "Flatten" }

// OutputShape colapsa tudo depois da primeira dimensao.
func (f *Flatten) OutputShape(in []int) ([]int, error) {
	if len(in) < 2 {
		return nil, fmt.Errorf("nn: Flatten espera rank >= 2, recebeu %v", in)
	}
	n := in[0]
	resto := 1
	for _, d := range in[1:] {
		resto *= d
	}
	return []int{n, resto}, nil
}

// Forward reinterpreta o tensor, sem mover dados.
func (f *Flatten) Forward(ws *Workspace, x *tensor.Tensor) (*tensor.Tensor, error) {
	shape, err := f.OutputShape(x.Shape)
	if err != nil {
		return nil, err
	}
	// Reshape exige contiguidade; Contiguous nao copia se ja estiver ok.
	return x.Contiguous().Reshape(shape...)
}
