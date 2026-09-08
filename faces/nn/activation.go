package nn

import (
	"fmt"

	"github.com/iafrotamacedo-cloud/era/faces/tensor"
)

// PReLU e a ativacao usada pelas redes de reconhecimento facial:
//
//	y = x           se x > 0
//	y = alfa * x    se x <= 0
//
// A ReLU comum zera tudo que e negativo, e neuronios que ficam sempre
// negativos param de aprender de vez. A PReLU deixa passar uma fracao, e
// essa fracao e aprendida junto com o resto -- um alfa por canal.
//
// E o que ArcFace e MobileFaceNet usam, e por isso a ERA precisa dela.
type PReLU struct {
	name string

	// Alpha tem um elemento por canal, ou um so, compartilhado por todos.
	Alpha []float32
}

// NewPReLU monta a ativacao. alpha precisa ter um elemento por canal, ou
// exatamente um, para ser compartilhado.
func NewPReLU(name string, alpha []float32) (*PReLU, error) {
	if len(alpha) == 0 {
		return nil, fmt.Errorf("nn: %s: alpha vazio", name)
	}
	return &PReLU{name: name, Alpha: append([]float32(nil), alpha...)}, nil
}

// Name identifica a camada.
func (p *PReLU) Name() string {
	if p.name == "" {
		return "PReLU"
	}
	return p.name
}

// OutputShape: a ativacao preserva a forma.
func (p *PReLU) OutputShape(in []int) ([]int, error) {
	if len(in) < 2 {
		return nil, fmt.Errorf("nn: %s espera rank >= 2, recebeu %v", p.Name(), in)
	}
	if len(p.Alpha) != 1 && len(p.Alpha) != in[1] {
		return nil, fmt.Errorf("nn: %s tem %d alfas, a entrada tem %d canais",
			p.Name(), len(p.Alpha), in[1])
	}
	return append([]int(nil), in...), nil
}

// Forward aplica a ativacao.
func (p *PReLU) Forward(ws *Workspace, x *tensor.Tensor) (*tensor.Tensor, error) {
	if _, err := p.OutputShape(x.Shape); err != nil {
		return nil, err
	}

	src := x.Flat()
	out := ws.Tensor(x.Shape...)

	// Alfa unico: um laco so sobre tudo.
	if len(p.Alpha) == 1 {
		a := p.Alpha[0]
		for i, v := range src {
			if v > 0 {
				out.Data[i] = v
			} else {
				out.Data[i] = a * v
			}
		}
		return out, nil
	}

	n, c := x.Shape[0], x.Shape[1]
	espacial := 1
	for _, d := range x.Shape[2:] {
		espacial *= d
	}

	for i := 0; i < n; i++ {
		for ch := 0; ch < c; ch++ {
			a := p.Alpha[ch]
			base := (i*c + ch) * espacial

			in := src[base : base+espacial]
			o := out.Data[base : base+espacial]
			for k, v := range in {
				if v > 0 {
					o[k] = v
				} else {
					o[k] = a * v
				}
			}
		}
	}

	return out, nil
}

// ReLU zera tudo que e negativo. Nao aparece no ArcFace, mas aparece em
// outras arquiteturas que a ERA precisa conseguir carregar.
type ReLU struct{}

// Name identifica a camada.
func (r *ReLU) Name() string { return "ReLU" }

// OutputShape: a ativacao preserva a forma.
func (r *ReLU) OutputShape(in []int) ([]int, error) { return append([]int(nil), in...), nil }

// Forward aplica a ativacao.
func (r *ReLU) Forward(ws *Workspace, x *tensor.Tensor) (*tensor.Tensor, error) {
	src := x.Flat()
	out := ws.Tensor(x.Shape...)
	for i, v := range src {
		if v > 0 {
			out.Data[i] = v
		} else {
			out.Data[i] = 0
		}
	}
	return out, nil
}

// Add soma dois tensores elemento a elemento.
//
// E o que fecha uma conexao residual: o caminho principal e o atalho se
// encontram aqui. Nao implementa Layer, porque Layer tem uma entrada so --
// o executor de grafo da Fase 3 e que vai saber alimentar as duas.
func Add(ws *Workspace, a, b *tensor.Tensor) (*tensor.Tensor, error) {
	if !a.SameShape(b) {
		return nil, fmt.Errorf("nn: Add recebeu formas diferentes, %v e %v", a.Shape, b.Shape)
	}

	af, bf := a.Flat(), b.Flat()
	out := ws.Tensor(a.Shape...)
	for i, v := range af {
		out.Data[i] = v + bf[i]
	}
	return out, nil
}
