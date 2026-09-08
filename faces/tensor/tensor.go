// Package tensor fornece o array N-dimensional que sustenta toda a ERA.
//
// Um Tensor guarda um bloco plano de float32 mais dois metadados: a forma
// (Shape) e o passo (Strides). Guardar strides -- em vez de arrays aninhados
// -- permite remodelar, transpor e fatiar sem copiar dado nenhum: apenas os
// metadados mudam. Numa rede neural, que faz isso centenas de vezes por
// imagem, essa diferenca decide o desempenho do projeto inteiro.
package tensor

import (
	"fmt"
	"strings"
)

// Tensor e um array N-dimensional de float32.
//
// Data pode ser maior que o tensor logico quando este e uma view de outro
// tensor. Use sempre Size() para saber quantos elementos o tensor tem,
// nunca len(Data).
type Tensor struct {
	Data    []float32 // buffer subjacente, possivelmente compartilhado
	Shape   []int     // tamanho de cada dimensao
	Strides []int     // quantos float32 pular para andar 1 em cada dimensao
	Offset  int       // onde este tensor comeca dentro de Data
}

// New cria um tensor zerado com a forma dada.
//
//	New(2, 3, 4) // tensor 2x3x4, 24 elementos em zero
func New(shape ...int) *Tensor {
	n := 1
	for _, d := range shape {
		if d < 0 {
			panic(fmt.Sprintf("tensor: dimensao negativa em %v", shape))
		}
		n *= d
	}
	return &Tensor{
		Data:    make([]float32, n),
		Shape:   append([]int(nil), shape...),
		Strides: contiguousStrides(shape),
	}
}

// FromSlice constroi um tensor sobre um slice ja existente, sem copiar.
// O slice passa a ser propriedade do tensor.
func FromSlice(data []float32, shape ...int) (*Tensor, error) {
	n := 1
	for _, d := range shape {
		if d < 0 {
			return nil, fmt.Errorf("tensor: dimensao negativa em %v", shape)
		}
		n *= d
	}
	if len(data) != n {
		return nil, fmt.Errorf("tensor: forma %v pede %d elementos, recebi %d", shape, n, len(data))
	}
	return &Tensor{
		Data:    data,
		Shape:   append([]int(nil), shape...),
		Strides: contiguousStrides(shape),
	}, nil
}

// MustFromSlice e como FromSlice, mas entra em panico no erro.
// Destinado a testes e a dados constantes conhecidos em tempo de escrita.
func MustFromSlice(data []float32, shape ...int) *Tensor {
	t, err := FromSlice(data, shape...)
	if err != nil {
		panic(err)
	}
	return t
}

// Size devolve o numero total de elementos do tensor.
func (t *Tensor) Size() int {
	n := 1
	for _, d := range t.Shape {
		n *= d
	}
	return n
}

// Rank devolve o numero de dimensoes.
func (t *Tensor) Rank() int { return len(t.Shape) }

// IsContiguous informa se os elementos estao dispostos em Data na ordem
// natural (ultima dimensao variando mais rapido), sem buracos.
//
// So um tensor contiguo pode ser passado direto para os kernels numericos.
func (t *Tensor) IsContiguous() bool {
	acc := 1
	for i := len(t.Shape) - 1; i >= 0; i-- {
		if t.Shape[i] == 1 {
			continue // dimensao de tamanho 1 aceita qualquer stride
		}
		if t.Strides[i] != acc {
			return false
		}
		acc *= t.Shape[i]
	}
	return true
}

// flatIndex traduz um indice multidimensional na posicao dentro de Data.
func (t *Tensor) flatIndex(idx []int) int {
	if len(idx) != len(t.Shape) {
		panic(fmt.Sprintf("tensor: indice %v nao tem o rank da forma %v", idx, t.Shape))
	}
	off := t.Offset
	for i, v := range idx {
		if v < 0 || v >= t.Shape[i] {
			panic(fmt.Sprintf("tensor: indice %v fora da forma %v", idx, t.Shape))
		}
		off += v * t.Strides[i]
	}
	return off
}

// At le um elemento pelo indice multidimensional.
func (t *Tensor) At(idx ...int) float32 { return t.Data[t.flatIndex(idx)] }

// Set escreve um elemento pelo indice multidimensional.
func (t *Tensor) Set(v float32, idx ...int) { t.Data[t.flatIndex(idx)] = v }

// Reshape devolve uma view com outra forma, sem copiar dados.
//
// Exige que o tensor seja contiguo. Uma unica dimensao pode ser -1, e sera
// inferida a partir das demais.
func (t *Tensor) Reshape(shape ...int) (*Tensor, error) {
	if !t.IsContiguous() {
		return nil, fmt.Errorf("tensor: Reshape exige tensor contiguo (chame Contiguous() antes)")
	}

	total := t.Size()
	inferAt := -1
	known := 1
	for i, d := range shape {
		switch {
		case d == -1:
			if inferAt >= 0 {
				return nil, fmt.Errorf("tensor: so uma dimensao pode ser -1, recebi %v", shape)
			}
			inferAt = i
		case d < 0:
			return nil, fmt.Errorf("tensor: dimensao invalida %d em %v", d, shape)
		default:
			known *= d
		}
	}

	out := append([]int(nil), shape...)
	if inferAt >= 0 {
		if known == 0 || total%known != 0 {
			return nil, fmt.Errorf("tensor: nao da para inferir -1 em %v a partir de %d elementos", shape, total)
		}
		out[inferAt] = total / known
		known *= out[inferAt]
	}
	if known != total {
		return nil, fmt.Errorf("tensor: forma %v pede %d elementos, o tensor tem %d", shape, known, total)
	}

	return &Tensor{
		Data:    t.Data,
		Shape:   out,
		Strides: contiguousStrides(out),
		Offset:  t.Offset,
	}, nil
}

// Transpose devolve uma view com as dimensoes reordenadas, sem copiar dados.
//
// perm deve ser uma permutacao de 0..Rank-1. Sem argumentos, inverte a ordem
// de todas as dimensoes.
func (t *Tensor) Transpose(perm ...int) (*Tensor, error) {
	r := t.Rank()
	if len(perm) == 0 {
		perm = make([]int, r)
		for i := range perm {
			perm[i] = r - 1 - i
		}
	}
	if len(perm) != r {
		return nil, fmt.Errorf("tensor: permutacao %v nao tem o rank %d", perm, r)
	}

	seen := make([]bool, r)
	for _, p := range perm {
		if p < 0 || p >= r || seen[p] {
			return nil, fmt.Errorf("tensor: %v nao e uma permutacao valida de 0..%d", perm, r-1)
		}
		seen[p] = true
	}

	shape := make([]int, r)
	strides := make([]int, r)
	for i, p := range perm {
		shape[i] = t.Shape[p]
		strides[i] = t.Strides[p]
	}
	return &Tensor{Data: t.Data, Shape: shape, Strides: strides, Offset: t.Offset}, nil
}

// Contiguous devolve um tensor equivalente com dados contiguos comecando em
// zero. Se o tensor ja atende a isso, devolve ele mesmo sem copiar.
func (t *Tensor) Contiguous() *Tensor {
	if t.Offset == 0 && t.IsContiguous() && len(t.Data) == t.Size() {
		return t
	}
	return t.Clone()
}

// Clone devolve uma copia independente e contigua do tensor.
func (t *Tensor) Clone() *Tensor {
	out := New(t.Shape...)
	n := t.Size()
	idx := make([]int, t.Rank())
	for i := 0; i < n; i++ {
		out.Data[i] = t.Data[t.flatIndex(idx)]
		incIndex(idx, t.Shape)
	}
	return out
}

// Fill escreve o mesmo valor em todos os elementos.
func (t *Tensor) Fill(v float32) {
	n := t.Size()
	idx := make([]int, t.Rank())
	for i := 0; i < n; i++ {
		t.Data[t.flatIndex(idx)] = v
		incIndex(idx, t.Shape)
	}
}

// Flat devolve os elementos em ordem de leitura natural, como slice plano.
// Se o tensor ja for contiguo e comecar em zero, nao ha copia.
func (t *Tensor) Flat() []float32 {
	c := t.Contiguous()
	return c.Data[:c.Size()]
}

// SameShape informa se dois tensores tem exatamente a mesma forma.
func (t *Tensor) SameShape(o *Tensor) bool {
	if t.Rank() != o.Rank() {
		return false
	}
	for i := range t.Shape {
		if t.Shape[i] != o.Shape[i] {
			return false
		}
	}
	return true
}

// String descreve o tensor de forma legivel, truncando tensores grandes.
func (t *Tensor) String() string {
	var b strings.Builder
	fmt.Fprintf(&b, "Tensor%v [", t.Shape)
	flat := t.Flat()
	const max = 12
	n := len(flat)
	for i := 0; i < n && i < max; i++ {
		if i > 0 {
			b.WriteString(" ")
		}
		fmt.Fprintf(&b, "%.4g", flat[i])
	}
	if n > max {
		fmt.Fprintf(&b, " ... +%d", n-max)
	}
	b.WriteString("]")
	return b.String()
}

// contiguousStrides calcula os passos naturais para uma forma: a ultima
// dimensao anda de 1 em 1, e cada dimensao a esquerda anda pelo produto de
// todas as dimensoes a sua direita.
//
//	shape   [2, 3, 4]
//	strides [12, 4, 1]
func contiguousStrides(shape []int) []int {
	s := make([]int, len(shape))
	acc := 1
	for i := len(shape) - 1; i >= 0; i-- {
		s[i] = acc
		acc *= shape[i]
	}
	return s
}

// incIndex avanca um indice multidimensional em uma posicao, na ordem de
// leitura natural. Serve para percorrer um tensor de strides arbitrarios.
func incIndex(idx, shape []int) {
	for d := len(idx) - 1; d >= 0; d-- {
		idx[d]++
		if idx[d] < shape[d] {
			return
		}
		idx[d] = 0
	}
}
