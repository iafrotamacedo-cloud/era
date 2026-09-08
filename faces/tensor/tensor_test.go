package tensor

import (
	"reflect"
	"testing"
)

func TestNewShapeStrides(t *testing.T) {
	x := New(2, 3, 4)

	if got := x.Size(); got != 24 {
		t.Errorf("Size() = %d, quero 24", got)
	}
	if got := x.Rank(); got != 3 {
		t.Errorf("Rank() = %d, quero 3", got)
	}
	if want := []int{12, 4, 1}; !reflect.DeepEqual(x.Strides, want) {
		t.Errorf("Strides = %v, quero %v", x.Strides, want)
	}
	if !x.IsContiguous() {
		t.Error("tensor recem-criado deveria ser contiguo")
	}
	for i, v := range x.Data {
		if v != 0 {
			t.Fatalf("New deveria zerar, mas Data[%d] = %v", i, v)
		}
	}
}

func TestAtSet(t *testing.T) {
	x := New(2, 3)
	x.Set(7, 1, 2)

	if got := x.At(1, 2); got != 7 {
		t.Errorf("At(1,2) = %v, quero 7", got)
	}
	// posicao plana esperada: 1*3 + 2 = 5
	if x.Data[5] != 7 {
		t.Errorf("Data[5] = %v, quero 7 (strides errados?)", x.Data[5])
	}
	if got := x.At(0, 0); got != 0 {
		t.Errorf("At(0,0) = %v, quero 0", got)
	}
}

func TestAtPanicsForaDosLimites(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Error("At fora dos limites deveria entrar em panico")
		}
	}()
	New(2, 2).At(2, 0)
}

func TestFromSliceValidaTamanho(t *testing.T) {
	if _, err := FromSlice([]float32{1, 2, 3}, 2, 2); err == nil {
		t.Error("FromSlice deveria recusar 3 elementos numa forma 2x2")
	}
	if _, err := FromSlice([]float32{1, 2, 3, 4}, 2, 2); err != nil {
		t.Errorf("FromSlice recusou dado valido: %v", err)
	}
}

func TestReshape(t *testing.T) {
	x := MustFromSlice([]float32{1, 2, 3, 4, 5, 6}, 2, 3)

	y, err := x.Reshape(3, 2)
	if err != nil {
		t.Fatalf("Reshape: %v", err)
	}
	if want := []int{3, 2}; !reflect.DeepEqual(y.Shape, want) {
		t.Errorf("Shape = %v, quero %v", y.Shape, want)
	}
	if got := y.At(2, 1); got != 6 {
		t.Errorf("At(2,1) = %v, quero 6", got)
	}

	// Reshape e uma view: mexer num reflete no outro.
	y.Set(99, 0, 0)
	if got := x.At(0, 0); got != 99 {
		t.Errorf("Reshape copiou dados; x.At(0,0) = %v, quero 99", got)
	}
}

func TestReshapeInfereDimensao(t *testing.T) {
	x := New(2, 3, 4)

	y, err := x.Reshape(6, -1)
	if err != nil {
		t.Fatalf("Reshape: %v", err)
	}
	if want := []int{6, 4}; !reflect.DeepEqual(y.Shape, want) {
		t.Errorf("Shape = %v, quero %v", y.Shape, want)
	}

	if _, err := x.Reshape(-1, -1); err == nil {
		t.Error("duas dimensoes -1 deveriam dar erro")
	}
	if _, err := x.Reshape(5, -1); err == nil {
		t.Error("24 nao divide por 5; deveria dar erro")
	}
	if _, err := x.Reshape(2, 3); err == nil {
		t.Error("forma com menos elementos deveria dar erro")
	}
}

func TestTranspose(t *testing.T) {
	// 2x3:  1 2 3
	//       4 5 6
	x := MustFromSlice([]float32{1, 2, 3, 4, 5, 6}, 2, 3)

	y, err := x.Transpose()
	if err != nil {
		t.Fatalf("Transpose: %v", err)
	}
	if want := []int{3, 2}; !reflect.DeepEqual(y.Shape, want) {
		t.Errorf("Shape = %v, quero %v", y.Shape, want)
	}
	if y.IsContiguous() {
		t.Error("uma transposta 2x3 nao deveria ser contigua")
	}

	// 3x2:  1 4
	//       2 5
	//       3 6
	want := []float32{1, 4, 2, 5, 3, 6}
	if got := y.Flat(); !reflect.DeepEqual(got, want) {
		t.Errorf("Flat() = %v, quero %v", got, want)
	}
	if got := y.At(2, 0); got != 3 {
		t.Errorf("At(2,0) = %v, quero 3", got)
	}
}

func TestTransposePermutacaoInvalida(t *testing.T) {
	x := New(2, 3, 4)
	for _, perm := range [][]int{{0, 0, 1}, {0, 1}, {0, 1, 3}, {-1, 0, 1}} {
		if _, err := x.Transpose(perm...); err == nil {
			t.Errorf("Transpose(%v) deveria dar erro", perm)
		}
	}
}

func TestContiguousMaterializa(t *testing.T) {
	x := MustFromSlice([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
	y, _ := x.Transpose()

	z := y.Contiguous()
	if !z.IsContiguous() {
		t.Error("Contiguous() devolveu tensor nao contiguo")
	}
	if want := []float32{1, 4, 2, 5, 3, 6}; !reflect.DeepEqual(z.Data, want) {
		t.Errorf("Data = %v, quero %v", z.Data, want)
	}

	// Materializou: agora e uma copia independente.
	z.Set(99, 0, 0)
	if got := x.At(0, 0); got != 1 {
		t.Errorf("Contiguous nao desacoplou; x.At(0,0) = %v, quero 1", got)
	}

	// Ja contiguo: devolve o mesmo ponteiro, sem copiar.
	if x.Contiguous() != x {
		t.Error("Contiguous() copiou um tensor que ja era contiguo")
	}
}

func TestReshapeExigeContiguo(t *testing.T) {
	x := New(2, 3)
	y, _ := x.Transpose()
	if _, err := y.Reshape(6); err == nil {
		t.Error("Reshape de tensor nao contiguo deveria dar erro")
	}
}

func TestFillEClone(t *testing.T) {
	x := New(2, 2)
	x.Fill(3)
	for i, v := range x.Data {
		if v != 3 {
			t.Fatalf("Fill: Data[%d] = %v, quero 3", i, v)
		}
	}

	c := x.Clone()
	c.Set(9, 0, 0)
	if x.At(0, 0) != 3 {
		t.Error("Clone nao criou copia independente")
	}
}

func TestIsContiguousDimensaoUnitaria(t *testing.T) {
	// Dimensoes de tamanho 1 aceitam qualquer stride sem quebrar a
	// contiguidade -- nunca se anda por elas.
	x := New(1, 4, 1)
	if !x.IsContiguous() {
		t.Errorf("shape %v strides %v deveria ser contiguo", x.Shape, x.Strides)
	}
}

func TestSameShape(t *testing.T) {
	a, b, c := New(2, 3), New(2, 3), New(3, 2)
	if !a.SameShape(b) {
		t.Error("2x3 e 2x3 deveriam ter a mesma forma")
	}
	if a.SameShape(c) {
		t.Error("2x3 e 3x2 nao deveriam ter a mesma forma")
	}
	if a.SameShape(New(2, 3, 1)) {
		t.Error("ranks diferentes nao deveriam bater")
	}
}
