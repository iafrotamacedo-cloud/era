package kernel

import (
	"fmt"
	"math/rand"
	"testing"
)

// tolerancia combina um piso absoluto com um termo relativo. As duas
// implementacoes somam os produtos em ordens diferentes, e float32 acumula
// arredondamento -- divergir na 5a casa e esperado, nao e bug.
func closeEnough(got, want float32) bool {
	d := got - want
	if d < 0 {
		d = -d
	}
	m := want
	if m < 0 {
		m = -m
	}
	return d <= 1e-4+1e-4*m
}

func randSlice(r *rand.Rand, n int) []float32 {
	s := make([]float32, n)
	for i := range s {
		s[i] = r.Float32()*2 - 1 // [-1, 1)
	}
	return s
}

func TestMatMulContraReferencia(t *testing.T) {
	casos := []struct{ m, k, n int }{
		{1, 1, 1},
		{2, 3, 4},
		{4, 4, 4},
		{7, 5, 3},    // dimensoes primas, nada alinha com os blocos
		{64, 64, 64}, // exatamente um bloco
		{65, 65, 65}, // um a mais que o bloco: pega erro de borda
		{63, 129, 257},
		{128, 256, 64},
		{1, 512, 1},   // vetor x vetor
		{512, 1, 512}, // produto externo
		{100, 200, 300},
	}

	r := rand.New(rand.NewSource(42))

	for _, c := range casos {
		t.Run(fmt.Sprintf("%dx%dx%d", c.m, c.k, c.n), func(t *testing.T) {
			a := randSlice(r, c.m*c.k)
			b := randSlice(r, c.k*c.n)

			got := make([]float32, c.m*c.n)
			want := make([]float32, c.m*c.n)

			MatMul(a, b, got, c.m, c.k, c.n)
			MatMulRef(a, b, want, c.m, c.k, c.n)

			for i := range want {
				if !closeEnough(got[i], want[i]) {
					t.Fatalf("C[%d] = %v, referencia diz %v (linha %d, coluna %d)",
						i, got[i], want[i], i/c.n, i%c.n)
				}
			}
		})
	}
}

func TestMatMulIdentidade(t *testing.T) {
	const n = 37

	r := rand.New(rand.NewSource(7))
	a := randSlice(r, n*n)

	id := make([]float32, n*n)
	for i := 0; i < n; i++ {
		id[i*n+i] = 1
	}

	got := make([]float32, n*n)
	MatMul(a, id, got, n, n, n)

	for i := range a {
		if !closeEnough(got[i], a[i]) {
			t.Fatalf("A x I nao devolveu A: posicao %d deu %v, quero %v", i, got[i], a[i])
		}
	}
}

func TestMatMulSobrescreveC(t *testing.T) {
	// MatMul precisa zerar C sozinha. Se ela acumulasse, o lixo anterior
	// contaminaria o resultado -- e esse bug so aparece em buffers reusados.
	a := []float32{1, 2, 3, 4}
	b := []float32{1, 0, 0, 1}

	c := []float32{999, 999, 999, 999}
	MatMul(a, b, c, 2, 2, 2)

	for i, want := range a {
		if !closeEnough(c[i], want) {
			t.Fatalf("C[%d] = %v, quero %v (MatMul nao zerou C)", i, c[i], want)
		}
	}
}

func TestMatMulMesmoResultadoEmQualquerParalelismo(t *testing.T) {
	// O resultado nao pode depender de quantas goroutines rodaram. Cada uma
	// escreve so nas suas linhas, entao a divisao nao muda conta nenhuma.
	const m, k, n = 133, 97, 71

	r := rand.New(rand.NewSource(99))
	a := randSlice(r, m*k)
	b := randSlice(r, k*n)

	original := Threads()
	defer SetThreads(original)

	SetThreads(1)
	serial := make([]float32, m*n)
	MatMul(a, b, serial, m, k, n)

	for _, nt := range []int{2, 3, 8, 64} {
		SetThreads(nt)
		got := make([]float32, m*n)
		MatMul(a, b, got, m, k, n)

		for i := range serial {
			if got[i] != serial[i] {
				t.Fatalf("%d threads: C[%d] = %v, serial deu %v", nt, i, got[i], serial[i])
			}
		}
	}
}

func TestMatMulDimensoesZeradas(t *testing.T) {
	buf := make([]float32, 16)

	// Nao deve entrar em panico nem escrever fora.
	MatMul(buf, buf, buf, 0, 4, 4)
	MatMul(buf, buf, buf, 4, 4, 0)

	// k == 0 significa somar zero termos: a saida e a matriz zero.
	c := []float32{1, 2, 3, 4}
	MatMul(buf, buf, c, 2, 0, 2)
	for i, v := range c {
		if v != 0 {
			t.Errorf("k=0 deveria zerar C, mas C[%d] = %v", i, v)
		}
	}
}

func TestMatMulBufferPequenoEntraEmPanico(t *testing.T) {
	casos := []struct {
		nome    string
		a, b, c int
	}{
		{"A curto", 3, 16, 16},
		{"B curto", 16, 3, 16},
		{"C curto", 16, 16, 3},
	}

	for _, tc := range casos {
		t.Run(tc.nome, func(t *testing.T) {
			defer func() {
				if recover() == nil {
					t.Error("deveria entrar em panico com buffer pequeno demais")
				}
			}()
			MatMul(make([]float32, tc.a), make([]float32, tc.b), make([]float32, tc.c), 4, 4, 4)
		})
	}
}

func TestMatMulAddAcumula(t *testing.T) {
	a := []float32{1, 2, 3, 4}
	b := []float32{1, 0, 0, 1}

	c := []float32{10, 20, 30, 40}
	MatMulAdd(a, b, c, 2, 2, 2)

	want := []float32{11, 22, 33, 44}
	for i := range want {
		if !closeEnough(c[i], want[i]) {
			t.Fatalf("C[%d] = %v, quero %v", i, c[i], want[i])
		}
	}
}
