package nn

import (
	"math/rand"
	"testing"
)

func randSlice(r *rand.Rand, n int) []float32 {
	s := make([]float32, n)
	for i := range s {
		s[i] = r.Float32()*2 - 1
	}
	return s
}

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

func TestWorkspaceAlloc(t *testing.T) {
	ws := NewWorkspace()

	a := ws.Alloc(10)
	if len(a) != 10 {
		t.Fatalf("len = %d, quero 10", len(a))
	}

	b := ws.Alloc(20)
	if len(b) != 20 {
		t.Fatalf("len = %d, quero 20", len(b))
	}

	// Fatias distintas nao podem se sobrepor.
	for i := range a {
		a[i] = 1
	}
	for i := range b {
		b[i] = 2
	}
	for i, v := range a {
		if v != 1 {
			t.Fatalf("a[%d] = %v, quero 1 (as alocacoes se sobrepuseram)", i, v)
		}
	}

	if ws.Alloc(0) != nil {
		t.Error("Alloc(0) deveria devolver nil")
	}
}

func TestWorkspaceCapacidadeLimitada(t *testing.T) {
	// A fatia devolvida precisa vir com capacidade travada. Sem isso, um
	// append distraido de quem chama escreveria por cima da alocacao
	// seguinte -- e o bug apareceria longe da causa.
	ws := NewWorkspace()

	a := ws.Alloc(10)
	if cap(a) != 10 {
		t.Errorf("cap = %d, quero 10 (a fatia precisa ter capacidade travada)", cap(a))
	}
}

func TestWorkspaceReset(t *testing.T) {
	ws := NewWorkspace()

	a := ws.Alloc(100)
	capAntes := ws.Cap()

	ws.Reset()
	b := ws.Alloc(100)

	if ws.Cap() != capAntes {
		t.Errorf("Cap depois do Reset = %d, quero %d (deveria reaproveitar)", ws.Cap(), capAntes)
	}
	if &a[0] != &b[0] {
		t.Error("depois do Reset, a mesma alocacao deveria devolver a mesma memoria")
	}
}

// TestWorkspaceCrescerNaoInvalida cobre a razao de o workspace ser feito de
// blocos em vez de um buffer unico. Se ele crescesse realocando, as fatias
// entregues antes passariam a apontar para memoria abandonada -- e o dado
// sumiria em silencio.
func TestWorkspaceCrescerNaoInvalida(t *testing.T) {
	ws := NewWorkspace()

	primeira := ws.Alloc(10)
	for i := range primeira {
		primeira[i] = float32(i + 1)
	}

	// Forca varios blocos novos.
	for i := 0; i < 5; i++ {
		grande := ws.Alloc(blocoMinimo * 2)
		for j := range grande {
			grande[j] = -1
		}
	}

	for i, v := range primeira {
		if v != float32(i+1) {
			t.Fatalf("primeira[%d] = %v, quero %v (crescer corrompeu a fatia anterior)", i, v, i+1)
		}
	}
}

func TestWorkspaceAllocZeroed(t *testing.T) {
	ws := NewWorkspace()

	a := ws.Alloc(50)
	for i := range a {
		a[i] = 999
	}

	ws.Reset()
	b := ws.AllocZeroed(50)
	for i, v := range b {
		if v != 0 {
			t.Fatalf("AllocZeroed devolveu lixo em [%d]: %v", i, v)
		}
	}
}

func TestWorkspaceTensor(t *testing.T) {
	ws := NewWorkspace()

	x := ws.Tensor(2, 3, 4)
	if x.Size() != 24 {
		t.Errorf("Size = %d, quero 24", x.Size())
	}
	if len(x.Data) != 24 {
		t.Errorf("len(Data) = %d, quero 24", len(x.Data))
	}
}

func TestWorkspaceRelease(t *testing.T) {
	ws := NewWorkspace()
	ws.Alloc(1000)

	if ws.Cap() == 0 {
		t.Fatal("Cap deveria ser maior que zero depois de alocar")
	}

	ws.Release()
	if ws.Cap() != 0 {
		t.Errorf("Cap depois do Release = %d, quero 0", ws.Cap())
	}

	// Continua utilizavel.
	if len(ws.Alloc(10)) != 10 {
		t.Error("workspace deveria voltar a funcionar depois do Release")
	}
}

// BenchmarkWorkspaceVsAlocacao mede o motivo de o Workspace existir.
func BenchmarkWorkspaceReuso(b *testing.B) {
	ws := NewWorkspace()
	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		ws.Reset()
		for j := 0; j < 30; j++ { // ~30 camadas
			ws.Alloc(64 * 56 * 56)
		}
	}
}

func BenchmarkAlocacaoDireta(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		for j := 0; j < 30; j++ {
			buf := make([]float32, 64*56*56)
			_ = buf
		}
	}
}
