package nn

import (
	"math"
	"math/rand"
	"testing"

	"github.com/iafrotamacedo-cloud/era/tensor"
)

// redeDeTeste monta uma rede com a cara de um bloco de MobileFaceNet:
// convolucao comum, depthwise, normalizacoes, ativacoes e a camada final que
// produz o vetor.
func redeDeTeste(t testing.TB, seed int64) *Sequential {
	t.Helper()
	r := rand.New(rand.NewSource(seed))

	conv1, err := NewConv2D("conv1",
		Conv2DConfig{InC: 3, OutC: 8, KH: 3, KW: 3, PadH: 1, PadW: 1, StrideH: 2, StrideW: 2},
		randSlice(r, 8*3*9), nil)
	if err != nil {
		t.Fatal(err)
	}

	bn1, err := NewBatchNorm("bn1",
		randSlice(r, 8), randSlice(r, 8), randSlice(r, 8), positivos(r, 8), 1e-5)
	if err != nil {
		t.Fatal(err)
	}

	prelu1, err := NewPReLU("prelu1", randSlice(r, 8))
	if err != nil {
		t.Fatal(err)
	}

	dw, err := NewConv2D("dw",
		Conv2DConfig{InC: 8, OutC: 8, KH: 3, KW: 3, PadH: 1, PadW: 1, Groups: 8},
		randSlice(r, 8*9), nil)
	if err != nil {
		t.Fatal(err)
	}

	bn2, err := NewBatchNorm("bn2",
		randSlice(r, 8), randSlice(r, 8), randSlice(r, 8), positivos(r, 8), 1e-5)
	if err != nil {
		t.Fatal(err)
	}

	prelu2, err := NewPReLU("prelu2", randSlice(r, 8))
	if err != nil {
		t.Fatal(err)
	}

	fc, err := NewLinear("fc", 8, 16, randSlice(r, 16*8), randSlice(r, 16))
	if err != nil {
		t.Fatal(err)
	}

	return NewSequential("rede",
		conv1, bn1, prelu1,
		dw, bn2, prelu2,
		&GlobalAvgPool{}, &Flatten{}, fc,
	)
}

// positivos gera variancias plausiveis: sempre maiores que zero.
func positivos(r *rand.Rand, n int) []float32 {
	s := make([]float32, n)
	for i := range s {
		s[i] = r.Float32() + 0.1
	}
	return s
}

func TestSequentialForward(t *testing.T) {
	rede := redeDeTeste(t, 1)

	r := rand.New(rand.NewSource(99))
	x := tensor.MustFromSlice(randSlice(r, 1*3*16*16), 1, 3, 16, 16)

	ws := NewWorkspace()
	out, err := rede.Forward(ws, x)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	if out.Shape[0] != 1 || out.Shape[1] != 16 {
		t.Errorf("forma = %v, quero [1 16]", out.Shape)
	}
	for i, v := range out.Flat() {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("saida[%d] = %v", i, v)
		}
	}
}

func TestSequentialOutputShape(t *testing.T) {
	rede := redeDeTeste(t, 1)

	forma, err := rede.OutputShape([]int{1, 3, 16, 16})
	if err != nil {
		t.Fatalf("OutputShape: %v", err)
	}
	if len(forma) != 2 || forma[0] != 1 || forma[1] != 16 {
		t.Errorf("forma = %v, quero [1 16]", forma)
	}

	// A forma precisa ser calculavel sem rodar nada, e precisa recusar
	// entradas incompativeis.
	if _, err := rede.OutputShape([]int{1, 5, 16, 16}); err == nil {
		t.Error("numero de canais errado deveria dar erro")
	}
}

// TestFuseNaoMudaResultado e o teste que da valor a fusao de BatchNorm.
// Fundir precisa ser uma reescrita algebrica exata: o resultado tem que ser
// o mesmo, ou a otimizacao esta corrompendo a rede em silencio.
func TestFuseNaoMudaResultado(t *testing.T) {
	r := rand.New(rand.NewSource(7))
	entrada := randSlice(r, 1*3*16*16)
	x := tensor.MustFromSlice(entrada, 1, 3, 16, 16)

	semFusao := redeDeTeste(t, 1)
	comFusao := redeDeTeste(t, 1) // mesma seed: pesos identicos

	ws := NewWorkspace()
	antes, err := semFusao.Forward(ws, x)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	esperado := append([]float32(nil), antes.Flat()...)

	fundidas, err := comFusao.Fuse()
	if err != nil {
		t.Fatalf("Fuse: %v", err)
	}
	if fundidas != 2 {
		t.Errorf("fundiu %d camadas, quero 2", fundidas)
	}

	ws2 := NewWorkspace()
	depois, err := comFusao.Forward(ws2, x)
	if err != nil {
		t.Fatalf("Forward depois da fusao: %v", err)
	}

	got := depois.Flat()
	if len(got) != len(esperado) {
		t.Fatalf("saida tem %d elementos, quero %d", len(got), len(esperado))
	}
	for i := range esperado {
		if !closeEnough(got[i], esperado[i]) {
			t.Fatalf("saida[%d] = %v, sem fusao dava %v", i, got[i], esperado[i])
		}
	}
}

func TestFuseRemoveAsCamadas(t *testing.T) {
	rede := redeDeTeste(t, 1)
	antes := len(rede.Layers)

	fundidas, err := rede.Fuse()
	if err != nil {
		t.Fatal(err)
	}

	if len(rede.Layers) != antes-fundidas {
		t.Errorf("sobraram %d camadas, quero %d", len(rede.Layers), antes-fundidas)
	}
	for _, l := range rede.Layers {
		if _, ehBN := l.(*BatchNorm); ehBN {
			t.Errorf("camada %s deveria ter sido fundida", l.Name())
		}
	}
}

func TestFuseIgnoraQuandoNaoDa(t *testing.T) {
	// BatchNorm que nao vem logo depois de uma convolucao nao pode ser
	// fundida, e precisa continuar na rede.
	bn, err := NewBatchNormFromScaleShift("bn", []float32{2, 2}, []float32{1, 1})
	if err != nil {
		t.Fatal(err)
	}

	rede := NewSequential("rede", &ReLU{}, bn)
	fundidas, err := rede.Fuse()
	if err != nil {
		t.Fatal(err)
	}

	if fundidas != 0 {
		t.Errorf("fundiu %d camadas, quero 0", fundidas)
	}
	if len(rede.Layers) != 2 {
		t.Errorf("sobraram %d camadas, quero 2", len(rede.Layers))
	}
}

func TestFuseRecusaCanaisIncompativeis(t *testing.T) {
	conv, err := NewConv2D("conv", Conv2DConfig{InC: 3, OutC: 4, KH: 1, KW: 1},
		make([]float32, 12), nil)
	if err != nil {
		t.Fatal(err)
	}
	bn, err := NewBatchNormFromScaleShift("bn", []float32{1, 1}, []float32{0, 0})
	if err != nil {
		t.Fatal(err)
	}

	rede := NewSequential("rede", conv, bn)
	if _, err := rede.Fuse(); err == nil {
		t.Error("fundir BatchNorm de 2 canais numa conv de 4 saidas deveria dar erro")
	}
}

// TestWorkspaceSujoNaoAfetaResultado e um teste de higiene do pacote todo.
//
// O Workspace entrega memoria reaproveitada, com lixo de passagens
// anteriores. Toda camada precisa escrever em cada posicao da sua saida --
// se alguma so acumular, herda o lixo. Envenenar o workspace com NaN antes
// de rodar torna a falha impossivel de passar despercebida: NaN contamina
// qualquer conta que o toque.
func TestWorkspaceSujoNaoAfetaResultado(t *testing.T) {
	rede := redeDeTeste(t, 3)

	r := rand.New(rand.NewSource(555))
	x := tensor.MustFromSlice(randSlice(r, 1*3*16*16), 1, 3, 16, 16)

	limpo := NewWorkspace()
	ref, err := rede.Forward(limpo, x)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	esperado := append([]float32(nil), ref.Flat()...)

	// Enche o workspace de NaN e devolve tudo ao pool.
	sujo := NewWorkspace()
	veneno := sujo.Alloc(blocoMinimo * 4)
	nan := float32(math.NaN())
	for i := range veneno {
		veneno[i] = nan
	}
	sujo.Reset()

	out, err := rede.Forward(sujo, x)
	if err != nil {
		t.Fatalf("Forward com workspace sujo: %v", err)
	}

	for i, v := range out.Flat() {
		if math.IsNaN(float64(v)) {
			t.Fatalf("saida[%d] virou NaN: alguma camada leu memoria nao inicializada", i)
		}
		if !closeEnough(v, esperado[i]) {
			t.Fatalf("saida[%d] = %v com workspace sujo, %v com limpo", i, v, esperado[i])
		}
	}
}

// TestWorkspaceReutilizadoEntrePassagens confere que Reset deixa o workspace
// pronto para a proxima imagem sem alocar de novo.
func TestWorkspaceReutilizadoEntrePassagens(t *testing.T) {
	rede := redeDeTeste(t, 4)

	r := rand.New(rand.NewSource(11))
	x := tensor.MustFromSlice(randSlice(r, 1*3*16*16), 1, 3, 16, 16)

	ws := NewWorkspace()
	if _, err := rede.Forward(ws, x); err != nil {
		t.Fatal(err)
	}
	capDepoisDaPrimeira := ws.Cap()

	for i := 0; i < 10; i++ {
		ws.Reset()
		if _, err := rede.Forward(ws, x); err != nil {
			t.Fatal(err)
		}
	}

	if ws.Cap() != capDepoisDaPrimeira {
		t.Errorf("workspace cresceu de %d para %d em passagens repetidas",
			capDepoisDaPrimeira, ws.Cap())
	}
}

func TestSequentialPropagaErro(t *testing.T) {
	rede := redeDeTeste(t, 1)
	ws := NewWorkspace()

	// Numero de canais errado: a primeira camada tem que reclamar, e a
	// mensagem tem que dizer onde foi.
	_, err := rede.Forward(ws, tensor.New(1, 7, 16, 16))
	if err == nil {
		t.Fatal("entrada invalida deveria dar erro")
	}
	if got := err.Error(); got == "" {
		t.Error("mensagem de erro vazia")
	}
}

func BenchmarkRedeSemFusao(b *testing.B) {
	rede := redeDeTeste(b, 1)

	r := rand.New(rand.NewSource(1))
	x := tensor.MustFromSlice(randSlice(r, 1*3*112*112), 1, 3, 112, 112)
	ws := NewWorkspace()

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ws.Reset()
		if _, err := rede.Forward(ws, x); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkRedeComFusao(b *testing.B) {
	rede := redeDeTeste(b, 1)
	if _, err := rede.Fuse(); err != nil {
		b.Fatal(err)
	}

	r := rand.New(rand.NewSource(1))
	x := tensor.MustFromSlice(randSlice(r, 1*3*112*112), 1, 3, 112, 112)
	ws := NewWorkspace()

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ws.Reset()
		if _, err := rede.Forward(ws, x); err != nil {
			b.Fatal(err)
		}
	}
}
