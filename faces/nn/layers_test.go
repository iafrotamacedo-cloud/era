package nn

import (
	"math"
	"math/rand"
	"testing"

	"github.com/iafrotamacedo-cloud/era/faces/kernel"
	"github.com/iafrotamacedo-cloud/era/faces/tensor"
)

func TestConv2DContraKernelDeReferencia(t *testing.T) {
	casos := []struct {
		nome string
		n    int
		cfg  Conv2DConfig
		h, w int
	}{
		{"3x3 pad 1", 1, Conv2DConfig{InC: 3, OutC: 8, KH: 3, KW: 3, PadH: 1, PadW: 1}, 8, 8},
		{"lote de 4", 4, Conv2DConfig{InC: 3, OutC: 6, KH: 3, KW: 3, PadH: 1, PadW: 1}, 6, 6},
		{"1x1 pontual", 2, Conv2DConfig{InC: 8, OutC: 16, KH: 1, KW: 1}, 5, 5},
		{"stride 2", 1, Conv2DConfig{InC: 4, OutC: 8, KH: 3, KW: 3, PadH: 1, PadW: 1, StrideH: 2, StrideW: 2}, 9, 9},
		{"depthwise", 2, Conv2DConfig{InC: 8, OutC: 8, KH: 3, KW: 3, PadH: 1, PadW: 1, Groups: 8}, 7, 7},
		{"agrupada", 1, Conv2DConfig{InC: 8, OutC: 8, KH: 3, KW: 3, PadH: 1, PadW: 1, Groups: 2}, 6, 6},
		{"dilatada", 1, Conv2DConfig{InC: 2, OutC: 4, KH: 3, KW: 3, DilH: 2, DilW: 2}, 9, 9},
	}

	r := rand.New(rand.NewSource(4242))

	for _, c := range casos {
		t.Run(c.nome, func(t *testing.T) {
			grupos := max(c.cfg.Groups, 1)
			pesos := randSlice(r, c.cfg.OutC*(c.cfg.InC/grupos)*c.cfg.KH*c.cfg.KW)
			bias := randSlice(r, c.cfg.OutC)

			conv, err := NewConv2D("conv", c.cfg, pesos, bias)
			if err != nil {
				t.Fatalf("NewConv2D: %v", err)
			}

			entrada := randSlice(r, c.n*c.cfg.InC*c.h*c.w)
			x := tensor.MustFromSlice(entrada, c.n, c.cfg.InC, c.h, c.w)

			ws := NewWorkspace()
			got, err := conv.Forward(ws, x)
			if err != nil {
				t.Fatalf("Forward: %v", err)
			}

			// Referencia: uma imagem por vez, pelo kernel ingenuo.
			p := conv.params(c.h, c.w)
			porImagemIn := c.cfg.InC * c.h * c.w
			porImagemOut := c.cfg.OutC * p.OutH() * p.OutW()
			want := make([]float32, c.n*porImagemOut)

			for i := 0; i < c.n; i++ {
				kernel.Conv2DRef(
					entrada[i*porImagemIn:(i+1)*porImagemIn],
					pesos, bias, c.cfg.OutC, p,
					want[i*porImagemOut:(i+1)*porImagemOut],
				)
			}

			flat := got.Flat()
			if len(flat) != len(want) {
				t.Fatalf("saida tem %d elementos, quero %d", len(flat), len(want))
			}
			for i := range want {
				if !closeEnough(flat[i], want[i]) {
					t.Fatalf("saida[%d] = %v, referencia diz %v", i, flat[i], want[i])
				}
			}
		})
	}
}

func TestConv2DRecusaConfiguracaoInvalida(t *testing.T) {
	casos := []struct {
		nome  string
		cfg   Conv2DConfig
		pesos int
		bias  []float32
	}{
		{"canais zerados", Conv2DConfig{InC: 0, OutC: 4, KH: 3, KW: 3}, 100, nil},
		{"kernel zerado", Conv2DConfig{InC: 4, OutC: 4, KH: 0, KW: 3}, 100, nil},
		{"grupos nao dividem", Conv2DConfig{InC: 3, OutC: 4, KH: 3, KW: 3, Groups: 2}, 100, nil},
		{"pesos com tamanho errado", Conv2DConfig{InC: 3, OutC: 4, KH: 3, KW: 3}, 5, nil},
		{"bias com tamanho errado", Conv2DConfig{InC: 3, OutC: 4, KH: 3, KW: 3}, 108, make([]float32, 2)},
	}

	for _, c := range casos {
		t.Run(c.nome, func(t *testing.T) {
			_, err := NewConv2D("conv", c.cfg, make([]float32, c.pesos), c.bias)
			if err == nil {
				t.Error("NewConv2D deveria ter devolvido erro")
			}
		})
	}
}

func TestConv2DRecusaEntradaComFormaErrada(t *testing.T) {
	conv, err := NewConv2D("conv", Conv2DConfig{InC: 3, OutC: 4, KH: 3, KW: 3, PadH: 1, PadW: 1},
		make([]float32, 4*3*9), nil)
	if err != nil {
		t.Fatal(err)
	}

	ws := NewWorkspace()

	if _, err := conv.Forward(ws, tensor.New(1, 3, 8)); err == nil {
		t.Error("rank 3 deveria dar erro")
	}
	if _, err := conv.Forward(ws, tensor.New(1, 5, 8, 8)); err == nil {
		t.Error("numero de canais errado deveria dar erro")
	}
}

func TestBatchNormMatematica(t *testing.T) {
	// Um canal, valores escolhidos para dar conta redonda.
	//   gamma=2, beta=1, mean=5, var=4, eps=0
	//   escala = 2/sqrt(4) = 1
	//   desloc = 1 - 5*1 = -4
	//   y = x*1 - 4
	bn, err := NewBatchNorm("bn",
		[]float32{2}, []float32{1}, []float32{5}, []float32{4}, 0)
	if err != nil {
		t.Fatalf("NewBatchNorm: %v", err)
	}

	if !closeEnough(bn.Scale[0], 1) {
		t.Errorf("Scale = %v, quero 1", bn.Scale[0])
	}
	if !closeEnough(bn.Shift[0], -4) {
		t.Errorf("Shift = %v, quero -4", bn.Shift[0])
	}

	x := tensor.MustFromSlice([]float32{5, 7, 3}, 1, 1, 1, 3)
	ws := NewWorkspace()

	out, err := bn.Forward(ws, x)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	want := []float32{1, 3, -1}
	for i, w := range want {
		if !closeEnough(out.Flat()[i], w) {
			t.Errorf("saida[%d] = %v, quero %v", i, out.Flat()[i], w)
		}
	}
}

func TestBatchNormPorCanal(t *testing.T) {
	// Dois canais com parametros diferentes: prova que nao ha vazamento de
	// um canal para o outro.
	bn, err := NewBatchNormFromScaleShift("bn", []float32{2, 10}, []float32{0, 100})
	if err != nil {
		t.Fatal(err)
	}

	x := tensor.MustFromSlice([]float32{1, 2, 3, 4}, 1, 2, 1, 2)
	ws := NewWorkspace()

	out, err := bn.Forward(ws, x)
	if err != nil {
		t.Fatal(err)
	}

	want := []float32{2, 4, 130, 140}
	for i, w := range want {
		if !closeEnough(out.Flat()[i], w) {
			t.Errorf("saida[%d] = %v, quero %v", i, out.Flat()[i], w)
		}
	}
}

func TestBatchNormRecusaEntradaInvalida(t *testing.T) {
	if _, err := NewBatchNorm("bn", nil, nil, nil, nil, 0); err == nil {
		t.Error("vetores vazios deveriam dar erro")
	}
	if _, err := NewBatchNorm("bn", []float32{1}, []float32{1, 2}, []float32{1}, []float32{1}, 0); err == nil {
		t.Error("tamanhos diferentes deveriam dar erro")
	}
	// Variancia negativa somada a eps zero produziria raiz de numero
	// negativo -- NaN silencioso em toda a rede.
	if _, err := NewBatchNorm("bn", []float32{1}, []float32{0}, []float32{0}, []float32{-1}, 0); err == nil {
		t.Error("variancia+eps nao positiva deveria dar erro")
	}
}

func TestPReLU(t *testing.T) {
	p, err := NewPReLU("prelu", []float32{0.25, 0.5})
	if err != nil {
		t.Fatal(err)
	}

	x := tensor.MustFromSlice([]float32{2, -4, 2, -4}, 1, 2, 1, 2)
	ws := NewWorkspace()

	out, err := p.Forward(ws, x)
	if err != nil {
		t.Fatal(err)
	}

	// canal 0: positivo passa, negativo vira -4*0.25 = -1
	// canal 1: positivo passa, negativo vira -4*0.5  = -2
	want := []float32{2, -1, 2, -2}
	for i, w := range want {
		if !closeEnough(out.Flat()[i], w) {
			t.Errorf("saida[%d] = %v, quero %v", i, out.Flat()[i], w)
		}
	}
}

func TestPReLUAlfaCompartilhado(t *testing.T) {
	p, err := NewPReLU("prelu", []float32{0.1})
	if err != nil {
		t.Fatal(err)
	}

	x := tensor.MustFromSlice([]float32{1, -1, 2, -2}, 1, 2, 1, 2)
	ws := NewWorkspace()

	out, err := p.Forward(ws, x)
	if err != nil {
		t.Fatal(err)
	}

	want := []float32{1, -0.1, 2, -0.2}
	for i, w := range want {
		if !closeEnough(out.Flat()[i], w) {
			t.Errorf("saida[%d] = %v, quero %v", i, out.Flat()[i], w)
		}
	}
}

func TestPReLURecusaAlfaIncompativel(t *testing.T) {
	p, _ := NewPReLU("prelu", []float32{0.1, 0.2, 0.3})
	ws := NewWorkspace()
	if _, err := p.Forward(ws, tensor.New(1, 2, 4, 4)); err == nil {
		t.Error("3 alfas para 2 canais deveria dar erro")
	}
}

func TestReLU(t *testing.T) {
	r := &ReLU{}
	x := tensor.MustFromSlice([]float32{-1, 0, 1, 2}, 1, 4)
	ws := NewWorkspace()

	out, err := r.Forward(ws, x)
	if err != nil {
		t.Fatal(err)
	}

	want := []float32{0, 0, 1, 2}
	for i, w := range want {
		if out.Flat()[i] != w {
			t.Errorf("saida[%d] = %v, quero %v", i, out.Flat()[i], w)
		}
	}
}

func TestAdd(t *testing.T) {
	ws := NewWorkspace()
	a := tensor.MustFromSlice([]float32{1, 2, 3}, 1, 3)
	b := tensor.MustFromSlice([]float32{10, 20, 30}, 1, 3)

	out, err := Add(ws, a, b)
	if err != nil {
		t.Fatal(err)
	}

	want := []float32{11, 22, 33}
	for i, w := range want {
		if out.Flat()[i] != w {
			t.Errorf("saida[%d] = %v, quero %v", i, out.Flat()[i], w)
		}
	}

	if _, err := Add(ws, a, tensor.New(1, 4)); err == nil {
		t.Error("formas diferentes deveriam dar erro")
	}
}

func TestLinearContraCalculoDireto(t *testing.T) {
	for _, n := range []int{1, 2, 5} {
		t.Run(map[bool]string{true: "lote1", false: "lote"}[n == 1], func(t *testing.T) {
			const inF, outF = 7, 4

			r := rand.New(rand.NewSource(int64(n)))
			pesos := randSlice(r, outF*inF)
			bias := randSlice(r, outF)
			entrada := randSlice(r, n*inF)

			l, err := NewLinear("fc", inF, outF, pesos, bias)
			if err != nil {
				t.Fatal(err)
			}

			x := tensor.MustFromSlice(entrada, n, inF)
			ws := NewWorkspace()

			out, err := l.Forward(ws, x)
			if err != nil {
				t.Fatal(err)
			}

			for i := 0; i < n; i++ {
				for o := 0; o < outF; o++ {
					var soma float32
					for k := 0; k < inF; k++ {
						soma += pesos[o*inF+k] * entrada[i*inF+k]
					}
					soma += bias[o]

					got := out.Flat()[i*outF+o]
					if !closeEnough(got, soma) {
						t.Fatalf("lote %d, saida %d: %v, quero %v", i, o, got, soma)
					}
				}
			}
		})
	}
}

func TestLinearRecusaEntradaInvalida(t *testing.T) {
	l, _ := NewLinear("fc", 4, 2, make([]float32, 8), nil)
	ws := NewWorkspace()

	if _, err := l.Forward(ws, tensor.New(1, 3, 4, 4)); err == nil {
		t.Error("rank 4 deveria dar erro (falta Flatten)")
	}
	if _, err := l.Forward(ws, tensor.New(1, 9)); err == nil {
		t.Error("numero de atributos errado deveria dar erro")
	}
	if _, err := NewLinear("fc", 4, 2, make([]float32, 3), nil); err == nil {
		t.Error("pesos com tamanho errado deveriam dar erro")
	}
}

func TestFlatten(t *testing.T) {
	f := &Flatten{}
	x := tensor.MustFromSlice([]float32{1, 2, 3, 4, 5, 6, 7, 8}, 2, 2, 1, 2)
	ws := NewWorkspace()

	out, err := f.Forward(ws, x)
	if err != nil {
		t.Fatal(err)
	}

	if out.Shape[0] != 2 || out.Shape[1] != 4 {
		t.Errorf("forma = %v, quero [2 4]", out.Shape)
	}
	for i, v := range out.Flat() {
		if v != float32(i+1) {
			t.Errorf("saida[%d] = %v, quero %v", i, v, i+1)
		}
	}
}

func TestGlobalAvgPool(t *testing.T) {
	g := &GlobalAvgPool{}
	// canal 0: media de 1,2,3,4 = 2.5;  canal 1: media de 10,20,30,40 = 25
	x := tensor.MustFromSlice([]float32{1, 2, 3, 4, 10, 20, 30, 40}, 1, 2, 2, 2)
	ws := NewWorkspace()

	out, err := g.Forward(ws, x)
	if err != nil {
		t.Fatal(err)
	}

	if out.Shape[2] != 1 || out.Shape[3] != 1 {
		t.Errorf("forma = %v, quero [1 2 1 1]", out.Shape)
	}
	want := []float32{2.5, 25}
	for i, w := range want {
		if !closeEnough(out.Flat()[i], w) {
			t.Errorf("saida[%d] = %v, quero %v", i, out.Flat()[i], w)
		}
	}
}

func TestMaxPool2D(t *testing.T) {
	m, err := NewMaxPool2D("pool", 2, 2, 0, 0, 0, 0) // stride assume o tamanho da janela
	if err != nil {
		t.Fatal(err)
	}

	// 4x4 -> 2x2
	x := tensor.MustFromSlice([]float32{
		1, 2, 3, 4,
		5, 6, 7, 8,
		9, 10, 11, 12,
		13, 14, 15, 16,
	}, 1, 1, 4, 4)

	ws := NewWorkspace()
	out, err := m.Forward(ws, x)
	if err != nil {
		t.Fatal(err)
	}

	want := []float32{6, 8, 14, 16}
	for i, w := range want {
		if out.Flat()[i] != w {
			t.Errorf("saida[%d] = %v, quero %v", i, out.Flat()[i], w)
		}
	}
}

// TestMaxPoolPaddingNaoVenceComZero cobre uma sutileza real: se o padding
// valesse zero em vez de menos infinito, ele venceria qualquer ativacao
// negativa e falsearia a borda da imagem.
func TestMaxPoolPaddingNaoVenceComZero(t *testing.T) {
	m, err := NewMaxPool2D("pool", 3, 3, 1, 1, 1, 1)
	if err != nil {
		t.Fatal(err)
	}

	// Todos os valores negativos: o maximo tem que continuar negativo.
	x := tensor.MustFromSlice([]float32{-1, -2, -3, -4}, 1, 1, 2, 2)
	ws := NewWorkspace()

	out, err := m.Forward(ws, x)
	if err != nil {
		t.Fatal(err)
	}

	for i, v := range out.Flat() {
		if v >= 0 {
			t.Errorf("saida[%d] = %v; o padding venceu (deveria ser -inf, nao 0)", i, v)
		}
		if math.IsInf(float64(v), -1) {
			t.Errorf("saida[%d] = -Inf; nenhuma posicao real foi considerada", i)
		}
	}
}
