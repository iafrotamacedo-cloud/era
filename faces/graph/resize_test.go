package graph

import (
	"reflect"
	"strings"
	"testing"

	"github.com/iafrotamacedo-cloud/era/faces/nn"
	"github.com/iafrotamacedo-cloud/era/faces/onnx"
	"github.com/iafrotamacedo-cloud/era/faces/tensor"
)

// grafoResize monta um Resize com os atributos e as escalas dados.
func grafoResize(escalas []float32, tamanhos []float32, attrs ...*onnx.Attribute) *onnx.Model {
	entradas := []string{"x", "roi", "escalas"}
	inits := []*onnx.Tensor{
		peso("roi", []int64{0}, nil),
		peso("escalas", []int64{int64(len(escalas))}, escalas),
	}
	if tamanhos != nil {
		entradas = append(entradas, "tamanhos")
		inits = append(inits, peso("tamanhos", []int64{int64(len(tamanhos))}, tamanhos))
	}

	return modelo(&onnx.Graph{
		Name:         "resize",
		Nodes:        []*onnx.Node{no("Resize", "r", entradas, []string{"y"}, attrs...)},
		Initializers: inits,
		Inputs:       []*onnx.ValueInfo{{Name: "x"}},
		Outputs:      []*onnx.ValueInfo{{Name: "y"}},
	})
}

func rodarResize(t *testing.T, m *onnx.Model, x *tensor.Tensor) *tensor.Tensor {
	t.Helper()
	g, err := New(m)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	outs, err := g.Run(nn.NewWorkspace(), map[string]*tensor.Tensor{"x": x})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	return outs["y"]
}

// TestResizeUpsampleDoYuNet cobre exatamente a configuracao que o YuNet usa
// no caminho de cima para baixo da piramide: vizinho mais proximo, escala 2,
// modo assimetrico com piso.
//
// Com esses parametros cada pixel simplesmente vira um bloco 2x2.
func TestResizeUpsampleDoYuNet(t *testing.T) {
	m := grafoResize([]float32{1, 1, 2, 2}, nil,
		aString("mode", "nearest"),
		aString("coordinate_transformation_mode", "asymmetric"),
		aString("nearest_mode", "floor"))

	x := tensor.MustFromSlice([]float32{
		1, 2,
		3, 4,
	}, 1, 1, 2, 2)

	out := rodarResize(t, m, x)

	if want := []int{1, 1, 4, 4}; !reflect.DeepEqual(out.Shape, want) {
		t.Fatalf("forma = %v, quero %v", out.Shape, want)
	}
	want := []float32{
		1, 1, 2, 2,
		1, 1, 2, 2,
		3, 3, 4, 4,
		3, 3, 4, 4,
	}
	if got := out.Flat(); !reflect.DeepEqual(got, want) {
		t.Errorf("saida =\n%v\nquero\n%v", got, want)
	}
}

// TestResizeModosDeArredondamento: os modos "prefer" so diferem no empate
// exato -- e com escala inteira o empate acontece em TODOS os pixels, o que
// desloca a imagem inteira em um pixel se a escolha for errada.
func TestResizeModosDeArredondamento(t *testing.T) {
	// Reduzir 4 para 2 com half_pixel poe as coordenadas em 0.5 e 2.5,
	// que sao empates exatos.
	x := tensor.MustFromSlice([]float32{10, 20, 30, 40}, 1, 1, 1, 4)

	casos := []struct {
		modo  string
		quero []float32
	}{
		{"round_prefer_floor", []float32{10, 30}}, // 0.5 -> 0,  2.5 -> 2
		{"round_prefer_ceil", []float32{20, 40}},  // 0.5 -> 1,  2.5 -> 3
		{"floor", []float32{10, 30}},
		{"ceil", []float32{20, 40}},
	}

	for _, c := range casos {
		t.Run(c.modo, func(t *testing.T) {
			m := grafoResize([]float32{1, 1, 1, 0.5}, nil,
				aString("mode", "nearest"),
				aString("coordinate_transformation_mode", "half_pixel"),
				aString("nearest_mode", c.modo))

			out := rodarResize(t, m, x)
			if got := out.Flat(); !reflect.DeepEqual(got, c.quero) {
				t.Errorf("saida = %v, quero %v", got, c.quero)
			}
		})
	}
}

// TestResizeTransformacaoDeCoordenada compara as convencoes que as
// bibliotecas escolheram ao longo dos anos. Usar a errada desloca a imagem
// por uma fracao de pixel.
func TestResizeTransformacaoDeCoordenada(t *testing.T) {
	// 1x1x1x2 ampliado para 4, bilinear.
	x := tensor.MustFromSlice([]float32{0, 10}, 1, 1, 1, 2)

	casos := []struct {
		modo  string
		quero []float32
	}{
		// asymmetric: x_ent = x_sai/2 -> 0, 0.5, 1, 1.5 (limitado)
		{"asymmetric", []float32{0, 5, 10, 10}},
		// align_corners: primeiro e ultimo coincidem -> 0, 1/3, 2/3, 1
		{"align_corners", []float32{0, 10.0 / 3, 20.0 / 3, 10}},
		// half_pixel: (x+0.5)/2 - 0.5 -> -0.25, 0.25, 0.75, 1.25 (limitado)
		{"half_pixel", []float32{0, 2.5, 7.5, 10}},
	}

	for _, c := range casos {
		t.Run(c.modo, func(t *testing.T) {
			m := grafoResize([]float32{1, 1, 1, 2}, nil,
				aString("mode", "linear"),
				aString("coordinate_transformation_mode", c.modo))

			out := rodarResize(t, m, x)
			got := out.Flat()
			for i := range c.quero {
				if !closeEnough(got[i], c.quero[i]) {
					t.Errorf("saida = %v, quero %v", got, c.quero)
					break
				}
			}
		})
	}
}

func TestResizeBilinearAmplia(t *testing.T) {
	m := grafoResize([]float32{1, 1, 2, 2}, nil,
		aString("mode", "linear"),
		aString("coordinate_transformation_mode", "align_corners"))

	x := tensor.MustFromSlice([]float32{
		0, 10,
		20, 30,
	}, 1, 1, 2, 2)

	out := rodarResize(t, m, x)
	if want := []int{1, 1, 4, 4}; !reflect.DeepEqual(out.Shape, want) {
		t.Fatalf("forma = %v, quero %v", out.Shape, want)
	}

	got := out.Flat()
	// Com align_corners os cantos sao preservados exatamente.
	cantos := map[int]float32{0: 0, 3: 10, 12: 20, 15: 30}
	for pos, quero := range cantos {
		if !closeEnough(got[pos], quero) {
			t.Errorf("canto %d = %v, quero %v", pos, got[pos], quero)
		}
	}
	// E os valores intermediarios ficam entre os vizinhos.
	if !(got[1] > got[0] && got[1] < got[3]) {
		t.Errorf("interpolacao nao e monotona: %v", got[:4])
	}
}

func TestResizeReduz(t *testing.T) {
	m := grafoResize([]float32{1, 1, 0.5, 0.5}, nil,
		aString("mode", "nearest"),
		aString("coordinate_transformation_mode", "asymmetric"),
		aString("nearest_mode", "floor"))

	x := tensor.MustFromSlice([]float32{
		1, 2, 3, 4,
		5, 6, 7, 8,
		9, 10, 11, 12,
		13, 14, 15, 16,
	}, 1, 1, 4, 4)

	out := rodarResize(t, m, x)
	if want := []int{1, 1, 2, 2}; !reflect.DeepEqual(out.Shape, want) {
		t.Fatalf("forma = %v, quero %v", out.Shape, want)
	}
	// asymmetric + floor: pega os indices 0 e 2 nos dois eixos.
	if want := []float32{1, 3, 9, 11}; !reflect.DeepEqual(out.Flat(), want) {
		t.Errorf("saida = %v, quero %v", out.Flat(), want)
	}
}

func TestResizePorTamanhos(t *testing.T) {
	// Com a entrada 3 preenchida, os tamanhos mandam e as escalas sao
	// ignoradas.
	m := grafoResize(nil, []float32{1, 1, 3, 3},
		aString("mode", "nearest"),
		aString("coordinate_transformation_mode", "asymmetric"),
		aString("nearest_mode", "floor"))

	x := tensor.MustFromSlice([]float32{1, 2, 3, 4}, 1, 1, 2, 2)

	out := rodarResize(t, m, x)
	if want := []int{1, 1, 3, 3}; !reflect.DeepEqual(out.Shape, want) {
		t.Errorf("forma = %v, quero %v", out.Shape, want)
	}
}

func TestResizeVariosCanaisELote(t *testing.T) {
	m := grafoResize([]float32{1, 1, 2, 2}, nil,
		aString("mode", "nearest"),
		aString("coordinate_transformation_mode", "asymmetric"),
		aString("nearest_mode", "floor"))

	// 2 imagens, 2 canais, 1x1 cada: cada plano tem um valor so.
	x := tensor.MustFromSlice([]float32{1, 2, 3, 4}, 2, 2, 1, 1)

	out := rodarResize(t, m, x)
	if want := []int{2, 2, 2, 2}; !reflect.DeepEqual(out.Shape, want) {
		t.Fatalf("forma = %v, quero %v", out.Shape, want)
	}

	// Cada plano vira um bloco 2x2 do proprio valor, sem vazar para o
	// vizinho.
	got := out.Flat()
	for plano := 0; plano < 4; plano++ {
		quero := float32(plano + 1)
		for i := 0; i < 4; i++ {
			if got[plano*4+i] != quero {
				t.Fatalf("plano %d, posicao %d = %v, quero %v", plano, i, got[plano*4+i], quero)
			}
		}
	}
}

func TestResizeRecusaOQueNaoSuporta(t *testing.T) {
	casos := []struct {
		nome    string
		escalas []float32
		attrs   []*onnx.Attribute
		trecho  string
	}{
		{"cubic", []float32{1, 1, 2, 2},
			[]*onnx.Attribute{aString("mode", "cubic")}, "cubic"},
		{"modo desconhecido", []float32{1, 1, 2, 2},
			[]*onnx.Attribute{aString("mode", "inventado")}, "desconhecido"},
		{"tf_crop_and_resize", []float32{1, 1, 2, 2},
			[]*onnx.Attribute{aString("coordinate_transformation_mode", "tf_crop_and_resize")}, "roi"},
		{"nearest_mode desconhecido", []float32{1, 1, 2, 2},
			[]*onnx.Attribute{aString("nearest_mode", "inventado")}, "nearest_mode"},
		{"antialias", []float32{1, 1, 2, 2},
			[]*onnx.Attribute{aInt("antialias", 1)}, "antialias"},
		{"exclude_outside", []float32{1, 1, 2, 2},
			[]*onnx.Attribute{aInt("exclude_outside", 1)}, "exclude_outside"},
		{"sem escalas nem tamanhos", nil, nil, "escalas ou tamanhos"},
	}

	for _, c := range casos {
		t.Run(c.nome, func(t *testing.T) {
			_, err := New(grafoResize(c.escalas, nil, c.attrs...))
			if err == nil {
				t.Fatal("deveria dar erro")
			}
			if !strings.Contains(err.Error(), c.trecho) {
				t.Errorf("a mensagem deveria conter %q: %v", c.trecho, err)
			}
		})
	}
}

// TestResizeRecusaMexerEmLoteOuCanais: escalar a dimensao de canais mudaria
// o significado do tensor, e nenhuma rede de visao faz isso. Aproximar
// silenciosamente seria pior que recusar.
func TestResizeRecusaMexerEmLoteOuCanais(t *testing.T) {
	m := grafoResize([]float32{1, 2, 2, 2}, nil,
		aString("mode", "nearest"),
		aString("coordinate_transformation_mode", "asymmetric"))

	g, err := New(m)
	if err != nil {
		t.Fatal(err)
	}
	_, err = g.Run(nn.NewWorkspace(), map[string]*tensor.Tensor{
		"x": tensor.New(1, 2, 2, 2),
	})
	if err == nil {
		t.Error("escalar os canais deveria dar erro")
	}
}

func TestResizeRecusaRankErrado(t *testing.T) {
	m := grafoResize([]float32{1, 1, 2, 2}, nil,
		aString("mode", "nearest"),
		aString("coordinate_transformation_mode", "asymmetric"))

	g, err := New(m)
	if err != nil {
		t.Fatal(err)
	}
	_, err = g.Run(nn.NewWorkspace(), map[string]*tensor.Tensor{
		"x": tensor.New(1, 4),
	})
	if err == nil {
		t.Error("entrada de rank 2 deveria dar erro")
	}
}
