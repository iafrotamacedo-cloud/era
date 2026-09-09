package graph

import (
	"fmt"
	"math"

	"github.com/iafrotamacedo-cloud/era/faces/nn"
	"github.com/iafrotamacedo-cloud/era/faces/onnx"
	"github.com/iafrotamacedo-cloud/era/faces/tensor"
)

// Resize muda a resolucao de um tensor.
//
// Em redes de deteccao ele aparece no caminho de cima para baixo da piramide
// de caracteristicas: o mapa de resolucao menor e ampliado para se somar ao
// de resolucao maior. E como o detector consegue enxergar rostos grandes e
// pequenos ao mesmo tempo.
//
// # Por que o operador tem tantos parametros
//
// Redimensionar parece trivial ate a hora de decidir a que coordenada da
// ENTRADA corresponde cada pixel da SAIDA. Bibliotecas diferentes escolheram
// convencoes diferentes ao longo dos anos, e o ONNX teve que acomodar todas
// elas -- por isso coordinate_transformation_mode existe. Escolher a errada
// desloca a imagem por uma fracao de pixel, o que numa piramide de
// caracteristicas se traduz em caixas levemente fora do lugar.

// transformaCoordenada devolve a funcao que mapeia coordenada de saida em
// coordenada de entrada, conforme o modo pedido.
//
//	asymmetric          x_ent = x_sai / escala
//	half_pixel          x_ent = (x_sai + 0.5) / escala - 0.5
//	pytorch_half_pixel  como half_pixel, mas 0 quando a saida tem 1 pixel
//	align_corners       primeiro e ultimo pixels coincidem exatamente
func transformaCoordenada(modo string) (func(xSaida float64, escala float64, tamEnt, tamSai int) float64, error) {
	switch modo {
	case "", "half_pixel":
		return func(x, escala float64, _, _ int) float64 {
			return (x+0.5)/escala - 0.5
		}, nil

	case "asymmetric":
		return func(x, escala float64, _, _ int) float64 {
			return x / escala
		}, nil

	case "pytorch_half_pixel":
		return func(x, escala float64, _, tamSai int) float64 {
			if tamSai <= 1 {
				return 0
			}
			return (x+0.5)/escala - 0.5
		}, nil

	case "align_corners":
		return func(x, _ float64, tamEnt, tamSai int) float64 {
			if tamSai <= 1 {
				return 0
			}
			return x * float64(tamEnt-1) / float64(tamSai-1)
		}, nil

	case "tf_crop_and_resize":
		return nil, fmt.Errorf("coordinate_transformation_mode=%q depende do parametro roi e ainda nao e suportado", modo)
	}

	return nil, fmt.Errorf("coordinate_transformation_mode=%q desconhecido", modo)
}

// arredondaVizinho devolve a funcao que escolhe o pixel inteiro mais
// proximo, conforme nearest_mode.
//
// Os dois modos "prefer" so diferem no empate exato -- quando a coordenada
// cai bem no meio de dois pixels. Parece irrelevante e nao e: com escala
// inteira, o empate acontece em TODOS os pixels, e a escolha errada desloca
// a imagem inteira em um pixel.
func arredondaVizinho(modo string) (func(float64) int, error) {
	switch modo {
	case "", "round_prefer_floor":
		return func(v float64) int {
			return int(math.Ceil(v - 0.5))
		}, nil

	case "round_prefer_ceil":
		return func(v float64) int {
			return int(math.Floor(v + 0.5))
		}, nil

	case "floor":
		return func(v float64) int { return int(math.Floor(v)) }, nil

	case "ceil":
		return func(v float64) int { return int(math.Ceil(v)) }, nil
	}

	return nil, fmt.Errorf("nearest_mode=%q desconhecido", modo)
}

func montaResize(b *builder, n *onnx.Node) (*operation, error) {
	if len(n.Inputs) < 1 || len(n.Inputs) > 4 {
		return nil, fmt.Errorf("Resize espera de 1 a 4 entradas, recebeu %d", len(n.Inputs))
	}

	modo := n.AttrString("mode", "nearest")
	switch modo {
	case "nearest", "linear":
	case "cubic":
		return nil, fmt.Errorf("Resize com mode=cubic ainda nao e suportado")
	default:
		return nil, fmt.Errorf("Resize com mode=%q desconhecido", modo)
	}

	if n.AttrInt("antialias", 0) != 0 {
		return nil, fmt.Errorf("Resize com antialias=1 ainda nao e suportado")
	}
	if n.AttrInt("exclude_outside", 0) != 0 {
		return nil, fmt.Errorf("Resize com exclude_outside=1 ainda nao e suportado")
	}

	transforma, err := transformaCoordenada(n.AttrString("coordinate_transformation_mode", "half_pixel"))
	if err != nil {
		return nil, err
	}
	vizinho, err := arredondaVizinho(n.AttrString("nearest_mode", "round_prefer_floor"))
	if err != nil {
		return nil, err
	}

	// A entrada 1 e o roi, usado apenas por tf_crop_and_resize, que ja foi
	// recusado acima. Ela costuma vir como tensor vazio.

	var escalas []float32
	if t, err := b.pesoOpcional(n, 2, "escalas"); err != nil {
		return nil, err
	} else if t != nil && t.Size() > 0 {
		escalas = t.Flat()
	}

	var tamanhos []int
	if t, err := b.pesoOpcional(n, 3, "tamanhos"); err != nil {
		return nil, err
	} else if t != nil && t.Size() > 0 {
		for _, v := range t.Flat() {
			tamanhos = append(tamanhos, int(v))
		}
	}

	if len(escalas) == 0 && len(tamanhos) == 0 {
		return nil, fmt.Errorf("Resize precisa de escalas ou tamanhos constantes")
	}

	op := novaOp(n, func(ws *nn.Workspace, ins []*tensor.Tensor) ([]*tensor.Tensor, error) {
		out, err := redimensionar(ws, ins[0], escalas, tamanhos, modo, transforma, vizinho)
		if err != nil {
			return nil, err
		}
		return []*tensor.Tensor{out}, nil
	})
	// Roi, escalas e tamanhos ja foram lidos na montagem; em execucao so a
	// imagem interessa.
	op.inputs = n.Inputs[:1]
	return op, nil
}

func redimensionar(
	ws *nn.Workspace, x *tensor.Tensor,
	escalas []float32, tamanhos []int,
	modo string,
	transforma func(float64, float64, int, int) float64,
	vizinho func(float64) int,
) (*tensor.Tensor, error) {
	if x.Rank() != 4 {
		return nil, fmt.Errorf("Resize espera [N,C,H,W], recebeu %v", x.Shape)
	}

	n, c, h, w := x.Shape[0], x.Shape[1], x.Shape[2], x.Shape[3]

	// Resolve a geometria de saida, por escalas ou por tamanhos.
	var oh, ow int
	var escalaH, escalaW float64

	switch {
	case len(tamanhos) == 4:
		if tamanhos[0] != n || tamanhos[1] != c {
			return nil, fmt.Errorf("Resize so redimensiona altura e largura; tamanhos %v mudam lote ou canais", tamanhos)
		}
		oh, ow = tamanhos[2], tamanhos[3]
		escalaH = float64(oh) / float64(h)
		escalaW = float64(ow) / float64(w)

	case len(escalas) == 4:
		if escalas[0] != 1 || escalas[1] != 1 {
			return nil, fmt.Errorf("Resize so redimensiona altura e largura; escalas %v mexem em lote ou canais", escalas)
		}
		escalaH, escalaW = float64(escalas[2]), float64(escalas[3])
		// A especificacao manda truncar, nao arredondar.
		oh = int(math.Floor(float64(h) * escalaH))
		ow = int(math.Floor(float64(w) * escalaW))

	default:
		return nil, fmt.Errorf("Resize recebeu %d escalas e %d tamanhos; quero 4 de um dos dois",
			len(escalas), len(tamanhos))
	}

	if oh <= 0 || ow <= 0 {
		return nil, fmt.Errorf("Resize produz saida vazia %dx%d", oh, ow)
	}

	src := x.Flat()
	out := ws.Tensor(n, c, oh, ow)

	limita := func(v, max int) int {
		if v < 0 {
			return 0
		}
		if v >= max {
			return max - 1
		}
		return v
	}

	if modo == "nearest" {
		// As coordenadas de origem so dependem do indice de saida, nao do
		// plano. Calcular uma vez e reutilizar em todos os N*C planos evita
		// refazer a conta milhares de vezes.
		mapaY := make([]int, oh)
		for y := 0; y < oh; y++ {
			mapaY[y] = limita(vizinho(transforma(float64(y), escalaH, h, oh)), h)
		}
		mapaX := make([]int, ow)
		for xx := 0; xx < ow; xx++ {
			mapaX[xx] = limita(vizinho(transforma(float64(xx), escalaW, w, ow)), w)
		}

		for plano := 0; plano < n*c; plano++ {
			in := src[plano*h*w : (plano+1)*h*w]
			o := out.Data[plano*oh*ow : (plano+1)*oh*ow]

			for y := 0; y < oh; y++ {
				linha := in[mapaY[y]*w : mapaY[y]*w+w]
				destino := o[y*ow : (y+1)*ow]
				for xx := range destino {
					destino[xx] = linha[mapaX[xx]]
				}
			}
		}

		return out, nil
	}

	// Bilinear: os pesos tambem dependem so do indice de saida.
	type peso struct {
		i0, i1 int
		f      float32
	}
	pesosY := make([]peso, oh)
	for y := 0; y < oh; y++ {
		v := transforma(float64(y), escalaH, h, oh)
		i0 := int(math.Floor(v))
		pesosY[y] = peso{i0: limita(i0, h), i1: limita(i0+1, h), f: float32(v - float64(i0))}
	}
	pesosX := make([]peso, ow)
	for xx := 0; xx < ow; xx++ {
		v := transforma(float64(xx), escalaW, w, ow)
		i0 := int(math.Floor(v))
		pesosX[xx] = peso{i0: limita(i0, w), i1: limita(i0+1, w), f: float32(v - float64(i0))}
	}

	for plano := 0; plano < n*c; plano++ {
		in := src[plano*h*w : (plano+1)*h*w]
		o := out.Data[plano*oh*ow : (plano+1)*oh*ow]

		for y := 0; y < oh; y++ {
			py := pesosY[y]
			linha0 := in[py.i0*w : py.i0*w+w]
			linha1 := in[py.i1*w : py.i1*w+w]
			destino := o[y*ow : (y+1)*ow]

			for xx := range destino {
				px := pesosX[xx]

				cima := linha0[px.i0] + (linha0[px.i1]-linha0[px.i0])*px.f
				baixo := linha1[px.i0] + (linha1[px.i1]-linha1[px.i0])*px.f
				destino[xx] = cima + (baixo-cima)*py.f
			}
		}
	}

	return out, nil
}
