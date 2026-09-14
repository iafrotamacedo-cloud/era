package nn

import (
	"math/rand"
	"testing"

	"github.com/iafrotamacedo-cloud/era/faces/kernel"
	"github.com/iafrotamacedo-cloud/era/faces/tensor"
)

func TestConvTranspose2DContraKernelDeReferencia(t *testing.T) {
	casos := []struct {
		nome string
		n    int
		cfg  ConvTranspose2DConfig
		h, w int
	}{
		{"kernel 2x2 stride 2, o caso real", 1, ConvTranspose2DConfig{InC: 4, OutC: 2, KH: 2, KW: 2, StrideH: 2, StrideW: 2}, 5, 5},
		{"lote de 3", 3, ConvTranspose2DConfig{InC: 4, OutC: 2, KH: 2, KW: 2, StrideH: 2, StrideW: 2}, 4, 4},
		{"stride 1 com padding", 1, ConvTranspose2DConfig{InC: 3, OutC: 6, KH: 3, KW: 3, PadH: 1, PadW: 1}, 6, 6},
		{"agrupada", 1, ConvTranspose2DConfig{InC: 4, OutC: 4, KH: 2, KW: 2, StrideH: 2, StrideW: 2, Groups: 2}, 4, 4},
		{"dilatada", 1, ConvTranspose2DConfig{InC: 2, OutC: 4, KH: 3, KW: 3, DilH: 2, DilW: 2}, 6, 6},
	}

	r := rand.New(rand.NewSource(1729))

	for _, c := range casos {
		t.Run(c.nome, func(t *testing.T) {
			grupos := max(c.cfg.Groups, 1)
			pesos := randSlice(r, c.cfg.InC*(c.cfg.OutC/grupos)*c.cfg.KH*c.cfg.KW)
			bias := randSlice(r, c.cfg.OutC)

			camada, err := NewConvTranspose2D("deconv", c.cfg, pesos, bias)
			if err != nil {
				t.Fatalf("NewConvTranspose2D: %v", err)
			}

			entrada := randSlice(r, c.n*c.cfg.InC*c.h*c.w)
			x := tensor.MustFromSlice(entrada, c.n, c.cfg.InC, c.h, c.w)

			ws := NewWorkspace()
			got, err := camada.Forward(ws, x)
			if err != nil {
				t.Fatalf("Forward: %v", err)
			}

			p := camada.params(c.h, c.w)
			porImagemIn := c.cfg.InC * c.h * c.w
			porImagemOut := c.cfg.OutC * p.OutH() * p.OutW()
			want := make([]float32, c.n*porImagemOut)

			for i := 0; i < c.n; i++ {
				kernel.ConvTranspose2DRef(
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

func TestConvTranspose2DRecusaConfiguracaoInvalida(t *testing.T) {
	casos := []struct {
		nome  string
		cfg   ConvTranspose2DConfig
		pesos int
		bias  int
	}{
		{"canal de entrada zero", ConvTranspose2DConfig{InC: 0, OutC: 4, KH: 2, KW: 2}, 0, 4},
		{"canal de saida zero", ConvTranspose2DConfig{InC: 4, OutC: 0, KH: 2, KW: 2}, 0, 0},
		{"kernel zero", ConvTranspose2DConfig{InC: 4, OutC: 4, KH: 0, KW: 2}, 0, 4},
		{"grupos nao dividem", ConvTranspose2DConfig{InC: 3, OutC: 4, KH: 2, KW: 2, Groups: 2}, 12, 4},
		{"pesos com tamanho errado", ConvTranspose2DConfig{InC: 4, OutC: 4, KH: 2, KW: 2}, 5, 4},
		{"bias com tamanho errado", ConvTranspose2DConfig{InC: 4, OutC: 4, KH: 2, KW: 2}, 64, 3},
	}
	for _, c := range casos {
		t.Run(c.nome, func(t *testing.T) {
			pesos := make([]float32, c.pesos)
			bias := make([]float32, c.bias)
			if _, err := NewConvTranspose2D("x", c.cfg, pesos, bias); err == nil {
				t.Error("deveria falhar")
			}
		})
	}
}

func TestConvTranspose2DRecusaEntradaComFormaErrada(t *testing.T) {
	cfg := ConvTranspose2DConfig{InC: 4, OutC: 2, KH: 2, KW: 2, StrideH: 2, StrideW: 2}
	camada, err := NewConvTranspose2D("deconv", cfg, make([]float32, 4*2*2*2), make([]float32, 2))
	if err != nil {
		t.Fatalf("NewConvTranspose2D: %v", err)
	}

	ws := NewWorkspace()

	// rank errado
	x3d := tensor.New(4, 5, 5)
	if _, err := camada.Forward(ws, x3d); err == nil {
		t.Error("rank 3 deveria falhar")
	}

	// canais errados
	xCanaisErrados := tensor.New(1, 3, 5, 5)
	if _, err := camada.Forward(ws, xCanaisErrados); err == nil {
		t.Error("canais errados deveriam falhar")
	}
}

func TestConvTranspose2DName(t *testing.T) {
	cfg := ConvTranspose2DConfig{InC: 2, OutC: 2, KH: 2, KW: 2}
	sem, _ := NewConvTranspose2D("", cfg, make([]float32, 2*2*2*2), nil)
	if sem.Name() != "ConvTranspose2D" {
		t.Errorf("Name() sem nome = %q, quero %q", sem.Name(), "ConvTranspose2D")
	}
	com, _ := NewConvTranspose2D("deconv1", cfg, make([]float32, 2*2*2*2), nil)
	if com.Name() != "deconv1" {
		t.Errorf("Name() com nome = %q, quero %q", com.Name(), "deconv1")
	}
}
