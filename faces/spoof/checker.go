package spoof

import (
	"fmt"
	"image"
	"math"
	"os"
	"sync"

	"github.com/iafrotamacedo-cloud/era/faces/detect"
	"github.com/iafrotamacedo-cloud/era/faces/graph"
	"github.com/iafrotamacedo-cloud/era/faces/nn"
	"github.com/iafrotamacedo-cloud/era/faces/onnx"
	"github.com/iafrotamacedo-cloud/era/faces/tensor"
)

// Options configura o classificador passivo anti-spoof.
type Options struct {
	// Expansao da caixa ao recortar o rosto. Zero vale 1.5 (facenox).
	Expansao float64

	// Tamanho da entrada do modelo. Zero vale 128.
	Tamanho int

	// Limiar de probabilidade para classificar como real. Zero vale 0.5.
	// Internamente convertido para limiar de logit (real - spoof).
	Limiar float64
}

// Result e a saida do classificador.
type Result struct {
	Live       bool
	LogitDiff  float32
	RealLogit  float32
	SpoofLogit float32
}

// Checker roda o modelo MiniFAS (facenox) sobre um recorte de rosto.
type Checker struct {
	g       *graph.Graph
	entrada string
	opts    Options
	pool    sync.Pool
}

// CaminhoPadrao devolve ERA_ANTISPOOF ou "models/anti-spoof.onnx".
func CaminhoPadrao() string {
	if p := os.Getenv("ERA_ANTISPOOF"); p != "" {
		return p
	}
	return "models/anti-spoof.onnx"
}

// New abre o modelo anti-spoof.
func New(caminho string, opts Options) (*Checker, error) {
	if caminho == "" {
		caminho = CaminhoPadrao()
	}
	if opts.Expansao == 0 {
		opts.Expansao = 1.5
	}
	if opts.Tamanho == 0 {
		opts.Tamanho = tamanhoPadrao
	}
	if opts.Limiar == 0 {
		opts.Limiar = 0.5
	}

	m, err := onnx.Load(caminho)
	if err != nil {
		return nil, err
	}
	g, err := graph.New(m)
	if err != nil {
		return nil, err
	}
	if len(g.Inputs()) != 1 || len(g.Outputs()) != 1 {
		return nil, fmt.Errorf("spoof: modelo espera 1 entrada e 1 saida, tem %d/%d",
			len(g.Inputs()), len(g.Outputs()))
	}

	return &Checker{
		g:       g,
		entrada: g.Inputs()[0],
		opts:    opts,
		pool:    sync.Pool{New: func() any { return nn.NewWorkspace() }},
	}, nil
}

// Check classifica um rosto ja detectado.
func (c *Checker) Check(img image.Image, face detect.Face) (Result, error) {
	recorte, err := crop(img, face, c.opts.Expansao)
	if err != nil {
		return Result{}, err
	}
	x, err := preprocess(recorte, c.opts.Tamanho)
	if err != nil {
		return Result{}, err
	}
	return c.infer(x)
}

// CheckTensor classifica uma entrada ja preprocessada [1,3,H,W].
func (c *Checker) CheckTensor(x *tensor.Tensor) (Result, error) {
	if x == nil {
		return Result{}, fmt.Errorf("spoof: tensor nulo")
	}
	return c.infer(x)
}

func (c *Checker) infer(x *tensor.Tensor) (Result, error) {
	ws := c.pool.Get().(*nn.Workspace)
	defer func() {
		ws.Reset()
		c.pool.Put(ws)
	}()

	outs, err := c.g.Run(ws, map[string]*tensor.Tensor{c.entrada: x})
	if err != nil {
		return Result{}, err
	}
	logits := outs[c.g.Outputs()[0]].Flat()
	if len(logits) < 2 {
		return Result{}, fmt.Errorf("spoof: saida tem %d logits, quero 2", len(logits))
	}

	real, spoof := logits[0], logits[1]
	diff := real - spoof
	p := math.Max(1e-6, math.Min(1-1e-6, c.opts.Limiar))
	limiar := float32(math.Log(p / (1 - p)))

	return Result{
		Live:       diff >= limiar,
		LogitDiff:  diff,
		RealLogit:  real,
		SpoofLogit: spoof,
	}, nil
}
