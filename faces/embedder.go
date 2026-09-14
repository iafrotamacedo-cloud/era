package faces

import (
	"fmt"
	"sync"

	"github.com/iafrotamacedo-cloud/era/faces/graph"
	"github.com/iafrotamacedo-cloud/era/faces/nn"
	"github.com/iafrotamacedo-cloud/era/faces/onnx"
	"github.com/iafrotamacedo-cloud/era/faces/tensor"
)

// embedder roda o modelo de reconhecimento (SFace ou equivalente).
type embedder struct {
	g       *graph.Graph
	entrada string
	saida   string
	dim     int
	pool    sync.Pool
}

func novoEmbedder(caminho string) (*embedder, error) {
	if caminho == "" {
		return nil, fmt.Errorf("faces: caminho do modelo de reconhecimento vazio")
	}

	m, err := onnx.Load(caminho)
	if err != nil {
		return nil, err
	}

	g, err := graph.New(m)
	if err != nil {
		return nil, err
	}

	if len(g.Inputs()) != 1 {
		return nil, fmt.Errorf("faces: o modelo de reconhecimento espera 1 entrada, tem %d", len(g.Inputs()))
	}
	if len(g.Outputs()) != 1 {
		return nil, fmt.Errorf("faces: o modelo de reconhecimento espera 1 saida, tem %d", len(g.Outputs()))
	}

	saida := g.Outputs()[0]
	dim, err := dimensaoSaida(m, saida)
	if err != nil {
		return nil, err
	}

	return &embedder{
		g:       g,
		entrada: g.Inputs()[0],
		saida:   saida,
		dim:     dim,
		pool:    sync.Pool{New: func() any { return nn.NewWorkspace() }},
	}, nil
}

func (e *embedder) Run(x *tensor.Tensor) ([]float32, error) {
	if x == nil {
		return nil, fmt.Errorf("faces: tensor de entrada nulo")
	}

	ws := e.pool.Get().(*nn.Workspace)
	defer func() {
		ws.Reset()
		e.pool.Put(ws)
	}()

	outs, err := e.g.Run(ws, map[string]*tensor.Tensor{e.entrada: x})
	if err != nil {
		return nil, err
	}

	t := outs[e.saida]
	if t == nil {
		return nil, fmt.Errorf("faces: o modelo nao produziu a saida %q", e.saida)
	}

	flat := t.Flat()
	if len(flat) != e.dim {
		return nil, fmt.Errorf("faces: vetor de %d dimensoes, o modelo declara %d", len(flat), e.dim)
	}

	// Copia: a memoria do workspace sera reaproveitada na proxima passagem.
	return append([]float32(nil), flat...), nil
}

func dimensaoSaida(m *onnx.Model, nome string) (int, error) {
	if m == nil || m.Graph == nil {
		return 0, fmt.Errorf("faces: modelo sem grafo")
	}
	for _, out := range m.Graph.Outputs {
		if out.Name != nome {
			continue
		}
		if len(out.Shape) == 0 {
			return 0, fmt.Errorf("faces: saida %q sem forma declarada", nome)
		}
		d := out.Shape[len(out.Shape)-1]
		if d.Simbolica() || d.Value <= 0 {
			return 0, fmt.Errorf("faces: saida %q com dimensao simbolica ou invalida", nome)
		}
		return int(d.Value), nil
	}
	return 0, fmt.Errorf("faces: saida %q nao encontrada no modelo", nome)
}
