package faces

import (
	"fmt"
	"image"
	"os"

	"github.com/iafrotamacedo-cloud/era/faces/align"
	"github.com/iafrotamacedo-cloud/era/faces/detect"
	"github.com/iafrotamacedo-cloud/era/faces/index"
	"github.com/iafrotamacedo-cloud/era/faces/liveness"
	"github.com/iafrotamacedo-cloud/era/faces/spoof"
)

// Config carrega os dois modelos do motor.
type Config struct {
	// YuNet e o caminho do detector (.onnx). Vazio usa ERA_YUNET ou
	// "models/yunet.onnx".
	YuNet string

	// SFace e o caminho do reconhecedor (.onnx). Vazio usa ERA_SFACE ou
	// "models/sface.onnx".
	SFace string

	// Detect ajusta limiar, NMS e topK do YuNet. Zero vale os padroes do
	// pacote detect.
	Detect detect.Options

	// AntiSpoof habilita o classificador passivo e a verificacao de
	// movimento. Nil desliga anti-spoof.
	AntiSpoof *AntiSpoofConfig
}

// AntiSpoofConfig carrega o modelo MiniFAS e os limiares de liveness.
type AntiSpoofConfig struct {
	// Model e o caminho do .onnx. Vazio usa ERA_ANTISPOOF ou
	// "models/anti-spoof.onnx".
	Model string

	// Options ajusta recorte e limiar do classificador passivo.
	Options spoof.Options

	// Liveness ajusta a verificacao multi-frame (VerifyLive).
	Liveness liveness.Options
}

// Embedding e um rosto detectado com o vetor de identidade correspondente.
type Embedding struct {
	Face   detect.Face
	Vector []float32
}

// Identification junta deteccao, vetor e o melhor match no indice.
type Identification struct {
	Face    detect.Face
	Vector  []float32
	Match   index.Match
	Matched bool
}

// Engine e o motor pronto para detectar rostos e gerar vetores.
//
// Pode ser usado por varias goroutines ao mesmo tempo. O indice (pacote
// index) precisa de sincronizacao externa se Add ou Remove correrem junto
// com consultas.
type Engine struct {
	det      *detect.Detector
	rec      *embedder
	antispoof *spoof.Checker
	liveness liveness.Options
}

// Open monta o motor a partir dos caminhos em cfg.
func Open(cfg Config) (*Engine, error) {
	yunet := cfg.YuNet
	if yunet == "" {
		yunet = CaminhoYuNet()
	}
	sface := cfg.SFace
	if sface == "" {
		sface = CaminhoSFace()
	}

	det, err := detect.New(yunet, cfg.Detect)
	if err != nil {
		return nil, fmt.Errorf("faces: detector: %w", err)
	}

	rec, err := novoEmbedder(sface)
	if err != nil {
		return nil, fmt.Errorf("faces: reconhecedor: %w", err)
	}

	var chk *spoof.Checker
	var liveOpts liveness.Options
	if cfg.AntiSpoof != nil {
		caminho := cfg.AntiSpoof.Model
		if caminho == "" {
			caminho = CaminhoAntiSpoof()
		}
		chk, err = spoof.New(caminho, cfg.AntiSpoof.Options)
		if err != nil {
			return nil, fmt.Errorf("faces: anti-spoof: %w", err)
		}
		liveOpts = cfg.AntiSpoof.Liveness
	}

	return &Engine{det: det, rec: rec, antispoof: chk, liveness: liveOpts}, nil
}

// CaminhoYuNet devolve ERA_YUNET ou o padrao "models/yunet.onnx".
func CaminhoYuNet() string {
	if p := os.Getenv("ERA_YUNET"); p != "" {
		return p
	}
	return "models/yunet.onnx"
}

// CaminhoSFace devolve ERA_SFACE ou o padrao "models/sface.onnx".
func CaminhoSFace() string {
	if p := os.Getenv("ERA_SFACE"); p != "" {
		return p
	}
	return "models/sface.onnx"
}

// CaminhoAntiSpoof devolve ERA_ANTISPOOF ou o padrao "models/anti-spoof.onnx".
func CaminhoAntiSpoof() string {
	return spoof.CaminhoPadrao()
}

// HasAntiSpoof informa se o motor carregou o classificador passivo.
func (e *Engine) HasAntiSpoof() bool { return e.antispoof != nil }

// Dim devolve o tamanho do vetor de identidade (128 para o SFace).
func (e *Engine) Dim() int { return e.rec.dim }

// Detect acha rostos sem gerar vetores. Util para desenhar caixas na tela
// antes de decidir se vale a pena reconhecer.
func (e *Engine) Detect(img image.Image) ([]detect.Face, error) {
	return e.det.Detect(img)
}

// Embed detecta todos os rostos e gera um vetor para cada um.
//
// A ordem segue a confianca da deteccao, da maior para a menor — a mesma
// ordem que o detector devolve depois da supressao de nao-maximos.
func (e *Engine) Embed(img image.Image) ([]Embedding, error) {
	faces, err := e.det.Detect(img)
	if err != nil {
		return nil, err
	}

	out := make([]Embedding, 0, len(faces))

	for _, f := range faces {
		x, err := align.CropSFace(img, f.Points)
		if err != nil {
			return nil, fmt.Errorf("faces: alinhando rosto: %w", err)
		}

		vec, err := e.rec.Run(x)
		if err != nil {
			return nil, err
		}

		out = append(out, Embedding{Face: f, Vector: vec})
	}

	return out, nil
}

// EmbedLargest detecta rostos e devolve o vetor do rosto mais confiante.
func (e *Engine) EmbedLargest(img image.Image) (Embedding, error) {
	faces, err := e.det.Detect(img)
	if err != nil {
		return Embedding{}, err
	}
	if len(faces) == 0 {
		return Embedding{}, fmt.Errorf("faces: nenhum rosto encontrado")
	}

	f := faces[0]
	x, err := align.CropSFace(img, f.Points)
	if err != nil {
		return Embedding{}, fmt.Errorf("faces: alinhando rosto: %w", err)
	}

	vec, err := e.rec.Run(x)
	if err != nil {
		return Embedding{}, err
	}

	return Embedding{Face: f, Vector: vec}, nil
}

// CheckLive classifica um rosto como real ou spoof (modelo passivo).
func (e *Engine) CheckLive(img image.Image, face detect.Face) (spoof.Result, error) {
	if e.antispoof == nil {
		return spoof.Result{}, fmt.Errorf("faces: anti-spoof nao configurado")
	}
	return e.antispoof.Check(img, face)
}

// EmbedLive detecta o rosto principal, exige que passe no anti-spoof e gera o vetor.
func (e *Engine) EmbedLive(img image.Image) (Embedding, error) {
	emb, err := e.EmbedLargest(img)
	if err != nil {
		return Embedding{}, err
	}
	if e.antispoof == nil {
		return emb, nil
	}
	r, err := e.antispoof.Check(img, emb.Face)
	if err != nil {
		return Embedding{}, err
	}
	if !r.Live {
		return Embedding{}, fmt.Errorf("faces: rosto classificado como spoof")
	}
	return emb, nil
}

// VerifyLive exige movimento entre frames e anti-spoof no ultimo frame antes de embedar.
func (e *Engine) VerifyLive(frames []image.Image) (Embedding, error) {
	if e.antispoof == nil {
		return Embedding{}, fmt.Errorf("faces: anti-spoof nao configurado")
	}
	if len(frames) == 0 {
		return Embedding{}, fmt.Errorf("faces: nenhum frame")
	}

	seq := make([]align.Landmarks, 0, len(frames))
	var ultimo detect.Face
	for _, img := range frames {
		faces, err := e.det.Detect(img)
		if err != nil {
			return Embedding{}, err
		}
		if len(faces) == 0 {
			return Embedding{}, fmt.Errorf("faces: rosto nao encontrado num frame")
		}
		seq = append(seq, faces[0].Points)
		ultimo = faces[0]
	}

	mot, err := liveness.VerifySequence(seq, e.liveness)
	if err != nil {
		return Embedding{}, err
	}
	if !mot.Live {
		return Embedding{}, fmt.Errorf("faces: sem movimento suficiente (var=%v)", mot.Variance)
	}

	img := frames[len(frames)-1]
	r, err := e.antispoof.Check(img, ultimo)
	if err != nil {
		return Embedding{}, err
	}
	if !r.Live {
		return Embedding{}, fmt.Errorf("faces: rosto classificado como spoof")
	}

	x, err := align.CropSFace(img, ultimo.Points)
	if err != nil {
		return Embedding{}, fmt.Errorf("faces: alinhando rosto: %w", err)
	}
	vec, err := e.rec.Run(x)
	if err != nil {
		return Embedding{}, err
	}
	return Embedding{Face: ultimo, Vector: vec}, nil
}

// Compare devolve a similaridade de cosseno entre dois vetores.
func Compare(a, b []float32) float32 {
	return index.Cosine(a, b)
}

// Recognize detecta rostos, gera vetores e busca cada um no indice.
//
// Matched fica true quando ha match e Match.Score >= threshold. O limiar e
// decisao de quem chama — veja index.SugestaoTipica como ponto de partida.
func (e *Engine) Recognize(img image.Image, ix *index.Index, threshold float32) ([]Identification, error) {
	if ix == nil {
		return nil, fmt.Errorf("faces: indice nulo")
	}

	emb, err := e.Embed(img)
	if err != nil {
		return nil, err
	}

	out := make([]Identification, 0, len(emb))
	for _, r := range emb {
		m, ok, err := ix.Search(r.Vector)
		if err != nil {
			return nil, err
		}
		id := Identification{Face: r.Face, Vector: r.Vector}
		if ok {
			id.Match = m
			id.Matched = m.Score >= threshold
		}
		out = append(out, id)
	}

	return out, nil
}
