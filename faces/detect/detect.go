// Package detect acha rostos numa imagem.
//
// Envolve o detector YuNet: recebe uma imagem de tamanho qualquer e devolve,
// para cada rosto, a caixa delimitadora, os 5 pontos faciais e a confianca.
// Os 5 pontos sao exatamente os que o pacote align espera.
//
// # Como o modelo responde
//
// O YuNet nao devolve uma lista de rostos. Ele devolve doze tensores: quatro
// grandezas -- classificacao, presenca de objeto, caixa e pontos -- em tres
// escalas, com passos de 8, 16 e 32 pixels. Cada escala cobre a imagem com
// uma grade de ancoras: 80x80 no passo 8, 40x40 no passo 16, 20x20 no
// passo 32. Sao 8400 ancoras, e cada uma responde "se houvesse um rosto
// centrado perto de mim, ele estaria assim".
//
// Transformar isso em rostos exige tres etapas: decodificar as ancoras,
// filtrar por confianca e suprimir as vizinhas que descrevem o mesmo rosto.
//
// # De onde vem a formula
//
// A decodificacao nao esta documentada em lugar nenhum de forma
// verificavel. As formulas deste pacote foram derivadas comparando as saidas
// cruas do modelo com o que o cv2.FaceDetectorYN do OpenCV devolve na mesma
// imagem, e conferidas em cinco limiares diferentes: contagem identica e
// geometria batendo em 5e-04 de pixel.
package detect

import (
	"fmt"
	"image"
	"math"
	"sort"
	"strconv"
	"strings"
	"sync"

	"github.com/iafrotamacedo-cloud/era/faces/align"
	"github.com/iafrotamacedo-cloud/era/faces/graph"
	"github.com/iafrotamacedo-cloud/era/faces/nn"
	"github.com/iafrotamacedo-cloud/era/faces/onnx"
	"github.com/iafrotamacedo-cloud/era/faces/tensor"
)

// Rect e uma caixa delimitadora em coordenadas da imagem original.
type Rect struct {
	X, Y, W, H float64
}

// Area devolve a area da caixa.
func (r Rect) Area() float64 {
	if r.W <= 0 || r.H <= 0 {
		return 0
	}
	return r.W * r.H
}

// IoU e a razao entre a interseccao e a uniao de duas caixas.
//
// E a medida que decide se duas deteccoes descrevem o mesmo rosto: vale 1
// para caixas identicas e 0 para caixas que nao se tocam.
func (r Rect) IoU(o Rect) float64 {
	x1 := math.Max(r.X, o.X)
	y1 := math.Max(r.Y, o.Y)
	x2 := math.Min(r.X+r.W, o.X+o.W)
	y2 := math.Min(r.Y+r.H, o.Y+o.H)

	larg := x2 - x1
	alt := y2 - y1
	if larg <= 0 || alt <= 0 {
		return 0
	}

	inter := larg * alt
	uniao := r.Area() + o.Area() - inter
	if uniao <= 0 {
		return 0
	}
	return inter / uniao
}

// Face e um rosto detectado.
type Face struct {
	Box    Rect
	Points align.Landmarks
	Score  float32
}

// Options configura o detector.
type Options struct {
	// Score e a confianca minima para um rosto ser aceito. Zero vale 0.6.
	//
	// Mais alto perde rosto de perfil e rosto pequeno; mais baixo aceita
	// textura de parede como rosto. Calibre com imagens do seu uso real.
	Score float32

	// NMS e o limiar de sobreposicao acima do qual duas deteccoes sao
	// consideradas o mesmo rosto. Zero vale 0.3.
	NMS float64

	// TopK limita quantas deteccoes entram na supressao, das mais confiantes
	// para as menos. Zero vale 5000.
	TopK int
}

func (o Options) normalizada() Options {
	if o.Score == 0 {
		o.Score = 0.6
	}
	if o.NMS == 0 {
		o.NMS = 0.3
	}
	if o.TopK == 0 {
		o.TopK = 5000
	}
	return o
}

// escala descreve uma das tres resolucoes em que o modelo responde.
type escala struct {
	passo    int // 8, 16 ou 32
	colunas  int // largura da grade de ancoras
	linhas   int
	cls, obj string
	bbox     string
	kps      string
}

// Detector acha rostos.
//
// Pode ser usado por varias goroutines ao mesmo tempo: os pesos sao apenas
// lidos, e cada chamada pega um espaco de trabalho proprio de um pool.
type Detector struct {
	g         *graph.Graph
	entrada   string
	larg, alt int
	escalas   []escala
	opt       Options
	espacos   sync.Pool
}

// New carrega o detector de um arquivo .onnx.
func New(caminho string, opt Options) (*Detector, error) {
	m, err := onnx.Load(caminho)
	if err != nil {
		return nil, err
	}

	larg, alt, entrada, err := entradaDoModelo(m)
	if err != nil {
		return nil, err
	}

	g, err := graph.New(m)
	if err != nil {
		return nil, err
	}

	escalas, err := escalasDoModelo(g.Outputs(), larg, alt)
	if err != nil {
		return nil, err
	}

	return &Detector{
		g:       g,
		entrada: entrada,
		larg:    larg,
		alt:     alt,
		escalas: escalas,
		opt:     opt.normalizada(),
		espacos: sync.Pool{New: func() any { return nn.NewWorkspace() }},
	}, nil
}

// InputSize devolve a resolucao que o modelo espera.
//
// O YuNet de 2023 tem entrada FIXA. Imagens de outro tamanho passam por
// letterbox antes de entrar, e as coordenadas voltam convertidas.
func (d *Detector) InputSize() (larg, alt int) { return d.larg, d.alt }

// entradaDoModelo acha a entrada que nao e peso e le a resolucao dela.
func entradaDoModelo(m *onnx.Model) (larg, alt int, nome string, err error) {
	pesos := make(map[string]bool, len(m.Graph.Initializers))
	for _, t := range m.Graph.Initializers {
		pesos[t.Name] = true
	}

	var achada *onnx.ValueInfo
	for _, in := range m.Graph.Inputs {
		if pesos[in.Name] {
			continue // exportadores listam os pesos como entradas tambem
		}
		if achada != nil {
			return 0, 0, "", fmt.Errorf("detect: o modelo tem mais de uma entrada (%q e %q)", achada.Name, in.Name)
		}
		achada = in
	}
	if achada == nil {
		return 0, 0, "", fmt.Errorf("detect: o modelo nao declara nenhuma entrada")
	}

	if len(achada.Shape) != 4 {
		return 0, 0, "", fmt.Errorf("detect: a entrada %q tem forma de %d dimensoes, quero [N,C,H,W]",
			achada.Name, len(achada.Shape))
	}
	for i, d := range achada.Shape[2:] {
		if d.Simbolica() || d.Value <= 0 {
			return 0, 0, "", fmt.Errorf("detect: a entrada %q tem a dimensao %d simbolica (%v); "+
				"a ERA ainda so trata modelo de resolucao fixa", achada.Name, i+2, d)
		}
	}

	return int(achada.Shape[3].Value), int(achada.Shape[2].Value), achada.Name, nil
}

// escalasDoModelo descobre os passos pelos nomes das saidas.
//
// Ler os nomes -- em vez de fixar 8, 16 e 32 no codigo -- faz o pacote
// funcionar com variantes do YuNet que usem outro conjunto de escalas.
func escalasDoModelo(saidas []string, larg, alt int) ([]escala, error) {
	tem := make(map[string]bool, len(saidas))
	for _, s := range saidas {
		tem[s] = true
	}

	var passos []int
	for _, s := range saidas {
		if !strings.HasPrefix(s, "cls_") {
			continue
		}
		p, err := strconv.Atoi(strings.TrimPrefix(s, "cls_"))
		if err != nil || p <= 0 {
			continue
		}
		passos = append(passos, p)
	}
	if len(passos) == 0 {
		return nil, fmt.Errorf("detect: nenhuma saida com nome cls_<passo>; saidas: %v", saidas)
	}
	sort.Ints(passos)

	out := make([]escala, 0, len(passos))
	for _, p := range passos {
		e := escala{
			passo:   p,
			colunas: larg / p,
			linhas:  alt / p,
			cls:     fmt.Sprintf("cls_%d", p),
			obj:     fmt.Sprintf("obj_%d", p),
			bbox:    fmt.Sprintf("bbox_%d", p),
			kps:     fmt.Sprintf("kps_%d", p),
		}
		for _, nome := range []string{e.cls, e.obj, e.bbox, e.kps} {
			if !tem[nome] {
				return nil, fmt.Errorf("detect: falta a saida %q para o passo %d", nome, p)
			}
		}
		if e.colunas <= 0 || e.linhas <= 0 {
			return nil, fmt.Errorf("detect: entrada %dx%d nao comporta o passo %d", larg, alt, p)
		}
		out = append(out, e)
	}
	return out, nil
}

// Detect acha os rostos de uma imagem.
//
// As coordenadas devolvidas estao no sistema da imagem ORIGINAL, ja
// desfeito o redimensionamento interno.
func (d *Detector) Detect(img image.Image) ([]Face, error) {
	if img == nil {
		return nil, fmt.Errorf("detect: imagem nula")
	}
	b := img.Bounds()
	if b.Dx() <= 0 || b.Dy() <= 0 {
		return nil, fmt.Errorf("detect: imagem vazia (%v)", b)
	}

	// Letterbox: encolhe preservando a proporcao e preenche o resto.
	//
	// Esticar a imagem para o quadrado seria mais simples e degradaria a
	// deteccao: o modelo aprendeu com rostos de proporcao natural, e um
	// rosto achatado deixa de parecer rosto.
	fator := math.Min(float64(d.larg)/float64(b.Dx()), float64(d.alt)/float64(b.Dy()))
	t := align.Transform{A: fator, B: 0,
		Tx: -float64(b.Min.X) * fator,
		Ty: -float64(b.Min.Y) * fator}

	x, err := align.Warp(img, t, d.larg, d.alt, align.Options{
		Size:  d.larg,
		Scale: [3]float32{1, 1, 1},
		BGR:   true, // o YuNet vem do mundo OpenCV
		Borda: align.BordaConstante,
		Fill:  [3]float32{0, 0, 0},
	})
	if err != nil {
		return nil, err
	}

	faces, err := d.detectarTensor(x)
	if err != nil {
		return nil, err
	}

	// Volta para as coordenadas da imagem original.
	inv := 1 / fator
	desloc := align.Point{X: float64(b.Min.X), Y: float64(b.Min.Y)}
	for i := range faces {
		faces[i].Box.X = faces[i].Box.X*inv + desloc.X
		faces[i].Box.Y = faces[i].Box.Y*inv + desloc.Y
		faces[i].Box.W *= inv
		faces[i].Box.H *= inv
		for j := range faces[i].Points {
			faces[i].Points[j].X = faces[i].Points[j].X*inv + desloc.X
			faces[i].Points[j].Y = faces[i].Points[j].Y*inv + desloc.Y
		}
	}

	return faces, nil
}

// detectarTensor roda o modelo e decodifica, em coordenadas do modelo.
func (d *Detector) detectarTensor(x *tensor.Tensor) ([]Face, error) {
	ws := d.espacos.Get().(*nn.Workspace)
	defer func() {
		ws.Reset()
		d.espacos.Put(ws)
	}()

	outs, err := d.g.Run(ws, map[string]*tensor.Tensor{d.entrada: x})
	if err != nil {
		return nil, err
	}

	brutas, err := d.decodificar(outs)
	if err != nil {
		return nil, err
	}

	// Os tensores vivem no espaco de trabalho, que volta ao pool; decodificar
	// ja copiou tudo o que interessa para valores proprios.
	return suprimir(brutas, d.opt.NMS, d.opt.TopK), nil
}

// decodificar transforma as ancoras em rostos candidatos.
//
//	score = raiz(cls * obj)
//	cx = (coluna + bbox0) * passo      largura = exp(bbox2) * passo
//	cy = (linha  + bbox1) * passo      altura  = exp(bbox3) * passo
//	ponto_j = ((coluna + kps_2j) * passo, (linha + kps_2j+1) * passo)
//
// A caixa vem como centro e tamanho, e o tamanho em logaritmo. O logaritmo
// nao e enfeite: ele deixa a rede prever a RAZAO entre o tamanho do rosto e
// o da ancora, em vez da diferenca absoluta, e razao e o que se mantem
// estavel entre um rosto perto e um longe.
func (d *Detector) decodificar(outs map[string]*tensor.Tensor) ([]Face, error) {
	var faces []Face

	for _, e := range d.escalas {
		cls, err := plano(outs, e.cls, 1)
		if err != nil {
			return nil, err
		}
		obj, err := plano(outs, e.obj, 1)
		if err != nil {
			return nil, err
		}
		bbox, err := plano(outs, e.bbox, 4)
		if err != nil {
			return nil, err
		}
		kps, err := plano(outs, e.kps, 10)
		if err != nil {
			return nil, err
		}

		n := e.colunas * e.linhas
		if len(cls) < n || len(obj) < n || len(bbox) < n*4 || len(kps) < n*10 {
			return nil, fmt.Errorf("detect: passo %d espera %d ancoras, os tensores nao comportam", e.passo, n)
		}

		p := float64(e.passo)
		for i := 0; i < n; i++ {
			// Multiplicar antes da raiz evita calcular a raiz de 8400
			// ancoras das quais quase todas serao descartadas.
			produto := cls[i] * obj[i]
			if produto <= 0 || produto < d.opt.Score*d.opt.Score {
				continue
			}

			col := float64(i % e.colunas)
			lin := float64(i / e.colunas)

			b := bbox[i*4 : i*4+4]
			larg := math.Exp(float64(b[2])) * p
			alt := math.Exp(float64(b[3])) * p
			cx := (col + float64(b[0])) * p
			cy := (lin + float64(b[1])) * p

			f := Face{
				Box:   Rect{X: cx - larg/2, Y: cy - alt/2, W: larg, H: alt},
				Score: float32(math.Sqrt(float64(produto))),
			}

			k := kps[i*10 : i*10+10]
			for j := 0; j < 5; j++ {
				f.Points[j] = align.Point{
					X: (col + float64(k[2*j])) * p,
					Y: (lin + float64(k[2*j+1])) * p,
				}
			}

			faces = append(faces, f)
		}
	}

	return faces, nil
}

// plano devolve os valores de uma saida, conferindo que a ultima dimensao e
// a esperada.
func plano(outs map[string]*tensor.Tensor, nome string, porAncora int) ([]float32, error) {
	t, ok := outs[nome]
	if !ok {
		return nil, fmt.Errorf("detect: o modelo nao produziu a saida %q", nome)
	}
	if r := t.Rank(); r > 0 && t.Shape[r-1] != porAncora {
		return nil, fmt.Errorf("detect: a saida %q tem forma %v, esperava %d valores por ancora",
			nome, t.Shape, porAncora)
	}
	return t.Flat(), nil
}

// suprimir aplica supressao de nao-maximos.
//
// Um rosto ativa dezenas de ancoras vizinhas, e todas descrevem quase a
// mesma caixa. A supressao percorre da mais confiante para a menos, mantem a
// primeira de cada grupo e descarta as que se sobrepoem demais a ela.
func suprimir(faces []Face, limiar float64, topK int) []Face {
	if len(faces) == 0 {
		return nil
	}

	sort.SliceStable(faces, func(i, j int) bool {
		return faces[i].Score > faces[j].Score
	})
	if topK > 0 && len(faces) > topK {
		faces = faces[:topK]
	}

	morta := make([]bool, len(faces))
	out := make([]Face, 0, len(faces))

	for i := range faces {
		if morta[i] {
			continue
		}
		out = append(out, faces[i])

		for j := i + 1; j < len(faces); j++ {
			if !morta[j] && faces[i].Box.IoU(faces[j].Box) > limiar {
				morta[j] = true
			}
		}
	}

	return out
}
