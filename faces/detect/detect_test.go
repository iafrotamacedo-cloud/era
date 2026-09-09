package detect

import (
	"encoding/binary"
	"fmt"
	"image"
	"math"
	"os"
	"testing"

	"github.com/iafrotamacedo-cloud/era/faces/align"
)

func perto(a, b, tol float64) bool { return math.Abs(a-b) <= tol }

// ---------- geometria, sem modelo ----------

func TestIoU(t *testing.T) {
	casos := []struct {
		nome  string
		a, b  Rect
		quero float64
	}{
		{"identicas", Rect{0, 0, 10, 10}, Rect{0, 0, 10, 10}, 1},
		{"disjuntas", Rect{0, 0, 10, 10}, Rect{20, 20, 10, 10}, 0},
		{"encostadas", Rect{0, 0, 10, 10}, Rect{10, 0, 10, 10}, 0},
		// interseccao 5x10=50, uniao 100+100-50=150
		{"metade", Rect{0, 0, 10, 10}, Rect{5, 0, 10, 10}, 50.0 / 150},
		// uma dentro da outra: 25 / 100
		{"contida", Rect{0, 0, 10, 10}, Rect{0, 0, 5, 5}, 25.0 / 100},
		{"area zero", Rect{0, 0, 0, 10}, Rect{0, 0, 10, 10}, 0},
		{"largura negativa", Rect{0, 0, -5, 10}, Rect{0, 0, 10, 10}, 0},
	}

	for _, c := range casos {
		t.Run(c.nome, func(t *testing.T) {
			if got := c.a.IoU(c.b); !perto(got, c.quero, 1e-12) {
				t.Errorf("IoU = %v, quero %v", got, c.quero)
			}
			// A medida e simetrica.
			if got := c.b.IoU(c.a); !perto(got, c.quero, 1e-12) {
				t.Errorf("IoU invertida = %v, quero %v", got, c.quero)
			}
		})
	}
}

func TestSuprimir(t *testing.T) {
	// Tres caixas quase iguais e uma longe: a supressao deve deixar duas.
	faces := []Face{
		{Box: Rect{0, 0, 10, 10}, Score: 0.9},
		{Box: Rect{1, 1, 10, 10}, Score: 0.8},     // sobrepoe muito a primeira
		{Box: Rect{2, 2, 10, 10}, Score: 0.7},     // idem
		{Box: Rect{100, 100, 10, 10}, Score: 0.6}, // longe
	}

	out := suprimir(faces, 0.3, 0)
	if len(out) != 2 {
		t.Fatalf("sobraram %d deteccoes, quero 2: %+v", len(out), out)
	}
	if out[0].Score != 0.9 || out[1].Score != 0.6 {
		t.Errorf("pontuacoes = %v, %v; quero 0.9 e 0.6", out[0].Score, out[1].Score)
	}
}

// TestSuprimirMantemAMaisConfiante: a ordem de entrada nao pode influir. Se
// a supressao guardasse a primeira em vez da melhor, o resultado dependeria
// da ordem em que as ancoras foram percorridas.
func TestSuprimirMantemAMaisConfiante(t *testing.T) {
	faces := []Face{
		{Box: Rect{0, 0, 10, 10}, Score: 0.2},
		{Box: Rect{1, 0, 10, 10}, Score: 0.95},
		{Box: Rect{2, 0, 10, 10}, Score: 0.5},
	}

	out := suprimir(faces, 0.3, 0)
	if len(out) != 1 {
		t.Fatalf("sobraram %d, quero 1", len(out))
	}
	if out[0].Score != 0.95 {
		t.Errorf("guardou a de pontuacao %v, quero a de 0.95", out[0].Score)
	}
}

func TestSuprimirTopK(t *testing.T) {
	var faces []Face
	for i := 0; i < 10; i++ {
		faces = append(faces, Face{
			Box:   Rect{float64(i * 100), 0, 10, 10}, // todas disjuntas
			Score: float32(i) / 10,
		})
	}

	out := suprimir(faces, 0.3, 3)
	if len(out) != 3 {
		t.Fatalf("sobraram %d, quero 3", len(out))
	}
	// As tres mais confiantes: 0.9, 0.8, 0.7.
	for i, quero := range []float32{0.9, 0.8, 0.7} {
		if out[i].Score != quero {
			t.Errorf("posicao %d = %v, quero %v", i, out[i].Score, quero)
		}
	}
}

func TestSuprimirVazio(t *testing.T) {
	if out := suprimir(nil, 0.3, 0); out != nil {
		t.Errorf("lista vazia deveria devolver nil, deu %v", out)
	}
}

func TestOptionsNormalizada(t *testing.T) {
	o := Options{}.normalizada()
	if o.Score != 0.6 || o.NMS != 0.3 || o.TopK != 5000 {
		t.Errorf("padroes = %+v", o)
	}

	o = Options{Score: 0.9, NMS: 0.5, TopK: 10}.normalizada()
	if o.Score != 0.9 || o.NMS != 0.5 || o.TopK != 10 {
		t.Errorf("valores explicitos foram sobrescritos: %+v", o)
	}
}

func TestEscalasDoModelo(t *testing.T) {
	saidas := []string{
		"cls_8", "cls_16", "cls_32",
		"obj_8", "obj_16", "obj_32",
		"bbox_8", "bbox_16", "bbox_32",
		"kps_8", "kps_16", "kps_32",
	}

	esc, err := escalasDoModelo(saidas, 640, 640)
	if err != nil {
		t.Fatalf("escalasDoModelo: %v", err)
	}
	if len(esc) != 3 {
		t.Fatalf("%d escalas, quero 3", len(esc))
	}

	// Ordenadas por passo, com a grade certa.
	quero := []struct{ passo, colunas, linhas int }{
		{8, 80, 80}, {16, 40, 40}, {32, 20, 20},
	}
	for i, q := range quero {
		if esc[i].passo != q.passo || esc[i].colunas != q.colunas || esc[i].linhas != q.linhas {
			t.Errorf("escala %d = passo %d grade %dx%d, quero passo %d grade %dx%d",
				i, esc[i].passo, esc[i].colunas, esc[i].linhas, q.passo, q.colunas, q.linhas)
		}
	}
}

func TestEscalasDoModeloRecusaIncompleto(t *testing.T) {
	casos := []struct {
		nome   string
		saidas []string
	}{
		{"sem cls", []string{"obj_8", "bbox_8", "kps_8"}},
		{"falta obj", []string{"cls_8", "bbox_8", "kps_8"}},
		{"falta kps", []string{"cls_8", "obj_8", "bbox_8"}},
		{"nada", nil},
	}

	for _, c := range casos {
		t.Run(c.nome, func(t *testing.T) {
			if _, err := escalasDoModelo(c.saidas, 640, 640); err == nil {
				t.Error("deveria dar erro")
			}
		})
	}
}

func TestEscalasDoModeloRecusaPassoGrandeDemais(t *testing.T) {
	saidas := []string{"cls_64", "obj_64", "bbox_64", "kps_64"}
	if _, err := escalasDoModelo(saidas, 32, 32); err == nil {
		t.Error("passo maior que a entrada deveria dar erro")
	}
}

// ---------- integracao com o modelo de verdade ----------

// A imagem de teste e gerada por formula inteira, identica a que o script
// testdata/gerar_referencia.py usa. O repositorio nao guarda imagem de
// rosto, e para conferir a DECODIFICACAO -- que e o que este teste mede --
// uma imagem sintetica serve igual.
//
// O array do OpenCV e BGR; image.Image em Go e RGB. Por isso os canais 0 e 2
// trocam de lugar na montagem.
const (
	limiarDoTeste = 0.05
	nmsDoTeste    = 0.3
)

func imagemDeTeste(larg, alt int) *image.NRGBA {
	img := image.NewNRGBA(image.Rect(0, 0, larg, alt))
	for y := 0; y < alt; y++ {
		for x := 0; x < larg; x++ {
			base := int64((y*larg + x) * 3)
			b := uint8((base * 7919) % 251)
			g := uint8(((base + 1) * 7919) % 251)
			r := uint8(((base + 2) * 7919) % 251)

			i := img.PixOffset(x, y)
			img.Pix[i+0] = r
			img.Pix[i+1] = g
			img.Pix[i+2] = b
			img.Pix[i+3] = 255
		}
	}
	return img
}

func caminhoModelo() string {
	if p := os.Getenv("ERA_YUNET"); p != "" {
		return p
	}
	return "../../models/yunet.onnx"
}

func carregar(t testing.TB, opt Options) *Detector {
	t.Helper()

	caminho := caminhoModelo()
	if _, err := os.Stat(caminho); err != nil {
		t.Skipf("modelo nao encontrado em %s; baixe o YuNet do OpenCV Zoo ou defina ERA_YUNET", caminho)
	}

	d, err := New(caminho, opt)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return d
}

func lerReferencia(t testing.TB) []Face {
	t.Helper()

	b, err := os.ReadFile("testdata/deteccoes.bin")
	if err != nil {
		t.Fatalf("lendo a referencia: %v", err)
	}
	if len(b)%(15*4) != 0 {
		t.Fatalf("a referencia tem %d bytes, que nao dividem em linhas de 15 float32", len(b))
	}

	n := len(b) / (15 * 4)
	out := make([]Face, n)
	for i := 0; i < n; i++ {
		var v [15]float32
		for j := 0; j < 15; j++ {
			v[j] = math.Float32frombits(binary.LittleEndian.Uint32(b[(i*15+j)*4:]))
		}
		out[i] = Face{
			Box:   Rect{X: float64(v[0]), Y: float64(v[1]), W: float64(v[2]), H: float64(v[3])},
			Score: v[14],
		}
		for j := 0; j < 5; j++ {
			out[i].Points[j] = align.Point{X: float64(v[4+2*j]), Y: float64(v[4+2*j+1])}
		}
	}
	return out
}

// TestDetectBateComOpenCV e o teste que fecha a Fase 5.
//
// A decodificacao das ancoras nao esta documentada de forma verificavel em
// lugar nenhum: as formulas deste pacote foram derivadas dos dados. Este
// teste e o que impede que elas se percam numa refatoracao.
func TestDetectBateComOpenCV(t *testing.T) {
	d := carregar(t, Options{Score: limiarDoTeste, NMS: nmsDoTeste})

	larg, alt := d.InputSize()
	if larg != 640 || alt != 640 {
		t.Fatalf("entrada %dx%d; a referencia foi gerada para 640x640", larg, alt)
	}

	got, err := d.Detect(imagemDeTeste(larg, alt))
	if err != nil {
		t.Fatalf("Detect: %v", err)
	}
	want := lerReferencia(t)

	if len(got) != len(want) {
		t.Fatalf("%d deteccoes, o OpenCV achou %d", len(got), len(want))
	}

	for i := range want {
		g, w := got[i], want[i]

		if !perto(float64(g.Score), float64(w.Score), 1e-4) {
			t.Errorf("deteccao %d: score %v, quero %v", i, g.Score, w.Score)
		}
		for _, c := range []struct {
			nome    string
			got, wa float64
		}{
			{"x", g.Box.X, w.Box.X}, {"y", g.Box.Y, w.Box.Y},
			{"w", g.Box.W, w.Box.W}, {"h", g.Box.H, w.Box.H},
		} {
			if !perto(c.got, c.wa, 1e-2) {
				t.Errorf("deteccao %d: %s = %.4f, quero %.4f", i, c.nome, c.got, c.wa)
			}
		}
		for j := range w.Points {
			if !perto(g.Points[j].X, w.Points[j].X, 1e-2) ||
				!perto(g.Points[j].Y, w.Points[j].Y, 1e-2) {
				t.Errorf("deteccao %d, ponto %d = (%.4f, %.4f), quero (%.4f, %.4f)",
					i, j, g.Points[j].X, g.Points[j].Y, w.Points[j].X, w.Points[j].Y)
			}
		}
	}

	if !t.Failed() {
		t.Logf("%d deteccoes conferidas contra o OpenCV", len(got))
	}
}

// TestDetectLetterbox: uma imagem que nao e do tamanho da entrada precisa
// voltar com as coordenadas no sistema dela, nao no do modelo.
func TestDetectLetterbox(t *testing.T) {
	d := carregar(t, Options{Score: limiarDoTeste, NMS: nmsDoTeste})
	larg, alt := d.InputSize()

	// Metade da resolucao: as coordenadas devem sair aproximadamente
	// metade das da imagem em tamanho natural.
	pequena := imagemDeTeste(larg/2, alt/2)
	faces, err := d.Detect(pequena)
	if err != nil {
		t.Fatalf("Detect: %v", err)
	}

	for i, f := range faces {
		// Nao ha rosto de verdade nesta imagem sintetica, entao o que se
		// confere e a coerencia das coordenadas, nao o conteudo.
		if f.Box.W <= 0 || f.Box.H <= 0 {
			t.Errorf("deteccao %d tem caixa degenerada: %+v", i, f.Box)
		}
		if f.Box.X > float64(larg/2) || f.Box.Y > float64(alt/2) {
			t.Errorf("deteccao %d comeca fora da imagem de %dx%d: %+v",
				i, larg/2, alt/2, f.Box)
		}
	}
}

func TestDetectRecusaEntradaInvalida(t *testing.T) {
	d := carregar(t, Options{})

	if _, err := d.Detect(nil); err == nil {
		t.Error("imagem nula deveria dar erro")
	}
	if _, err := d.Detect(image.NewNRGBA(image.Rect(0, 0, 0, 0))); err == nil {
		t.Error("imagem vazia deveria dar erro")
	}
}

func TestNewRecusaArquivoInexistente(t *testing.T) {
	if _, err := New("/nao/existe/modelo.onnx", Options{}); err == nil {
		t.Error("arquivo inexistente deveria dar erro")
	}
}

// TestDetectorConcorrente: o Detector se anuncia seguro para varias
// goroutines. A declaracao vale o que a verificacao provar -- e no CI este
// teste roda sob o detector de corrida.
func TestDetectorConcorrente(t *testing.T) {
	d := carregar(t, Options{Score: limiarDoTeste, NMS: nmsDoTeste})
	larg, alt := d.InputSize()
	img := imagemDeTeste(larg, alt)

	esperado, err := d.Detect(img)
	if err != nil {
		t.Fatal(err)
	}

	const goroutines = 4
	erros := make(chan error, goroutines)
	for i := 0; i < goroutines; i++ {
		go func() {
			faces, err := d.Detect(img)
			if err != nil {
				erros <- err
				return
			}
			if len(faces) != len(esperado) {
				erros <- errf("achou %d deteccoes, esperava %d", len(faces), len(esperado))
				return
			}
			for j := range faces {
				if faces[j].Score != esperado[j].Score || faces[j].Box != esperado[j].Box {
					erros <- errf("deteccao %d diverge entre execucoes", j)
					return
				}
			}
			erros <- nil
		}()
	}

	for i := 0; i < goroutines; i++ {
		if err := <-erros; err != nil {
			t.Error(err)
		}
	}
}

func errf(formato string, args ...any) error {
	return fmt.Errorf(formato, args...)
}

func BenchmarkDetect(b *testing.B) {
	d := carregar(b, Options{Score: limiarDoTeste, NMS: nmsDoTeste})
	larg, alt := d.InputSize()
	img := imagemDeTeste(larg, alt)

	if _, err := d.Detect(img); err != nil {
		b.Fatal(err)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := d.Detect(img); err != nil {
			b.Fatal(err)
		}
	}
}
