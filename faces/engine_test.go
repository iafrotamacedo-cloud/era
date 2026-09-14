package faces

import (
	"encoding/binary"
	"fmt"
	"image"
	"math"
	"os"
	"testing"

	"github.com/iafrotamacedo-cloud/era/faces/detect"
	"github.com/iafrotamacedo-cloud/era/faces/index"
)

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

func caminhosModelo() (yunet, sface string) {
	yunet = "../models/yunet.onnx"
	sface = "../models/sface.onnx"
	if p := os.Getenv("ERA_YUNET"); p != "" {
		yunet = p
	}
	if p := os.Getenv("ERA_SFACE"); p != "" {
		sface = p
	}
	return yunet, sface
}

func carregarEngine(t testing.TB) *Engine {
	t.Helper()

	yunet, sface := caminhosModelo()
	if _, err := os.Stat(yunet); err != nil {
		t.Skipf("YuNet nao encontrado em %s; baixe do OpenCV Zoo ou defina ERA_YUNET", yunet)
	}
	if _, err := os.Stat(sface); err != nil {
		t.Skipf("SFace nao encontrado em %s; baixe do OpenCV Zoo ou defina ERA_SFACE", sface)
	}

	eng, err := Open(Config{
		YuNet:  yunet,
		SFace:  sface,
		Detect: detect.Options{Score: limiarDoTeste, NMS: nmsDoTeste},
	})
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	return eng
}

func lerEmbeddingsReferencia(t testing.TB, dim int) [][]float32 {
	t.Helper()

	b, err := os.ReadFile("testdata/embeddings.bin")
	if err != nil {
		t.Skipf("referencia nao encontrada: %v (rode testdata/gerar_pipeline.py)", err)
	}
	if len(b)%(dim*4) != 0 {
		t.Fatalf("embeddings.bin tem %d bytes, que nao dividem em vetores de %d float32", len(b), dim)
	}

	n := len(b) / (dim * 4)
	out := make([][]float32, n)
	for i := 0; i < n; i++ {
		v := make([]float32, dim)
		for j := 0; j < dim; j++ {
			off := (i*dim + j) * 4
			v[j] = math.Float32frombits(binary.LittleEndian.Uint32(b[off:]))
		}
		out[i] = v
	}
	return out
}

func TestOpenRecusaArquivoInexistente(t *testing.T) {
	if _, err := Open(Config{YuNet: "/nao/existe/yunet.onnx", SFace: "/nao/existe/sface.onnx"}); err == nil {
		t.Error("arquivos inexistentes deveriam dar erro")
	}
}

func TestDimSFace(t *testing.T) {
	eng := carregarEngine(t)
	if eng.Dim() != 128 {
		t.Errorf("dim = %d, quero 128 para o SFace", eng.Dim())
	}
}

// TestPipelineBateComOpenCV fecha a Fase 7: o pipeline publico precisa bater
// com o FaceRecognizerSF do OpenCV na mesma imagem e nos mesmos rostos.
func TestPipelineBateComOpenCV(t *testing.T) {
	eng := carregarEngine(t)

	img := imagemDeTeste(640, 640)
	got, err := eng.Embed(img)
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}

	want := lerEmbeddingsReferencia(t, eng.Dim())
	if len(got) != len(want) {
		t.Fatalf("%d rostos, o OpenCV achou %d", len(got), len(want))
	}

	for i := range want {
		cos := Compare(got[i].Vector, want[i])
		// 0,9999: o warp inteiro diverge no maximo 1 px do cv::warpAffine;
		// o grafo SFace e sensivel, mas nao o bastante para mudar identidade.
		if cos < 0.9999 {
			t.Errorf("rosto %d: cosseno %.9f abaixo de 0,9999", i, cos)
		}
	}

	if !t.Failed() {
		t.Logf("%d vetores conferidos contra o OpenCV", len(got))
	}
}

func TestEmbedDeterministico(t *testing.T) {
	eng := carregarEngine(t)
	img := imagemDeTeste(640, 640)

	a, err := eng.Embed(img)
	if err != nil {
		t.Fatal(err)
	}
	b, err := eng.Embed(img)
	if err != nil {
		t.Fatal(err)
	}
	if len(a) != len(b) {
		t.Fatalf("contagem diverge: %d vs %d", len(a), len(b))
	}

	for i := range a {
		if Compare(a[i].Vector, b[i].Vector) != 1 {
			t.Errorf("rosto %d: segunda passagem divergiu", i)
		}
	}
}

func TestEmbedLargest(t *testing.T) {
	eng := carregarEngine(t)
	img := imagemDeTeste(640, 640)

	todos, err := eng.Embed(img)
	if err != nil {
		t.Fatal(err)
	}
	if len(todos) == 0 {
		t.Skip("imagem de teste nao tem rostos neste limiar")
	}

	maior, err := eng.EmbedLargest(img)
	if err != nil {
		t.Fatal(err)
	}

	if maior.Face.Score != todos[0].Face.Score {
		t.Errorf("score do maior = %v, quero %v", maior.Face.Score, todos[0].Face.Score)
	}
	if Compare(maior.Vector, todos[0].Vector) != 1 {
		t.Error("vetor do maior diverge do primeiro de Embed")
	}
}

func TestEmbedLargestSemRosto(t *testing.T) {
	eng := carregarEngine(t)

	// Imagem uniforme: o YuNet nao acha rosto com limiar alto.
	yunet, sface := caminhosModelo()
	engAlto, err := Open(Config{
		YuNet:  yunet,
		SFace:  sface,
		Detect: detect.Options{Score: 0.99, NMS: nmsDoTeste},
	})
	if err != nil {
		t.Fatal(err)
	}

	img := image.NewRGBA(image.Rect(0, 0, 64, 64))
	for i := 0; i < len(img.Pix); i += 4 {
		img.Pix[i+0] = 128
		img.Pix[i+1] = 128
		img.Pix[i+2] = 128
		img.Pix[i+3] = 255
	}

	if _, err := engAlto.EmbedLargest(img); err == nil {
		t.Error("deveria falhar sem rosto")
	}
	_ = eng
}

func TestRecognize(t *testing.T) {
	eng := carregarEngine(t)
	img := imagemDeTeste(640, 640)

	emb, err := eng.Embed(img)
	if err != nil {
		t.Fatal(err)
	}
	if len(emb) == 0 {
		t.Skip("sem rostos na imagem de teste")
	}

	ix, err := index.New(eng.Dim())
	if err != nil {
		t.Fatal(err)
	}
	if err := ix.Add("pessoa-a", emb[0].Vector); err != nil {
		t.Fatal(err)
	}

	ids, err := eng.Recognize(img, ix, index.SugestaoTipica)
	if err != nil {
		t.Fatal(err)
	}
	if len(ids) == 0 {
		t.Fatal("Recognize deveria devolver pelo menos um rosto")
	}
	if !ids[0].Matched || ids[0].Match.ID != "pessoa-a" {
		t.Errorf("match = %+v, matched=%v; quero pessoa-a com Matched true", ids[0].Match, ids[0].Matched)
	}
}

func TestRecognizeIndiceVazio(t *testing.T) {
	eng := carregarEngine(t)
	img := imagemDeTeste(640, 640)

	ix, err := index.New(eng.Dim())
	if err != nil {
		t.Fatal(err)
	}

	ids, err := eng.Recognize(img, ix, index.SugestaoTipica)
	if err != nil {
		t.Fatal(err)
	}
	for i, id := range ids {
		if id.Matched {
			t.Errorf("rosto %d: Matched deveria ser false com indice vazio", i)
		}
	}
}

func TestEngineConcorrente(t *testing.T) {
	eng := carregarEngine(t)
	img := imagemDeTeste(640, 640)

	esperado, err := eng.Embed(img)
	if err != nil {
		t.Fatal(err)
	}

	const goroutines = 4
	erros := make(chan error, goroutines)
	for i := 0; i < goroutines; i++ {
		go func() {
			got, err := eng.Embed(img)
			if err != nil {
				erros <- err
				return
			}
			if len(got) != len(esperado) {
				erros <- errf("achou %d rostos, esperava %d", len(got), len(esperado))
				return
			}
			for j := range got {
				if Compare(got[j].Vector, esperado[j].Vector) != 1 {
					erros <- errf("rosto %d diverge entre execucoes", j)
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
