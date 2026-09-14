package faces

import (
	"os"
	"testing"

	"github.com/iafrotamacedo-cloud/era/faces/tensor"
)

// TestEmbedderComRecorteOpenCV isola o SFace: se o cosseno bater, o problema
// esta no alinhamento; se nao, no grafo ou no pre-processamento.
func TestEmbedderComRecorteOpenCV(t *testing.T) {
	ref, err := os.ReadFile("testdata/opencv_align0.bin")
	if err != nil {
		t.Skip()
	}
	want := lerEmbeddingsReferencia(t, 128)
	if len(want) == 0 {
		t.Skip()
	}

	_, sface := caminhosModelo()
	rec, err := novoEmbedder(sface)
	if err != nil {
		t.Fatal(err)
	}

	plano := 112 * 112
	data := make([]float32, 3*plano)
	for y := 0; y < 112; y++ {
		for x := 0; x < 112; x++ {
			off := (y*112 + x) * 3
			pos := y*112 + x
			data[0*plano+pos] = float32(ref[off+2])
			data[1*plano+pos] = float32(ref[off+1])
			data[2*plano+pos] = float32(ref[off+0])
		}
	}

	x, err := tensor.FromSlice(data, 1, 3, 112, 112)
	if err != nil {
		t.Fatal(err)
	}

	vec, err := rec.Run(x)
	if err != nil {
		t.Fatal(err)
	}

	cos := Compare(vec, want[0])
	t.Logf("so embedder com recorte OpenCV: cosseno = %.9f", cos)
	if cos < 0.99999 {
		t.Errorf("embedder diverge do OpenCV com o mesmo recorte: %.9f", cos)
	}
}
