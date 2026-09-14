package spoof

import (
	"encoding/binary"
	"math"
	"os"
	"testing"

	"github.com/iafrotamacedo-cloud/era/faces/tensor"
)

func lerFloat32(t testing.TB, caminho string) []float32 {
	t.Helper()
	b, err := os.ReadFile(caminho)
	if err != nil {
		t.Fatalf("lendo %s: %v", caminho, err)
	}
	out := make([]float32, len(b)/4)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[i*4:]))
	}
	return out
}

func TestBateComORT(t *testing.T) {
	caminho := os.Getenv("ERA_ANTISPOOF")
	if caminho == "" {
		caminho = "../../models/anti-spoof.onnx"
	}
	if _, err := os.Stat(caminho); err != nil {
		t.Skipf("modelo nao encontrado em %s", caminho)
	}

	refPath := "testdata/logits.bin"
	if _, err := os.Stat(refPath); err != nil {
		t.Skip("testdata/logits.bin ausente; rode faces/spoof/testdata/gerar_referencia.py")
	}

	entrada := lerFloat32(t, "testdata/entrada.bin")
	ref := lerFloat32(t, refPath)

	x, err := tensor.FromSlice(entrada, 1, 3, 128, 128)
	if err != nil {
		t.Fatal(err)
	}

	chk, err := New(caminho, Options{})
	if err != nil {
		t.Fatal(err)
	}

	got, err := chk.CheckTensor(x)
	if err != nil {
		t.Fatal(err)
	}

	if !closeEnough(got.RealLogit, ref[0]) || !closeEnough(got.SpoofLogit, ref[1]) {
		t.Errorf("logits ERA [%v,%v] ORT [%v,%v]", got.RealLogit, got.SpoofLogit, ref[0], ref[1])
	}
}

func closeEnough(a, b float32) bool {
	d := a - b
	if d < 0 {
		d = -d
	}
	return d <= 1e-4+1e-4*abs32(b)
}

func abs32(v float32) float32 {
	if v < 0 {
		return -v
	}
	return v
}
