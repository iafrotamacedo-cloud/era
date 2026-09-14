package align

import (
	"encoding/binary"
	"image"
	"math"
	"os"
	"testing"
)

func imagemSintetica640() *image.NRGBA {
	const larg, alt = 640, 640
	img := image.NewNRGBA(image.Rect(0, 0, larg, alt))
	for y := 0; y < alt; y++ {
		for x := 0; x < larg; x++ {
			base := int64((y*larg + x) * 3)
			b := uint8((base * 7919) % 251)
			g := uint8(((base + 1) * 7919) % 251)
			r := uint8(((base + 2) * 7919) % 251)
			i := img.PixOffset(x, y)
			img.Pix[i+0], img.Pix[i+1], img.Pix[i+2], img.Pix[i+3] = r, g, b, 255
		}
	}
	return img
}

func lerLandmarksReferencia(t testing.TB) Landmarks {
	t.Helper()
	b, err := os.ReadFile("../testdata/face0.bin")
	if err != nil {
		t.Skip("gere testdata/face0.bin com o script python")
	}
	var v [15]float32
	for j := 0; j < 15; j++ {
		v[j] = math.Float32frombits(binary.LittleEndian.Uint32(b[j*4:]))
	}
	var l Landmarks
	for j := 0; j < 5; j++ {
		l[j] = Point{X: float64(v[4+2*j]), Y: float64(v[4+2*j+1])}
	}
	return l
}

func TestParaOpenCVTransform(t *testing.T) {
	l := lerLandmarksReferencia(t)
	t1, err := Para(l, 112)
	if err != nil {
		t.Fatal(err)
	}
	t2, err := ParaOpenCV(l, 112)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("Para:       A=%.8f B=%.8f Tx=%.4f Ty=%.4f", t1.A, t1.B, t1.Tx, t1.Ty)
	t.Logf("ParaOpenCV: A=%.8f B=%.8f Tx=%.4f Ty=%.4f", t2.A, t2.B, t2.Tx, t2.Ty)
	// estimateAffinePartial2D no mesmo src/dst (ver script python)
	t.Logf("OpenCV est: A=%.8f B=%.8f Tx=%.4f Ty=%.4f", 0.26111495, 0.005336787, -121.98213, 30.18576)
}

func TestParaOpenCVBatePixelsComOpenCV(t *testing.T) {
	ref, err := os.ReadFile("../testdata/opencv_align0.bin")
	if err != nil {
		t.Skip("gere testdata/opencv_align0.bin com o script python")
	}

	img := imagemSintetica640()
	crop, err := CropImageOpenCV(img, lerLandmarksReferencia(t), 112)
	if err != nil {
		t.Fatal(err)
	}

	var maxDiff float64
	for y := 0; y < 112; y++ {
		for x := 0; x < 112; x++ {
			i := crop.PixOffset(x, y)
			r, g, b := crop.Pix[i], crop.Pix[i+1], crop.Pix[i+2]
			off := (y*112 + x) * 3
			// opencv_align0.bin vem do warpAffine em BGR; CropImageOpenCV e RGB.
			ob, og, or_ := ref[off], ref[off+1], ref[off+2]
			for _, d := range []float64{
				math.Abs(float64(r) - float64(or_)),
				math.Abs(float64(g) - float64(og)),
				math.Abs(float64(b) - float64(ob)),
			} {
				if d > maxDiff {
					maxDiff = d
				}
			}
		}
	}
	t.Logf("max diff pixels = %.2f", maxDiff)
	if maxDiff > 2 {
		t.Errorf("alinhamento diverge do OpenCV em %.2f px", maxDiff)
	}
}

// CropImageOpenCV devolve RGBA para conferencia visual.
func CropImageOpenCV(img image.Image, pontos Landmarks, tamanho int) (*image.RGBA, error) {
	t, err := ParaOpenCV(pontos, tamanho)
	if err != nil {
		return nil, err
	}
	inv, err := t.Inverse()
	if err != nil {
		return nil, err
	}
	am, err := novoAmostrador(img)
	if err != nil {
		return nil, err
	}
	am.borda = BordaConstante

	out := image.NewRGBA(image.Rect(0, 0, tamanho, tamanho))
	for y := 0; y < tamanho; y++ {
		for x := 0; x < tamanho; x++ {
			p := inv.Apply(Point{X: float64(x), Y: float64(y)})
			r, g, b := am.bilinear(p.X, p.Y)
			i := out.PixOffset(x, y)
			out.Pix[i+0] = clampByte(r)
			out.Pix[i+1] = clampByte(g)
			out.Pix[i+2] = clampByte(b)
			out.Pix[i+3] = 255
		}
	}
	return out, nil
}
