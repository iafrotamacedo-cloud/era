package align

import (
	"fmt"
	"image"
	"math"

	"github.com/iafrotamacedo-cloud/era/faces/tensor"
)

// mediaDstOpenCV e o centro fixo do gabarito no FaceRecognizerSF do OpenCV
// 4.x. Nao e a media aritmetica dos 5 pontos -- e uma constante do modelo.
var mediaDstOpenCV = Point{X: 56.0262, Y: 71.9008}

// ParaOpenCV calcula a transformacao de similaridade que o
// cv::FaceRecognizerSF::alignCrop usa: Umeyama com media de destino fixa e
// decomposicao em valores singulares em 2x2.
//
// O algoritmo fecha com o codigo em modules/objdetect/src/face_recognize.cpp
// do OpenCV 4.x. O Para() generico por numeros complexos e equivalente em
// teoria, mas diverge o bastante na pratica para degradar o SFace.
func ParaOpenCV(pontos Landmarks, tamanho int) (Transform, error) {
	if tamanho <= 0 {
		tamanho = 112
	}
	dst := Template(tamanho)
	k := float64(tamanho) / 112.0
	mediaDst := Point{X: mediaDstOpenCV.X * k, Y: mediaDstOpenCV.Y * k}

	var mediaSrc Point
	for i := range pontos {
		mediaSrc.X += pontos[i].X
		mediaSrc.Y += pontos[i].Y
	}
	mediaSrc.X /= 5
	mediaSrc.Y /= 5

	var srcD, dstD [5]Point
	for i := range pontos {
		srcD[i] = Point{X: pontos[i].X - mediaSrc.X, Y: pontos[i].Y - mediaSrc.Y}
		dstD[i] = Point{X: dst[i].X - mediaDst.X, Y: dst[i].Y - mediaDst.Y}
	}

	var a00, a01, a10, a11 float64
	for i := 0; i < 5; i++ {
		a00 += dstD[i].X * srcD[i].X
		a01 += dstD[i].X * srcD[i].Y
		a10 += dstD[i].Y * srcD[i].X
		a11 += dstD[i].Y * srcD[i].Y
	}
	a00 /= 5
	a01 /= 5
	a10 /= 5
	a11 /= 5

	u00, u01, u10, u11, s0, s1, vt00, vt01, vt10, vt11 := svd2x2(a00, a01, a10, a11)

	detA := a00*a11 - a01*a10
	d := [2]float64{1, 1}
	if detA < 0 {
		d[1] = -1
	}

	smax := s0
	if s1 > smax {
		smax = s1
	}
	tol := smax * 2 * float64(math.SmallestNonzeroFloat32)
	rank := 0
	if s0 > tol {
		rank++
	}
	if s1 > tol {
		rank++
	}

	detU := u00*u11 - u01*u10
	detVt := vt00*vt11 - vt01*vt10

	var t00, t01, t10, t11 float64

	switch rank {
	case 1:
		if detU*detVt > 0 {
			t00 = u00*vt00 + u01*vt10
			t01 = u00*vt01 + u01*vt11
			t10 = u10*vt00 + u11*vt10
			t11 = u10*vt01 + u11*vt11
		} else {
			dd := d[1]
			d[1] = -1
			t00 = u00*d[0]*vt00 + u01*d[1]*vt10
			t01 = u00*d[0]*vt01 + u01*d[1]*vt11
			t10 = u10*d[0]*vt00 + u11*d[1]*vt10
			t11 = u10*d[0]*vt01 + u11*d[1]*vt11
			d[1] = dd
		}
	default:
		t00 = u00*d[0]*vt00 + u01*d[1]*vt10
		t01 = u00*d[0]*vt01 + u01*d[1]*vt11
		t10 = u10*d[0]*vt00 + u11*d[1]*vt10
		t11 = u10*d[0]*vt01 + u11*d[1]*vt11
	}

	var var1, var2 float64
	for i := 0; i < 5; i++ {
		var1 += srcD[i].X * srcD[i].X
		var2 += srcD[i].Y * srcD[i].Y
	}
	var1 /= 5
	var2 /= 5

	scale := 1.0 / (var1 + var2) * (s0*d[0] + s1*d[1])
	ts0 := t00*mediaSrc.X + t01*mediaSrc.Y
	ts1 := t10*mediaSrc.X + t11*mediaSrc.Y

	t02 := mediaDst.X - scale*ts0
	t12 := mediaDst.Y - scale*ts1

	t00 *= scale
	t01 *= scale
	t10 *= scale
	t11 *= scale

	// OpenCV: x' = t00*x + t01*y + t02;  nosso: x' = A*x - B*y + Tx
	return Transform{
		A:  t00,
		B:  -t01,
		Tx: t02,
		Ty: t12,
	}, nil
}

// svd2x2 decompoe A em U * diag(s0,s1) * Vt.
func svd2x2(a00, a01, a10, a11 float64) (
	u00, u01, u10, u11 float64,
	s0, s1 float64,
	vt00, vt01, vt10, vt11 float64,
) {
	// A^T A e simetrica 2x2.
	ata00 := a00*a00 + a10*a10
	ata01 := a00*a01 + a10*a11
	ata11 := a01*a01 + a11*a11

	tr := ata00 + ata11
	det := ata00*ata11 - ata01*ata01
	disc := tr*tr/4 - det
	if disc < 0 {
		disc = 0
	}
	root := math.Sqrt(disc)
	l0 := tr/2 + root
	l1 := tr/2 - root

	s0 = math.Sqrt(math.Max(0, l0))
	s1 = math.Sqrt(math.Max(0, l1))

	// Vt: autovetores de A^T A.
	if math.Abs(ata01) < 1e-15 {
		vt00, vt01 = 1, 0
		vt10, vt11 = 0, 1
	} else {
		vt00 = ata01
		vt01 = l0 - ata00
		n := math.Hypot(vt00, vt01)
		if n > 0 {
			vt00 /= n
			vt01 /= n
		}
		vt10 = -vt01
		vt11 = vt00
	}

	// U = A * V * S^{-1}
	if s0 > 1e-15 {
		w00 := a00*vt00 + a01*vt01
		w10 := a10*vt00 + a11*vt01
		n := math.Hypot(w00, w10)
		if n > 0 {
			u00, u10 = w00/n, w10/n
		}
	}
	if s1 > 1e-15 {
		w01 := a00*vt10 + a01*vt11
		w11 := a10*vt10 + a11*vt11
		n := math.Hypot(w01, w11)
		if n > 0 {
			u01, u11 = w01/n, w11/n
		}
	}

	// Garante matrizes ortonormais com determinante coerente.
	if u00 == 0 && u10 == 0 {
		u00, u10 = 1, 0
	}
	if u01 == 0 && u11 == 0 {
		u01, u11 = 0, 1
	}

	return u00, u01, u10, u11, s0, s1, vt00, vt01, vt10, vt11
}

// CropSFace recorta como o FaceRecognizerSF do OpenCV: alinhamento Umeyama e
// tensor RGB 0-255 com borda preta fora da imagem.
func CropSFace(img image.Image, pontos Landmarks) (*tensor.Tensor, error) {
	t, err := ParaOpenCV(pontos, 112)
	if err != nil {
		return nil, err
	}
	return warpOpenCV(img, t, 112, 112, OpcoesSFace())
}

// warpOpenCV amostra como cv::warpAffine: para cada pixel INTEIRO (x, y) da
// saida, aplica a inversa da transformacao e interpola em bilinear. O Warp
// generico usa centro de pixel (+0.5), que diverge o bastante para degradar o
// SFace.
func warpOpenCV(img image.Image, t Transform, larg, alt int, opt Options) (*tensor.Tensor, error) {
	opt = opt.normalizada()
	if larg <= 0 || alt <= 0 {
		return nil, fmt.Errorf("align: tamanho de saida invalido (%dx%d)", larg, alt)
	}

	inv, err := t.Inverse()
	if err != nil {
		return nil, err
	}

	am, err := novoAmostrador(img)
	if err != nil {
		return nil, err
	}
	am.borda = opt.Borda
	am.fill = [3]float64{float64(opt.Fill[0]), float64(opt.Fill[1]), float64(opt.Fill[2])}

	out := tensor.New(1, 3, alt, larg)
	plano := alt * larg

	iR, iG, iB := 0, 1, 2
	if opt.BGR {
		iR, iB = 2, 0
	}

	for y := 0; y < alt; y++ {
		for x := 0; x < larg; x++ {
			p := inv.Apply(Point{X: float64(x), Y: float64(y)})
			r, g, b := am.bilinear(p.X, p.Y)
			pos := y*larg + x

			out.Data[iR*plano+pos] = (float32(r) - opt.Mean[0]) * opt.Scale[0]
			out.Data[iG*plano+pos] = (float32(g) - opt.Mean[1]) * opt.Scale[1]
			out.Data[iB*plano+pos] = (float32(b) - opt.Mean[2]) * opt.Scale[2]
		}
	}

	return out, nil
}
