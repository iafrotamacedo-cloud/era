package spoof

import (
	"fmt"
	"image"
	"image/color"
	"math"

	"github.com/iafrotamacedo-cloud/era/faces/detect"
	"github.com/iafrotamacedo-cloud/era/faces/tensor"
)

const tamanhoPadrao = 128

// crop recorta um rosto quadrado com expansao da caixa, como no facenox.
func crop(img image.Image, face detect.Face, expansao float64) (image.Image, error) {
	b := img.Bounds()
	larg, alt := b.Dx(), b.Dy()

	x := int(face.Box.X)
	y := int(face.Box.Y)
	w := int(face.Box.W)
	h := int(face.Box.H)
	if w <= 0 || h <= 0 {
		return nil, fmt.Errorf("spoof: caixa invalida %v", face.Box)
	}

	maxDim := math.Max(float64(w), float64(h))
	cx := float64(x) + float64(w)/2
	cy := float64(y) + float64(h)/2

	x0 := int(cx - maxDim*expansao/2)
	y0 := int(cy - maxDim*expansao/2)
	tam := int(maxDim * expansao)

	x1 := max(0, x0)
	y1 := max(0, y0)
	x2 := min(larg, x0+tam)
	y2 := min(alt, y0+tam)

	padT := max(0, -y0)
	padL := max(0, -x0)
	padB := max(0, (y0+tam) - alt)
	padR := max(0, (x0+tam) - larg)

	recorte := image.NewRGBA(image.Rect(0, 0, tam, tam))
	if x2 > x1 && y2 > y1 {
		dx := x1 - x0
		dy := y1 - y0
		for yy := y1; yy < y2; yy++ {
			for xx := x1; xx < x2; xx++ {
				recorte.Set(dx+(xx-x1), dy+(yy-y1), img.At(b.Min.X+xx, b.Min.Y+yy))
			}
		}
	}

	if padT > 0 || padL > 0 || padB > 0 || padR > 0 {
		recorte = refletirBorda(recorte, padT, padL, padB, padR)
	}

	if recorte.Bounds().Dx() != tam || recorte.Bounds().Dy() != tam {
		return nil, fmt.Errorf("spoof: recorte %dx%d, quero %dx%d",
			recorte.Bounds().Dx(), recorte.Bounds().Dy(), tam, tam)
	}
	return recorte, nil
}

func refletirBorda(img *image.RGBA, padT, padL, padB, padR int) *image.RGBA {
	w := img.Bounds().Dx()
	h := img.Bounds().Dy()
	out := image.NewRGBA(image.Rect(0, 0, w+padL+padR, h+padT+padB))

	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			out.Set(x+padL, y+padT, img.At(x, y))
		}
	}

	// Borda refletida (BORDER_REFLECT_101).
	for y := 0; y < out.Bounds().Dy(); y++ {
		sy := y - padT
		if sy < 0 {
			sy = -sy
		}
		if sy >= h {
			sy = 2*h - sy - 2
		}
		for x := 0; x < out.Bounds().Dx(); x++ {
			sx := x - padL
			if sx < 0 {
				sx = -sx
			}
			if sx >= w {
				sx = 2*w - sx - 2
			}
			if y < padT || y >= padT+h || x < padL || x >= padL+w {
				out.Set(x, y, img.At(sx, sy))
			}
		}
	}
	return out
}

// preprocess letterbox + normaliza para CHW [0,1], como facenox/preprocess.py.
func preprocess(img image.Image, tam int) (*tensor.Tensor, error) {
	b := img.Bounds()
	oh, ow := b.Dy(), b.Dx()
	if oh == 0 || ow == 0 {
		return nil, fmt.Errorf("spoof: imagem vazia")
	}

	ratio := float64(tam) / float64(max(oh, ow))
	nh := int(float64(oh) * ratio)
	nw := int(float64(ow) * ratio)
	if nh == 0 {
		nh = 1
	}
	if nw == 0 {
		nw = 1
	}

	redim := redimensionar(img, nw, nh)
	padT := (tam - nh) / 2
	padL := (tam - nw) / 2
	quad := image.NewRGBA(image.Rect(0, 0, tam, tam))
	for y := 0; y < tam; y++ {
		for x := 0; x < tam; x++ {
			sx, sy := x-padL, y-padT
			var c color.Color
			if sx < 0 || sy < 0 || sx >= nw || sy >= nh {
				c = refletirPixel(redim, sx, sy, nw, nh)
			} else {
				c = redim.At(sx, sy)
			}
			quad.Set(x, y, c)
		}
	}

	data := make([]float32, 3*tam*tam)
	for y := 0; y < tam; y++ {
		for x := 0; x < tam; x++ {
			r, g, b, _ := quad.At(x, y).RGBA()
			i := y*tam + x
			data[i] = float32(r>>8) / 255
			data[tam*tam+i] = float32(g>>8) / 255
			data[2*tam*tam+i] = float32(b>>8) / 255
		}
	}
	return tensor.FromSlice(data, 1, 3, tam, tam)
}

func refletirPixel(img image.Image, x, y, w, h int) color.Color {
	if x < 0 {
		x = -x
	}
	if x >= w {
		x = 2*w - x - 2
	}
	if y < 0 {
		y = -y
	}
	if y >= h {
		y = 2*h - y - 2
	}
	if x < 0 {
		x = 0
	}
	if y < 0 {
		y = 0
	}
	if x >= w {
		x = w - 1
	}
	if y >= h {
		y = h - 1
	}
	return img.At(x, y)
}

func redimensionar(img image.Image, nw, nh int) image.Image {
	b := img.Bounds()
	ow, oh := b.Dx(), b.Dy()
	out := image.NewRGBA(image.Rect(0, 0, nw, nh))
	for y := 0; y < nh; y++ {
		sy := float64(y) * float64(oh) / float64(nh)
		for x := 0; x < nw; x++ {
			sx := float64(x) * float64(ow) / float64(nw)
			out.Set(x, y, amostrar(img, sx, sy))
		}
	}
	return out
}

func amostrar(img image.Image, x, y float64) color.Color {
	b := img.Bounds()
	ix := int(math.Round(x))
	iy := int(math.Round(y))
	ix = max(b.Min.X, min(b.Max.X-1, ix))
	iy = max(b.Min.Y, min(b.Max.Y-1, iy))
	return img.At(ix, iy)
}
