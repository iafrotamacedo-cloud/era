package liveness

import (
	"fmt"
	"math"

	"github.com/iafrotamacedo-cloud/era/faces/align"
)

// Options configura a verificacao de movimento entre frames.
type Options struct {
	// MinFrames e o minimo de deteccoes consecutivas. Zero vale 3.
	MinFrames int

	// MinVar e a variancia minima normalizada dos landmarks (px^2 / face^2).
	// Zero vale 0.0005 — calibrar com camera real.
	MinVar float64
}

// Result resume o teste de movimento.
type Result struct {
	Live      bool
	Variance  float64
	FrameUsed int
}

// VerifySequence exige que os landmarks se movam entre frames — foto parada falha.
//
// Cada elemento e os 5 pontos do mesmo rosto num frame. A ordem temporal importa.
func VerifySequence(seq []align.Landmarks, opts Options) (Result, error) {
	if opts.MinFrames == 0 {
		opts.MinFrames = 3
	}
	if opts.MinVar == 0 {
		opts.MinVar = 0.0005
	}
	if len(seq) < opts.MinFrames {
		return Result{}, fmt.Errorf("liveness: preciso de pelo menos %d frames, tenho %d",
			opts.MinFrames, len(seq))
	}

	// Escala de referencia: distancia entre os olhos no primeiro frame.
	ref := distanciaOlhos(seq[0])
	if ref < 1 {
		return Result{}, fmt.Errorf("liveness: olhos muito proximos no primeiro frame")
	}

	var soma float64
	pontos := 5
	for p := 0; p < pontos; p++ {
		mediaX, mediaY := 0.0, 0.0
		for _, lm := range seq {
			mediaX += lm[p].X
			mediaY += lm[p].Y
		}
		n := float64(len(seq))
		mediaX /= n
		mediaY /= n

		var varX, varY float64
		for _, lm := range seq {
			dx := (lm[p].X - mediaX) / ref
			dy := (lm[p].Y - mediaY) / ref
			varX += dx * dx
			varY += dy * dy
		}
		soma += (varX + varY) / n
	}
	varMedia := soma / float64(pontos)

	return Result{
		Live:      varMedia >= opts.MinVar,
		Variance:  varMedia,
		FrameUsed: len(seq),
	}, nil
}

func distanciaOlhos(lm align.Landmarks) float64 {
	dx := lm[align.OlhoEsquerdo].X - lm[align.OlhoDireito].X
	dy := lm[align.OlhoEsquerdo].Y - lm[align.OlhoDireito].Y
	return math.Sqrt(dx*dx + dy*dy)
}
