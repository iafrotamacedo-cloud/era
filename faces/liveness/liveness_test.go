package liveness

import (
	"testing"

	"github.com/iafrotamacedo-cloud/era/faces/align"
)

func TestVerifySequence(t *testing.T) {
	base := align.Landmarks{
		{30, 40}, {70, 40}, {50, 60}, {35, 80}, {65, 80},
	}

	t.Run("foto parada falha", func(t *testing.T) {
		seq := []align.Landmarks{base, base, base}
		r, err := VerifySequence(seq, Options{})
		if err != nil {
			t.Fatal(err)
		}
		if r.Live {
			t.Error("sequencia identica deveria falhar")
		}
	})

	t.Run("movimento passa", func(t *testing.T) {
		seq := make([]align.Landmarks, 5)
		for i := range seq {
			seq[i] = base
			seq[i][align.Nariz].Y += float64(i) * 2
			seq[i][align.OlhoDireito].X += float64(i) * 0.5
		}
		r, err := VerifySequence(seq, Options{MinVar: 0.0001})
		if err != nil {
			t.Fatal(err)
		}
		if !r.Live {
			t.Errorf("movimento deveria passar, var=%v", r.Variance)
		}
	})
}
