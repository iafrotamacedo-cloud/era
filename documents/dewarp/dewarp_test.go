package dewarp

import "testing"

func TestLevelString(t *testing.T) {
	casos := []struct {
		l    Level
		want string
	}{
		{N0, "N0"},
		{N1, "N1"},
		{N2, "N2"},
		{N3, "N3"},
		{Level(99), "Level(99)"},
	}
	for _, c := range casos {
		if got := c.l.String(); got != c.want {
			t.Errorf("Level(%d).String() = %q, quero %q", c.l, got, c.want)
		}
	}
}

func TestLevelOrdem(t *testing.T) {
	if !(N0 < N1 && N1 < N2 && N2 < N3) {
		t.Fatal("Level tem que ser crescente N0 < N1 < N2 < N3 -- PageLevel depende disso para achar o pior caso")
	}
}
