package dewarp

import (
	"testing"

	"github.com/iafrotamacedo-cloud/era/documents/geom"
)

// pt e um atalho para geom.Point{X: x, Y: y} -- so para estes testes nao
// afogarem em campos nomeados: go vet exige campos nomeados em literais de
// struct de outro pacote, e geom.Point e de outro pacote aqui.
func pt(x, y float64) geom.Point { return geom.Point{X: x, Y: y} }

func TestExtractBaselineQuadrilatero(t *testing.T) {
	// superior-esquerdo, superior-direito, inferior-direito, inferior-esquerdo
	p := geom.Polygon{pt(0, 10), pt(100, 8), pt(100, 20), pt(0, 22)}

	got, err := ExtractBaseline(p)
	if err != nil {
		t.Fatalf("ExtractBaseline: %v", err)
	}
	want := []geom.Point{pt(0, 22), pt(100, 20)}
	if len(got) != len(want) {
		t.Fatalf("len = %d, quero %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("baseline[%d] = %v, quero %v", i, got[i], want[i])
		}
	}
}

func TestExtractBaselinePoligonoCurvo(t *testing.T) {
	// 3 pontos na borda de cima (esquerda->direita), 3 na de baixo
	// (direita->esquerda) -- convencao de poligono curvo.
	p := geom.Polygon{
		pt(0, 0), pt(50, -5), pt(100, 0), // topo, esq->dir
		pt(100, 20), pt(50, 15), pt(0, 20), // base, dir->esq
	}
	got, err := ExtractBaseline(p)
	if err != nil {
		t.Fatalf("ExtractBaseline: %v", err)
	}
	want := []geom.Point{pt(0, 20), pt(50, 15), pt(100, 20)}
	if len(got) != len(want) {
		t.Fatalf("len = %d, quero %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("baseline[%d] = %v, quero %v (deveria estar em ordem esq->dir)", i, got[i], want[i])
		}
	}
}

func TestExtractBaselineErros(t *testing.T) {
	casos := []struct {
		nome string
		p    geom.Polygon
	}{
		{"triangulo (3 vertices, poucos)", geom.Polygon{pt(0, 0), pt(1, 0), pt(1, 1)}},
		{"numero impar de vertices", geom.Polygon{pt(0, 0), pt(1, 0), pt(2, 0), pt(2, 1), pt(0, 1)}},
		{"vazio", geom.Polygon{}},
	}
	for _, c := range casos {
		t.Run(c.nome, func(t *testing.T) {
			if _, err := ExtractBaseline(c.p); err == nil {
				t.Errorf("ExtractBaseline(%v) deveria falhar", c.p)
			}
		})
	}
}
