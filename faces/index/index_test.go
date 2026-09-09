package index

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
)

func perto(a, b, tol float64) bool { return math.Abs(a-b) <= tol }

// vetorAleatorio gera um vetor de direcao arbitraria.
func vetorAleatorio(r *rand.Rand, dim int) []float32 {
	v := make([]float32, dim)
	for i := range v {
		v[i] = float32(r.NormFloat64())
	}
	return v
}

// perturbado devolve o vetor com um empurrao, simulando a mesma pessoa
// fotografada noutra condicao.
func perturbado(r *rand.Rand, v []float32, forca float64) []float32 {
	out := make([]float32, len(v))
	for i, x := range v {
		out[i] = x + float32(r.NormFloat64()*forca)
	}
	return out
}

// ---------- similaridade ----------

func TestCosine(t *testing.T) {
	casos := []struct {
		nome  string
		a, b  []float32
		quero float32
	}{
		{"identicos", []float32{1, 0, 0}, []float32{1, 0, 0}, 1},
		{"opostos", []float32{1, 0, 0}, []float32{-1, 0, 0}, -1},
		{"ortogonais", []float32{1, 0, 0}, []float32{0, 1, 0}, 0},
		{"45 graus", []float32{1, 0}, []float32{1, 1}, float32(1 / math.Sqrt2)},
		{"escala nao importa", []float32{1, 0}, []float32{7, 0}, 1},
		{"vetor zero", []float32{0, 0}, []float32{1, 0}, 0},
		{"tamanhos diferentes", []float32{1, 0}, []float32{1, 0, 0}, 0},
		{"vazios", nil, nil, 0},
	}

	for _, c := range casos {
		t.Run(c.nome, func(t *testing.T) {
			if got := Cosine(c.a, c.b); !perto(float64(got), float64(c.quero), 1e-6) {
				t.Errorf("Cosine = %v, quero %v", got, c.quero)
			}
			if got := Cosine(c.b, c.a); !perto(float64(got), float64(c.quero), 1e-6) {
				t.Errorf("Cosine invertido = %v, quero %v", got, c.quero)
			}
		})
	}
}

// TestCosineVarreIntervalo: um teste que so conferisse alguns valores
// escolhidos passaria com uma implementacao quase certa. Varrer angulos
// conhecidos nao deixa.
func TestCosineVarreAngulos(t *testing.T) {
	for grau := 0; grau <= 360; grau += 5 {
		rad := float64(grau) * math.Pi / 180
		a := []float32{1, 0}
		b := []float32{float32(math.Cos(rad)), float32(math.Sin(rad))}

		quero := math.Cos(rad)
		if got := Cosine(a, b); !perto(float64(got), quero, 1e-6) {
			t.Errorf("%d graus: Cosine = %v, quero %v", grau, got, quero)
		}
	}
}

func TestNormalize(t *testing.T) {
	r := rand.New(rand.NewSource(1))

	for i := 0; i < 50; i++ {
		v := vetorAleatorio(r, 16)
		n, err := Normalize(v)
		if err != nil {
			t.Fatalf("Normalize: %v", err)
		}

		var soma float64
		for _, x := range n {
			soma += float64(x) * float64(x)
		}
		if !perto(math.Sqrt(soma), 1, 1e-6) {
			t.Errorf("comprimento = %v, quero 1", math.Sqrt(soma))
		}

		// A direcao tem de ser preservada.
		if got := Cosine(v, n); !perto(float64(got), 1, 1e-6) {
			t.Errorf("normalizar mudou a direcao: cosseno %v com o original", got)
		}
	}
}

func TestNormalizeRecusaInvalidos(t *testing.T) {
	casos := []struct {
		nome string
		v    []float32
	}{
		{"vazio", nil},
		{"comprimento zero", []float32{0, 0, 0}},
		{"NaN", []float32{float32(math.NaN()), 1}},
		{"infinito", []float32{float32(math.Inf(1)), 1}},
	}

	for _, c := range casos {
		t.Run(c.nome, func(t *testing.T) {
			if _, err := Normalize(c.v); err == nil {
				t.Error("deveria dar erro")
			}
		})
	}
}

func TestNormalizeCopia(t *testing.T) {
	// O vetor de origem costuma vir do espaco de trabalho da rede, que sera
	// reaproveitado no proximo rosto. Guardar uma referencia seria um bug
	// que so aparece no segundo rosto.
	v := []float32{3, 4}
	n, err := Normalize(v)
	if err != nil {
		t.Fatal(err)
	}

	v[0] = 99
	if n[0] == 99 {
		t.Error("Normalize devolveu uma vista do original, nao uma copia")
	}
	if !perto(float64(n[0]), 0.6, 1e-6) {
		t.Errorf("n[0] = %v, quero 0.6", n[0])
	}
}

// ---------- cadastro e busca ----------

func TestNewRecusaDimensaoInvalida(t *testing.T) {
	for _, d := range []int{0, -1, -128} {
		if _, err := New(d); err == nil {
			t.Errorf("New(%d) deveria dar erro", d)
		}
	}
}

func TestAddRecusaInvalidos(t *testing.T) {
	ix, err := New(4)
	if err != nil {
		t.Fatal(err)
	}

	if err := ix.Add("", []float32{1, 0, 0, 0}); err == nil {
		t.Error("identidade vazia deveria dar erro")
	}
	if err := ix.Add("joao", []float32{1, 0}); err == nil {
		t.Error("dimensao errada deveria dar erro")
	}
	if err := ix.Add("joao", []float32{0, 0, 0, 0}); err == nil {
		t.Error("vetor de comprimento zero deveria dar erro")
	}
	if ix.Len() != 0 {
		t.Errorf("depois de tres erros o indice tem %d vetores, quero 0", ix.Len())
	}
}

func TestBuscaEmIndiceVazio(t *testing.T) {
	ix, _ := New(4)

	m, ok, err := ix.Search([]float32{1, 0, 0, 0})
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if ok {
		t.Errorf("indice vazio devolveu %+v, quero ok=false", m)
	}
}

// TestBuscaAchaAPessoaCerta e o teste central do pacote. Varre um cadastro
// inteiro em vez de conferir um caso sortudo: cada pessoa cadastrada e
// procurada com uma versao perturbada do proprio vetor, e tem de voltar ela
// mesma.
func TestBuscaAchaAPessoaCerta(t *testing.T) {
	const dim, pessoas, amostras = 128, 60, 4

	r := rand.New(rand.NewSource(20260909))
	ix, err := New(dim)
	if err != nil {
		t.Fatal(err)
	}

	base := make([][]float32, pessoas)
	for p := 0; p < pessoas; p++ {
		base[p] = vetorAleatorio(r, dim)
		id := fmt.Sprintf("pessoa-%02d", p)
		for a := 0; a < amostras; a++ {
			if err := ix.Add(id, perturbado(r, base[p], 0.25)); err != nil {
				t.Fatal(err)
			}
		}
	}

	if got := ix.Len(); got != pessoas*amostras {
		t.Errorf("%d vetores, quero %d", got, pessoas*amostras)
	}
	if got := ix.Identidades(); got != pessoas {
		t.Errorf("%d identidades, quero %d", got, pessoas)
	}

	erros := 0
	for p := 0; p < pessoas; p++ {
		consulta := perturbado(r, base[p], 0.25)
		m, ok, err := ix.Search(consulta)
		if err != nil {
			t.Fatal(err)
		}
		if !ok {
			t.Fatal("busca em indice cheio devolveu ok=false")
		}
		if quero := fmt.Sprintf("pessoa-%02d", p); m.ID != quero {
			t.Errorf("consulta de %s achou %s (score %.4f)", quero, m.ID, m.Score)
			erros++
		}
	}
	if erros > 0 {
		t.Fatalf("%d de %d consultas acharam a pessoa errada", erros, pessoas)
	}
}

// TestBuscaBateComOCossenoIngenuo confere o caminho otimizado -- que usa o
// produto matriz-vetor do kernel -- contra a conta obvia, feita vetor a
// vetor.
func TestBuscaBateComOCossenoIngenuo(t *testing.T) {
	const dim, n = 64, 200

	r := rand.New(rand.NewSource(7))
	ix, _ := New(dim)

	vetores := make([][]float32, n)
	for i := 0; i < n; i++ {
		vetores[i] = vetorAleatorio(r, dim)
		if err := ix.Add(fmt.Sprintf("id-%03d", i), vetores[i]); err != nil {
			t.Fatal(err)
		}
	}

	for consultaN := 0; consultaN < 20; consultaN++ {
		q := vetorAleatorio(r, dim)

		// Referencia: cosseno contra cada um, na mao.
		melhorI, melhorS := -1, float32(-2)
		for i, v := range vetores {
			if s := Cosine(q, v); s > melhorS {
				melhorI, melhorS = i, s
			}
		}

		m, ok, err := ix.Search(q)
		if err != nil || !ok {
			t.Fatalf("Search: %v ok=%v", err, ok)
		}

		if quero := fmt.Sprintf("id-%03d", melhorI); m.ID != quero {
			t.Errorf("consulta %d: achou %s, o cosseno ingenuo diz %s", consultaN, m.ID, quero)
		}
		if !perto(float64(m.Score), float64(melhorS), 1e-5) {
			t.Errorf("consulta %d: score %v, o cosseno ingenuo diz %v", consultaN, m.Score, melhorS)
		}
	}
}

// TestSearchKDevolveIdentidadesDistintas: sem isso, um k de 3 num cadastro
// com cinco amostras por pessoa devolveria a mesma pessoa tres vezes.
func TestSearchKDevolveIdentidadesDistintas(t *testing.T) {
	r := rand.New(rand.NewSource(3))
	ix, _ := New(32)

	for p := 0; p < 5; p++ {
		v := vetorAleatorio(r, 32)
		for a := 0; a < 5; a++ {
			if err := ix.Add(fmt.Sprintf("p%d", p), perturbado(r, v, 0.1)); err != nil {
				t.Fatal(err)
			}
		}
	}

	res, err := ix.SearchK(vetorAleatorio(r, 32), 3)
	if err != nil {
		t.Fatal(err)
	}
	if len(res) != 3 {
		t.Fatalf("%d resultados, quero 3", len(res))
	}

	visto := map[string]bool{}
	for _, m := range res {
		if visto[m.ID] {
			t.Errorf("%s apareceu duas vezes", m.ID)
		}
		visto[m.ID] = true
	}

	for i := 1; i < len(res); i++ {
		if res[i-1].Score < res[i].Score {
			t.Errorf("resultados fora de ordem: %v", res)
		}
	}
}

// TestSearchKPegaAMelhorAmostra: com varias amostras por pessoa, a
// pontuacao devolvida tem de ser a MELHOR delas -- e a razao de cadastrar
// mais de uma foto.
func TestSearchKPegaAMelhorAmostra(t *testing.T) {
	ix, _ := New(3)

	// Uma amostra apontando exatamente para a consulta, outra longe.
	if err := ix.Add("joao", []float32{1, 0, 0}); err != nil {
		t.Fatal(err)
	}
	if err := ix.Add("joao", []float32{0, 1, 0}); err != nil {
		t.Fatal(err)
	}

	m, ok, err := ix.Search([]float32{1, 0, 0})
	if err != nil || !ok {
		t.Fatalf("Search: %v ok=%v", err, ok)
	}
	if !perto(float64(m.Score), 1, 1e-6) {
		t.Errorf("score = %v, quero 1 (a melhor das duas amostras)", m.Score)
	}
}

func TestSearchKRecusaInvalidos(t *testing.T) {
	ix, _ := New(4)
	ix.Add("a", []float32{1, 0, 0, 0})

	if _, err := ix.SearchK([]float32{1, 0, 0, 0}, 0); err == nil {
		t.Error("k = 0 deveria dar erro")
	}
	if _, err := ix.SearchK([]float32{1, 0}, 1); err == nil {
		t.Error("dimensao errada deveria dar erro")
	}
	if _, err := ix.SearchK([]float32{0, 0, 0, 0}, 1); err == nil {
		t.Error("consulta de comprimento zero deveria dar erro")
	}
}

func TestSearchKMaiorQueOCadastro(t *testing.T) {
	ix, _ := New(3)
	ix.Add("a", []float32{1, 0, 0})
	ix.Add("b", []float32{0, 1, 0})

	res, err := ix.SearchK([]float32{1, 1, 0}, 10)
	if err != nil {
		t.Fatal(err)
	}
	if len(res) != 2 {
		t.Errorf("%d resultados, quero 2 (o cadastro so tem duas identidades)", len(res))
	}
}

// TestEmpateEDeterministico: duas identidades com a mesma pontuacao nao
// podem sair em ordem que dependa de como o mapa foi percorrido.
func TestEmpateEDeterministico(t *testing.T) {
	ix, _ := New(2)
	// Simetricos em relacao a consulta: mesma pontuacao.
	ix.Add("zeta", []float32{1, 1})
	ix.Add("alfa", []float32{1, -1})

	for i := 0; i < 20; i++ {
		res, err := ix.SearchK([]float32{1, 0}, 2)
		if err != nil {
			t.Fatal(err)
		}
		if res[0].ID != "alfa" {
			t.Fatalf("no empate saiu %s primeiro; quero alfa, por ordem de nome", res[0].ID)
		}
	}
}

// ---------- remocao ----------

// TestRemove cobre o direito de eliminacao. Dado biometrico e dado pessoal
// sensivel, e o direito so vale se houver como exerce-lo de verdade.
func TestRemove(t *testing.T) {
	r := rand.New(rand.NewSource(11))
	ix, _ := New(16)

	base := map[string][]float32{}
	for _, id := range []string{"ana", "bruno", "carla", "diego"} {
		v := vetorAleatorio(r, 16)
		base[id] = v
		for a := 0; a < 3; a++ {
			if err := ix.Add(id, perturbado(r, v, 0.1)); err != nil {
				t.Fatal(err)
			}
		}
	}

	if got := ix.Remove("bruno"); got != 3 {
		t.Errorf("removeu %d vetores, quero 3", got)
	}
	if got := ix.Remove("nao-existe"); got != 0 {
		t.Errorf("remover quem nao existe devolveu %d, quero 0", got)
	}

	if ix.Identidades() != 3 || ix.Len() != 9 {
		t.Errorf("sobraram %d identidades e %d vetores, quero 3 e 9", ix.Identidades(), ix.Len())
	}
	for _, id := range ix.IDs() {
		if id == "bruno" {
			t.Error("bruno continua na lista de identidades")
		}
	}

	// O que sobrou tem de continuar sendo achado corretamente -- e a parte
	// que a reindexacao dos donos pode quebrar em silencio.
	for _, id := range []string{"ana", "carla", "diego"} {
		m, ok, err := ix.Search(perturbado(r, base[id], 0.1))
		if err != nil || !ok {
			t.Fatalf("Search de %s: %v ok=%v", id, err, ok)
		}
		if m.ID != id {
			t.Errorf("depois da remocao, %s foi achado como %s", id, m.ID)
		}
	}

	// E bruno nao pode mais ser achado, nem com o vetor dele.
	m, ok, _ := ix.Search(base["bruno"])
	if ok && m.ID == "bruno" {
		t.Error("bruno foi achado depois de removido")
	}
}

func TestRemoveTudo(t *testing.T) {
	ix, _ := New(3)
	ix.Add("a", []float32{1, 0, 0})
	ix.Add("b", []float32{0, 1, 0})

	ix.Remove("a")
	ix.Remove("b")

	if ix.Len() != 0 || ix.Identidades() != 0 {
		t.Errorf("sobraram %d vetores e %d identidades, quero 0 e 0", ix.Len(), ix.Identidades())
	}
	if _, ok, _ := ix.Search([]float32{1, 0, 0}); ok {
		t.Error("indice vazio ainda devolve resultado")
	}

	// E tem de voltar a aceitar cadastro.
	if err := ix.Add("c", []float32{0, 0, 1}); err != nil {
		t.Fatalf("Add depois de esvaziar: %v", err)
	}
	if m, ok, _ := ix.Search([]float32{0, 0, 1}); !ok || m.ID != "c" {
		t.Errorf("depois de recadastrar, Search devolveu %+v ok=%v", m, ok)
	}
}

// ---------- benchmarks ----------

// BenchmarkSearch mede a busca exaustiva em varias escalas de cadastro.
//
// E o numero que decide se vale a pena um indice aproximado tipo HNSW: se a
// busca linear ja responde em microssegundos na escala real de uso, o indice
// aproximado seria complexidade sem beneficio.
//
// Sao cinco amostras por pessoa, que e o cadastro que a documentacao
// recomenda. A primeira versao deste benchmark usava uma amostra por pessoa,
// e isso escondia um gargalo: com uma identidade por vetor, o custo de
// escolher o melhor resultado crescia junto com o cadastro.
func BenchmarkSearch(b *testing.B) {
	const dim, amostras = 128, 5

	for _, pessoas := range []int{20, 200, 2000, 20000} {
		b.Run(fmt.Sprintf("%d_pessoas", pessoas), func(b *testing.B) {
			r := rand.New(rand.NewSource(1))
			ix, _ := New(dim)
			for p := 0; p < pessoas; p++ {
				v := vetorAleatorio(r, dim)
				for a := 0; a < amostras; a++ {
					if err := ix.Add(fmt.Sprintf("id-%06d", p), perturbado(r, v, 0.2)); err != nil {
						b.Fatal(err)
					}
				}
			}

			q := vetorAleatorio(r, dim)
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if _, _, err := ix.Search(q); err != nil {
					b.Fatal(err)
				}
			}
			b.ReportMetric(float64(ix.Len()), "vetores")
		})
	}
}

// BenchmarkSearchK mede o custo de pedir varios resultados, que e o caminho
// da insercao limitada.
func BenchmarkSearchK(b *testing.B) {
	const dim = 128

	r := rand.New(rand.NewSource(1))
	ix, _ := New(dim)
	for p := 0; p < 2000; p++ {
		v := vetorAleatorio(r, dim)
		for a := 0; a < 5; a++ {
			ix.Add(fmt.Sprintf("id-%06d", p), perturbado(r, v, 0.2))
		}
	}
	q := vetorAleatorio(r, dim)

	for _, k := range []int{1, 5, 50} {
		b.Run(fmt.Sprintf("k=%d", k), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				if _, err := ix.SearchK(q, k); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

func BenchmarkAdd(b *testing.B) {
	r := rand.New(rand.NewSource(1))
	ix, _ := New(128)
	v := vetorAleatorio(r, 128)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := ix.Add("mesma-pessoa", v); err != nil {
			b.Fatal(err)
		}
	}
}
