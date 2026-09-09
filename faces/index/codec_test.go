package index

import (
	"encoding/binary"
	"fmt"
	"math/rand"
	"strings"
	"testing"
)

// indiceDeTeste monta um indice com conteudo variado.
func indiceDeTeste(t testing.TB, dim, pessoas, amostras int) *Index {
	t.Helper()

	r := rand.New(rand.NewSource(99))
	ix, err := New(dim)
	if err != nil {
		t.Fatal(err)
	}
	for p := 0; p < pessoas; p++ {
		v := vetorAleatorio(r, dim)
		for a := 0; a < amostras; a++ {
			if err := ix.Add(fmt.Sprintf("pessoa-%03d", p), perturbado(r, v, 0.2)); err != nil {
				t.Fatal(err)
			}
		}
	}
	return ix
}

func TestIdaEVolta(t *testing.T) {
	orig := indiceDeTeste(t, 64, 20, 3)

	b, err := orig.MarshalBinary()
	if err != nil {
		t.Fatalf("MarshalBinary: %v", err)
	}

	volta, err := Load(b)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}

	if volta.Dim() != orig.Dim() {
		t.Errorf("Dim = %d, quero %d", volta.Dim(), orig.Dim())
	}
	if volta.Len() != orig.Len() {
		t.Errorf("Len = %d, quero %d", volta.Len(), orig.Len())
	}
	if volta.Identidades() != orig.Identidades() {
		t.Errorf("Identidades = %d, quero %d", volta.Identidades(), orig.Identidades())
	}

	origIDs, voltaIDs := orig.IDs(), volta.IDs()
	for i := range origIDs {
		if origIDs[i] != voltaIDs[i] {
			t.Fatalf("identidade %d = %q, quero %q", i, voltaIDs[i], origIDs[i])
		}
	}

	// O que importa de verdade: as buscas tem de dar exatamente o mesmo.
	r := rand.New(rand.NewSource(5))
	for i := 0; i < 30; i++ {
		q := vetorAleatorio(r, 64)

		a, okA, errA := orig.Search(q)
		bb, okB, errB := volta.Search(q)
		if errA != nil || errB != nil {
			t.Fatalf("Search: %v / %v", errA, errB)
		}
		if okA != okB || a.ID != bb.ID || a.Score != bb.Score {
			t.Fatalf("consulta %d: original %+v, carregado %+v", i, a, bb)
		}
	}
}

func TestIdaEVoltaVazio(t *testing.T) {
	ix, err := New(32)
	if err != nil {
		t.Fatal(err)
	}

	b, err := ix.MarshalBinary()
	if err != nil {
		t.Fatalf("MarshalBinary de indice vazio: %v", err)
	}

	volta, err := Load(b)
	if err != nil {
		t.Fatalf("Load de indice vazio: %v", err)
	}
	if volta.Dim() != 32 || volta.Len() != 0 || volta.Identidades() != 0 {
		t.Errorf("indice vazio voltou como dim=%d len=%d ids=%d",
			volta.Dim(), volta.Len(), volta.Identidades())
	}

	// E precisa continuar utilizavel.
	if err := volta.Add("primeira", vetorAleatorio(rand.New(rand.NewSource(1)), 32)); err != nil {
		t.Errorf("Add depois de carregar vazio: %v", err)
	}
}

func TestIdaEVoltaDepoisDeRemover(t *testing.T) {
	// A remocao reindexa os donos. Se ela deixar o indice inconsistente, a
	// serializacao e o lugar onde isso aparece.
	ix := indiceDeTeste(t, 32, 6, 2)
	ix.Remove("pessoa-002")
	ix.Remove("pessoa-000")

	b, err := ix.MarshalBinary()
	if err != nil {
		t.Fatalf("MarshalBinary: %v", err)
	}
	volta, err := Load(b)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}

	if volta.Identidades() != 4 || volta.Len() != 8 {
		t.Errorf("carregado com %d identidades e %d vetores, quero 4 e 8",
			volta.Identidades(), volta.Len())
	}
	for _, id := range volta.IDs() {
		if id == "pessoa-000" || id == "pessoa-002" {
			t.Errorf("%s sobreviveu a remocao e a serializacao", id)
		}
	}
}

func TestNomesComAcentoEUnicode(t *testing.T) {
	ix, _ := New(4)
	nomes := []string{
		"João Ção",
		"Ana-Lúcia Ñoño",
		"名前",
		"emoji 🙂 no nome",
		strings.Repeat("nome longo ", 50),
	}
	for i, n := range nomes {
		v := make([]float32, 4)
		v[i%4] = 1
		if err := ix.Add(n, v); err != nil {
			t.Fatal(err)
		}
	}

	b, _ := ix.MarshalBinary()
	volta, err := Load(b)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}

	got := volta.IDs()
	if len(got) != len(nomes) {
		t.Fatalf("%d identidades, quero %d", len(got), len(nomes))
	}
	for i := range nomes {
		if got[i] != nomes[i] {
			t.Errorf("identidade %d = %q, quero %q", i, got[i], nomes[i])
		}
	}
}

// TestCorrupcaoEDetectada e a razao de existir um checksum. Sem ele, meio
// arquivo carregaria e produziria respostas erradas em silencio.
func TestCorrupcaoEDetectada(t *testing.T) {
	ix := indiceDeTeste(t, 16, 5, 2)
	bom, err := ix.MarshalBinary()
	if err != nil {
		t.Fatal(err)
	}

	// Vira um bit em varias posicoes ao longo do arquivo: cabecalho, nomes,
	// donos, valores e o proprio checksum.
	for _, pos := range []int{0, 7, 9, 20, 40, len(bom) / 2, len(bom) - 10, len(bom) - 1} {
		if pos < 0 || pos >= len(bom) {
			continue
		}
		t.Run(fmt.Sprintf("byte_%d", pos), func(t *testing.T) {
			ruim := append([]byte(nil), bom...)
			ruim[pos] ^= 0x01

			if _, err := Load(ruim); err == nil {
				t.Error("corrupcao passou despercebida")
			}
		})
	}
}

func TestTruncamentoEDetectado(t *testing.T) {
	ix := indiceDeTeste(t, 16, 5, 2)
	bom, _ := ix.MarshalBinary()

	for _, corte := range []int{0, 1, 8, 15, 24, len(bom) / 3, len(bom) / 2, len(bom) - 1} {
		t.Run(fmt.Sprintf("ate_%d", corte), func(t *testing.T) {
			defer func() {
				if p := recover(); p != nil {
					t.Fatalf("entrou em panico com arquivo truncado: %v", p)
				}
			}()
			if _, err := Load(bom[:corte]); err == nil {
				t.Error("arquivo truncado deveria dar erro")
			}
		})
	}
}

func TestAssinaturaDesconhecida(t *testing.T) {
	lixo := make([]byte, 64)
	copy(lixo, "NAOEUMINDICE")

	_, err := Load(lixo)
	if err == nil {
		t.Fatal("arquivo que nao e um indice deveria dar erro")
	}
	if !strings.Contains(err.Error(), "assinatura") {
		t.Errorf("a mensagem deveria falar em assinatura: %v", err)
	}
}

func TestVersaoDesconhecida(t *testing.T) {
	ix := indiceDeTeste(t, 8, 2, 1)
	b, _ := ix.MarshalBinary()

	// Troca a versao e refaz o checksum, para que a versao seja o unico
	// problema.
	binary.LittleEndian.PutUint32(b[len(magico):], 999)
	refazCRC(b)

	_, err := Load(b)
	if err == nil {
		t.Fatal("versao desconhecida deveria dar erro")
	}
	if !strings.Contains(err.Error(), "versao") {
		t.Errorf("a mensagem deveria falar em versao: %v", err)
	}
}

// TestCabecalhoAbsurdoNaoAloca cobre o caso perigoso: um campo de tamanho
// corrompido pediria gigabytes. O carregamento confere os tamanhos contra o
// que resta no buffer ANTES de reservar memoria.
func TestCabecalhoAbsurdoNaoAloca(t *testing.T) {
	ix := indiceDeTeste(t, 8, 2, 1)

	casos := []struct {
		nome  string
		campo int // deslocamento do campo dentro do cabecalho
		valor uint32
	}{
		{"vetores demais", len(magico) + 12, 1 << 28},
		{"identidades demais", len(magico) + 8, 1 << 28},
		{"dimensao absurda", len(magico) + 4, 1 << 28},
		{"dimensao zero", len(magico) + 4, 0},
		{"vetores sem identidade", len(magico) + 8, 0},
	}

	for _, c := range casos {
		t.Run(c.nome, func(t *testing.T) {
			b, _ := ix.MarshalBinary()
			binary.LittleEndian.PutUint32(b[c.campo:], c.valor)
			refazCRC(b)

			defer func() {
				if p := recover(); p != nil {
					t.Fatalf("entrou em panico: %v", p)
				}
			}()
			if _, err := Load(b); err == nil {
				t.Error("cabecalho absurdo deveria dar erro")
			}
		})
	}
}

func TestDonoForaDoIntervalo(t *testing.T) {
	ix, _ := New(2)
	ix.Add("a", []float32{1, 0})
	ix.Add("b", []float32{0, 1})

	b, _ := ix.MarshalBinary()

	// O primeiro dono vem logo depois dos nomes.
	p := len(magico) + 16 // cabecalho
	p += 4 + len("a")
	p += 4 + len("b")
	binary.LittleEndian.PutUint32(b[p:], 42) // identidade que nao existe
	refazCRC(b)

	if _, err := Load(b); err == nil {
		t.Error("dono apontando para identidade inexistente deveria dar erro")
	}
}

// refazCRC recalcula o checksum depois de uma alteracao proposital, para que
// o teste meca a validacao que quer medir e nao o checksum.
func refazCRC(b []byte) {
	corpo := b[:len(b)-tamCRC]
	binary.LittleEndian.PutUint32(b[len(b)-tamCRC:], crc32IEEE(corpo))
}

func BenchmarkMarshalBinary(b *testing.B) {
	ix := indiceDeTeste(b, 128, 200, 5)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := ix.MarshalBinary(); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkLoad(b *testing.B) {
	ix := indiceDeTeste(b, 128, 200, 5)
	dados, err := ix.MarshalBinary()
	if err != nil {
		b.Fatal(err)
	}
	b.Logf("indice de %d vetores serializa em %.1f KB", ix.Len(), float64(len(dados))/1024)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := Load(dados); err != nil {
			b.Fatal(err)
		}
	}
}
