package graph

import (
	"encoding/binary"
	"math"
	"os"
	"testing"

	"github.com/iafrotamacedo-cloud/era/faces/nn"
	"github.com/iafrotamacedo-cloud/era/faces/onnx"
	"github.com/iafrotamacedo-cloud/era/faces/tensor"
)

// Teste de integracao contra um modelo de verdade.
//
// Os demais testes deste pacote provam que cada operador faz o que a
// especificacao manda. Este prova a unica coisa que eles nao alcancam: que a
// rede INTEIRA, com 88 operacoes e 9,6 milhoes de parametros, produz o mesmo
// vetor que o ONNX Runtime.
//
// # Por que ele pula por padrao
//
// O repositorio nao guarda pesos, por decisao de licenca -- ver
// docs/faces-modelos.md. Sem o .onnx na maquina, o teste pula em vez de
// falhar: o CI nao tem o modelo e nao deveria ficar vermelho por isso.
//
// # Como rodar
//
//	baixe o SFace do OpenCV Zoo (Apache 2.0) para models/sface.onnx
//	go test ./faces/graph/ -run Referencia -v
//
// Para outro caminho, use a variavel de ambiente ERA_SFACE.
//
// # De onde vem a referencia
//
// Os arquivos em testdata foram gerados por testdata/gerar_referencia.py,
// que roda o mesmo modelo no onnxruntime. A entrada e gravada em disco como
// float32 cru, e nao gerada nas duas linguagens: reproduzir o mesmo
// pseudo-aleatorio em Python e em Go seria uma fonte de divergencia que nao
// tem nada a ver com o que esta sob teste.

const caminhoPadraoSFace = "../../models/sface.onnx"

func caminhoModelo() string {
	if p := os.Getenv("ERA_SFACE"); p != "" {
		return p
	}
	return caminhoPadraoSFace
}

func lerFloat32(t testing.TB, caminho string) []float32 {
	t.Helper()

	b, err := os.ReadFile(caminho)
	if err != nil {
		t.Fatalf("lendo %s: %v", caminho, err)
	}
	if len(b)%4 != 0 {
		t.Fatalf("%s tem %d bytes, que nao dividem em float32", caminho, len(b))
	}

	out := make([]float32, len(b)/4)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[i*4:]))
	}
	return out
}

func carregarSFace(t testing.TB) *Graph {
	t.Helper()

	caminho := caminhoModelo()
	if _, err := os.Stat(caminho); err != nil {
		t.Skipf("modelo nao encontrado em %s; baixe o SFace do OpenCV Zoo ou defina ERA_SFACE", caminho)
	}

	m, err := onnx.Load(caminho)
	if err != nil {
		t.Fatalf("onnx.Load: %v", err)
	}
	g, err := New(m)
	if err != nil {
		t.Fatalf("graph.New: %v", err)
	}
	return g
}

// TestSFaceBateComReferencia e o teste que fecha a Fase 4.
//
// A metrica que decide e a similaridade de cosseno, e nao a diferenca
// elemento a elemento: e ela que o reconhecimento facial de fato usa. Dois
// vetores com pequenas diferencas numericas mas cosseno 1 representam a
// mesma pessoa; e essa a propriedade que precisa ser preservada.
func TestSFaceBateComReferencia(t *testing.T) {
	g := carregarSFace(t)

	entrada := lerFloat32(t, "testdata/entrada_112x112.bin")
	referencia := lerFloat32(t, "testdata/sface_referencia.bin")

	if n := 1 * 3 * 112 * 112; len(entrada) != n {
		t.Fatalf("a entrada tem %d valores, quero %d", len(entrada), n)
	}

	x, err := tensor.FromSlice(entrada, 1, 3, 112, 112)
	if err != nil {
		t.Fatal(err)
	}

	outs, err := g.Run(nn.NewWorkspace(), map[string]*tensor.Tensor{g.Inputs()[0]: x})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}

	nosso := outs[g.Outputs()[0]].Flat()
	if len(nosso) != len(referencia) {
		t.Fatalf("nosso vetor tem %d dimensoes, a referencia tem %d", len(nosso), len(referencia))
	}

	var maxAbs float64
	var pior int
	var dot, na, nb float64

	for i := range nosso {
		if d := math.Abs(float64(nosso[i] - referencia[i])); d > maxAbs {
			maxAbs, pior = d, i
		}
		dot += float64(nosso[i]) * float64(referencia[i])
		na += float64(nosso[i]) * float64(nosso[i])
		nb += float64(referencia[i]) * float64(referencia[i])
	}
	cos := dot / (math.Sqrt(na) * math.Sqrt(nb))

	t.Logf("dimensoes: %d", len(nosso))
	t.Logf("diferenca absoluta maxima: %.3e (posicao %d: %.6f vs %.6f)",
		maxAbs, pior, nosso[pior], referencia[pior])
	t.Logf("similaridade de cosseno: %.9f", cos)
	t.Logf("norma L2: nossa %.6f, referencia %.6f", math.Sqrt(na), math.Sqrt(nb))

	// O limite e folgado em relacao ao medido (1,9e-05) para nao quebrar por
	// diferenca de arredondamento entre arquiteturas -- ARM e x86 podem
	// contrair multiplicacao e soma numa instrucao so, mudando a ultima casa.
	if maxAbs > 1e-3 {
		t.Errorf("diferenca absoluta maxima %.3e passou de 1e-3", maxAbs)
	}
	// Ja o cosseno tem que ser praticamente exato: se ele cair, os vetores
	// apontam para direcoes diferentes e representariam pessoas diferentes.
	if cos < 0.99999 {
		t.Errorf("similaridade de cosseno %.9f abaixo de 0,99999", cos)
	}
}

// TestSFaceEstrutura confere o que a documentacao afirma sobre o modelo.
func TestSFaceEstrutura(t *testing.T) {
	g := carregarSFace(t)

	if got := g.Ops(); got != 88 {
		t.Errorf("%d operacoes, quero 88", got)
	}
	if want := []string{"data"}; len(g.Inputs()) != 1 || g.Inputs()[0] != want[0] {
		t.Errorf("entradas = %v, quero %v (os pesos nao deveriam ser exigidos)", g.Inputs(), want)
	}
	if want := "fc1"; len(g.Outputs()) != 1 || g.Outputs()[0] != want {
		t.Errorf("saidas = %v, quero [%s]", g.Outputs(), want)
	}
}

// TestSFaceDeterminista: a mesma entrada precisa dar exatamente o mesmo
// vetor, sempre. Reconhecimento facial compara vetores gerados em momentos
// diferentes -- se a inferencia variasse entre passagens, o limiar de decisao
// perderia o sentido.
func TestSFaceDeterminista(t *testing.T) {
	g := carregarSFace(t)

	entrada := lerFloat32(t, "testdata/entrada_112x112.bin")
	x, err := tensor.FromSlice(entrada, 1, 3, 112, 112)
	if err != nil {
		t.Fatal(err)
	}

	ws := nn.NewWorkspace()
	ins := map[string]*tensor.Tensor{g.Inputs()[0]: x}

	outs, err := g.Run(ws, ins)
	if err != nil {
		t.Fatal(err)
	}
	primeiro := append([]float32(nil), outs[g.Outputs()[0]].Flat()...)

	for i := 0; i < 5; i++ {
		ws.Reset()
		outs, err := g.Run(ws, ins)
		if err != nil {
			t.Fatal(err)
		}
		for j, v := range outs[g.Outputs()[0]].Flat() {
			if v != primeiro[j] {
				t.Fatalf("passagem %d, dimensao %d: %v, na primeira deu %v", i+1, j, v, primeiro[j])
			}
		}
	}
}

func BenchmarkSFace(b *testing.B) {
	g := carregarSFace(b)

	entrada := lerFloat32(b, "testdata/entrada_112x112.bin")
	x, err := tensor.FromSlice(entrada, 1, 3, 112, 112)
	if err != nil {
		b.Fatal(err)
	}

	ws := nn.NewWorkspace()
	ins := map[string]*tensor.Tensor{g.Inputs()[0]: x}

	// Uma passagem antes de medir, para o workspace ja estar alocado.
	if _, err := g.Run(ws, ins); err != nil {
		b.Fatal(err)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ws.Reset()
		if _, err := g.Run(ws, ins); err != nil {
			b.Fatal(err)
		}
	}
	b.StopTimer()

	b.ReportMetric(float64(ws.Cap()*4)/(1<<20), "MB_workspace")
}
