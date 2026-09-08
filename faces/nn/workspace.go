// Package nn monta as camadas da rede sobre os kernels numericos.
//
// Uma camada guarda os pesos e a geometria, e sabe transformar um tensor de
// entrada num de saida. O pacote kernel sabe fazer as contas; o nn sabe o
// que ligar em que.
//
// Duas decisoes atravessam o pacote inteiro:
//
// Workspace. Uma rede tem dezenas de camadas, e cada uma produz um tensor
// intermediario. Alocar tudo isso a cada rosto processado seria pressao de
// coletor de lixo pura, num caminho que roda em milissegundos. O Workspace
// aloca na primeira passagem e reaproveita em todas as seguintes.
//
// Fusao de BatchNorm. Na inferencia, BatchNorm e apenas uma multiplicacao e
// uma soma por canal -- e as duas cabem dentro dos pesos da convolucao
// anterior. Fundida, a camada some por completo do tempo de execucao. Ver
// Sequential.Fuse.
package nn

import "github.com/iafrotamacedo-cloud/era/faces/tensor"

// blocoMinimo e o tamanho minimo de cada bloco do Workspace, em float32.
// 64Ki float32 = 256 KB, grande o bastante para caber varios tensores
// intermediarios sem virar desperdicio.
const blocoMinimo = 1 << 16

// Workspace fornece memoria reutilizavel para os resultados intermediarios
// de uma passagem pela rede.
//
// Funciona como uma pilha: Alloc avanca, Reset volta ao inicio. A primeira
// passagem aloca; as seguintes nao alocam nada.
//
// A memoria e organizada em blocos que nunca sao realocados. Isso importa:
// se o Workspace crescesse realocando um unico buffer, as fatias devolvidas
// antes passariam a apontar para memoria abandonada -- um bug silencioso e
// desagradavel de rastrear.
//
// Um Workspace NAO e seguro para uso concorrente. Use um por goroutine.
type Workspace struct {
	blocks [][]float32
	cur    int // bloco em uso
	off    int // quanto do bloco atual ja foi entregue
}

// NewWorkspace cria um workspace vazio. Ele cresce conforme a necessidade.
func NewWorkspace() *Workspace { return &Workspace{} }

// Alloc devolve uma fatia de n float32.
//
// O conteudo e LIXO -- memoria reaproveitada de passagens anteriores. Quem
// chama precisa escrever em todas as posicoes, ou usar AllocZeroed.
func (w *Workspace) Alloc(n int) []float32 {
	if n <= 0 {
		return nil
	}

	// Procura espaco nos blocos que ja existem.
	for w.cur < len(w.blocks) {
		b := w.blocks[w.cur]
		if w.off+n <= len(b) {
			s := b[w.off : w.off+n : w.off+n]
			w.off += n
			return s
		}
		w.cur++
		w.off = 0
	}

	// Nenhum bloco serve: cria mais um. Blocos antigos ficam intactos, e as
	// fatias entregues antes continuam validas.
	size := max(n, blocoMinimo)
	b := make([]float32, size)
	w.blocks = append(w.blocks, b)
	w.cur = len(w.blocks) - 1
	w.off = n
	return b[0:n:n]
}

// AllocZeroed e como Alloc, mas devolve a fatia zerada.
func (w *Workspace) AllocZeroed(n int) []float32 {
	s := w.Alloc(n)
	clear(s)
	return s
}

// Tensor devolve um tensor com a forma pedida, apoiado na memoria do
// workspace.
//
// O conteudo e lixo, pela mesma razao que em Alloc.
func (w *Workspace) Tensor(shape ...int) *tensor.Tensor {
	n := 1
	for _, d := range shape {
		n *= d
	}
	t, err := tensor.FromSlice(w.Alloc(n), shape...)
	if err != nil {
		panic("nn: forma invalida no Workspace: " + err.Error())
	}
	return t
}

// Reset devolve toda a memoria entregue ao pool, sem liberar nada ao
// sistema. Chame entre uma passagem e outra.
//
// Todo tensor obtido do workspace antes do Reset passa a ser invalido: a
// memoria dele sera reentregue na proxima passagem. Se precisar guardar um
// resultado, copie antes.
func (w *Workspace) Reset() {
	w.cur = 0
	w.off = 0
}

// Cap informa quantos float32 o workspace tem reservados no total. Serve
// para medir quanta memoria uma rede consome de fato.
func (w *Workspace) Cap() int {
	n := 0
	for _, b := range w.blocks {
		n += len(b)
	}
	return n
}

// Release libera a memoria ao sistema. Use so ao descartar o workspace; o
// uso normal e Reset.
func (w *Workspace) Release() {
	w.blocks = nil
	w.cur = 0
	w.off = 0
}
