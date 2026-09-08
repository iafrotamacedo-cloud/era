// Package graph executa o grafo de um modelo ONNX.
//
// O pacote onnx le o arquivo e devolve a estrutura crua. Este pacote pega
// essa estrutura, resolve as ligacoes entre os nos, monta as camadas do
// pacote nn com os pesos certos e executa.
//
// # Como o ONNX liga os nos
//
// Nao ha ponteiros no formato: as ligacoes sao NOMES. A saida "conv1_out" de
// um no e a entrada "conv1_out" do proximo. Montar o grafo e resolver esses
// nomes, e executar e manter uma tabela de nome para tensor.
//
// Tres origens alimentam essa tabela:
//
//	initializers  pesos treinados, presentes desde o inicio
//	inputs        o que quem chama fornece -- a imagem
//	saidas        produzidas pelos nos, conforme executam
//
// # Ordem de execucao
//
// A especificacao do ONNX exige que os nos venham em ordem topologica. Este
// pacote nao confia nisso e reordena. Custa uma passada e elimina uma classe
// inteira de falha dificil de diagnosticar: um exportador que grave fora de
// ordem produziria um erro de "valor nao encontrado" no meio da rede, sem
// nenhuma pista da causa.
package graph

import (
	"fmt"
	"sort"
	"strings"

	"github.com/iafrotamacedo-cloud/era/faces/nn"
	"github.com/iafrotamacedo-cloud/era/faces/onnx"
	"github.com/iafrotamacedo-cloud/era/faces/tensor"
)

// Graph e um modelo pronto para executar.
//
// Nao guarda estado de execucao: o mesmo Graph pode ser usado por varias
// goroutines ao mesmo tempo, desde que cada uma traga o seu proprio
// nn.Workspace.
type Graph struct {
	Name string

	ops    []*operation
	consts map[string]*tensor.Tensor

	inputs  []string
	outputs []string
}

// operation e um no ja resolvido: entradas, saidas e o que executar.
//
// A funcao exec e criada na montagem, capturando os pesos e as camadas do
// pacote nn. Assim toda a validacao -- forma dos pesos, atributos coerentes,
// combinacoes nao suportadas -- acontece uma vez, ao carregar, e nao a cada
// rosto processado.
type operation struct {
	name    string
	opType  string
	inputs  []string
	outputs []string
	exec    func(ws *nn.Workspace, ins []*tensor.Tensor) ([]*tensor.Tensor, error)
}

// New monta um grafo executavel a partir de um modelo lido.
func New(m *onnx.Model) (*Graph, error) {
	if m == nil || m.Graph == nil {
		return nil, fmt.Errorf("graph: modelo sem grafo")
	}

	g := &Graph{
		Name:   m.Graph.Name,
		consts: make(map[string]*tensor.Tensor, len(m.Graph.Initializers)),
	}

	// 1. Os pesos treinados viram tensores de uma vez.
	for _, init := range m.Graph.Initializers {
		vals, err := init.Floats()
		if err != nil {
			return nil, fmt.Errorf("graph: peso %q: %w", init.Name, err)
		}
		t, err := tensor.FromSlice(vals, init.Shape()...)
		if err != nil {
			return nil, fmt.Errorf("graph: peso %q: %w", init.Name, err)
		}
		g.consts[init.Name] = t
	}

	// 2. Entradas do grafo sao as declaradas que nao sao pesos. Muitos
	// exportadores listam os initializers tambem como inputs, por
	// compatibilidade com versoes antigas do formato.
	for _, in := range m.Graph.Inputs {
		if _, ehPeso := g.consts[in.Name]; !ehPeso {
			g.inputs = append(g.inputs, in.Name)
		}
	}
	for _, out := range m.Graph.Outputs {
		g.outputs = append(g.outputs, out.Name)
	}

	// 3. Ordena antes de montar, para que a montagem possa contar com os
	// valores anteriores ja definidos.
	ordenados, err := ordenarTopologicamente(m.Graph.Nodes, g.consts, g.inputs)
	if err != nil {
		return nil, err
	}

	// 4. Monta cada no.
	b := &builder{consts: g.consts}
	var faltando []string

	for _, n := range ordenados {
		if n.Domain != "" && n.Domain != "ai.onnx" {
			return nil, fmt.Errorf("graph: no %s usa o dominio %q, que a ERA nao conhece", n, n.Domain)
		}

		monta, ok := registro[n.OpType]
		if !ok {
			faltando = append(faltando, n.OpType)
			continue
		}

		op, err := monta(b, n)
		if err != nil {
			return nil, fmt.Errorf("graph: %s: %w", n, err)
		}
		g.ops = append(g.ops, op)
	}

	// Junta TODOS os operadores faltantes numa mensagem so. Reportar um por
	// vez transformaria carregar um modelo novo numa sequencia de tentativas.
	if len(faltando) > 0 {
		return nil, fmt.Errorf("graph: operadores nao implementados: %s", listaUnica(faltando))
	}

	return g, nil
}

// Inputs devolve os nomes que Run espera receber.
func (g *Graph) Inputs() []string { return append([]string(nil), g.inputs...) }

// Outputs devolve os nomes que Run produz.
func (g *Graph) Outputs() []string { return append([]string(nil), g.outputs...) }

// Ops informa quantas operacoes o grafo tem.
func (g *Graph) Ops() int { return len(g.ops) }

// Run executa o grafo.
//
// Os tensores devolvidos costumam apontar para memoria do workspace, e so
// valem ate o proximo ws.Reset. Copie o que precisar guardar.
func (g *Graph) Run(ws *nn.Workspace, inputs map[string]*tensor.Tensor) (map[string]*tensor.Tensor, error) {
	if ws == nil {
		return nil, fmt.Errorf("graph: workspace nulo")
	}

	// A tabela nome -> tensor. Comeca com os pesos e as entradas.
	env := make(map[string]*tensor.Tensor, len(g.consts)+len(g.ops)*2)
	for k, v := range g.consts {
		env[k] = v
	}
	for _, nome := range g.inputs {
		t, ok := inputs[nome]
		if !ok || t == nil {
			return nil, fmt.Errorf("graph: falta a entrada %q (o grafo espera %v)", nome, g.inputs)
		}
		env[nome] = t
	}

	for _, op := range g.ops {
		ins := make([]*tensor.Tensor, len(op.inputs))
		for i, nome := range op.inputs {
			if nome == "" {
				continue // entrada opcional omitida: o ONNX marca com nome vazio
			}
			t, ok := env[nome]
			if !ok {
				return nil, fmt.Errorf("graph: %s (%s): o valor %q nao existe", op.name, op.opType, nome)
			}
			ins[i] = t
		}

		outs, err := op.exec(ws, ins)
		if err != nil {
			return nil, fmt.Errorf("graph: %s (%s): %w", op.name, op.opType, err)
		}
		if len(outs) != len(op.outputs) {
			return nil, fmt.Errorf("graph: %s (%s) produziu %d saidas, o no declara %d",
				op.name, op.opType, len(outs), len(op.outputs))
		}

		for i, nome := range op.outputs {
			if nome != "" {
				env[nome] = outs[i]
			}
		}
	}

	res := make(map[string]*tensor.Tensor, len(g.outputs))
	for _, nome := range g.outputs {
		t, ok := env[nome]
		if !ok {
			return nil, fmt.Errorf("graph: a saida %q nao foi produzida", nome)
		}
		res[nome] = t
	}
	return res, nil
}

// ordenarTopologicamente devolve os nos em ordem de execucao, pelo algoritmo
// de Kahn.
//
// Um no so pode rodar depois de todos os que produzem as suas entradas.
// Valores que ja existem no inicio -- pesos e entradas do grafo -- nao criam
// dependencia.
//
// Em caso de empate, mantem a ordem original do arquivo. Isso torna a saida
// deterministica, o que importa para que uma falha seja reproduzivel.
func ordenarTopologicamente(nodes []*onnx.Node, consts map[string]*tensor.Tensor, inputs []string) ([]*onnx.Node, error) {
	if len(nodes) == 0 {
		return nil, nil
	}

	// Quem produz cada valor.
	produtor := make(map[string]int, len(nodes)*2)
	for i, n := range nodes {
		for _, out := range n.Outputs {
			if out == "" {
				continue
			}
			if anterior, existe := produtor[out]; existe {
				return nil, fmt.Errorf("graph: o valor %q e produzido por dois nos (%s e %s)",
					out, nodes[anterior], n)
			}
			produtor[out] = i
		}
	}

	disponivel := make(map[string]bool, len(consts)+len(inputs))
	for k := range consts {
		disponivel[k] = true
	}
	for _, k := range inputs {
		disponivel[k] = true
	}

	// Arestas: de quem produz para quem consome.
	grau := make([]int, len(nodes))
	consumidores := make([][]int, len(nodes))

	for i, n := range nodes {
		for _, in := range n.Inputs {
			if in == "" || disponivel[in] {
				continue
			}
			j, ok := produtor[in]
			if !ok {
				return nil, fmt.Errorf("graph: %s consome %q, que ninguem produz e nao e peso nem entrada",
					n, in)
			}
			if j == i {
				return nil, fmt.Errorf("graph: %s depende de si mesmo", n)
			}
			grau[i]++
			consumidores[j] = append(consumidores[j], i)
		}
	}

	var fila []int
	for i, g := range grau {
		if g == 0 {
			fila = append(fila, i)
		}
	}
	sort.Ints(fila)

	ordem := make([]*onnx.Node, 0, len(nodes))
	for len(fila) > 0 {
		i := fila[0]
		fila = fila[1:]
		ordem = append(ordem, nodes[i])

		var liberados []int
		for _, j := range consumidores[i] {
			grau[j]--
			if grau[j] == 0 {
				liberados = append(liberados, j)
			}
		}
		// Reinsere mantendo a fila ordenada pelo indice original, para que a
		// ordem final nao dependa da ordem de iteracao.
		fila = append(fila, liberados...)
		sort.Ints(fila)
	}

	if len(ordem) != len(nodes) {
		var presos []string
		for i, g := range grau {
			if g > 0 {
				presos = append(presos, nodes[i].String())
			}
		}
		return nil, fmt.Errorf("graph: o grafo tem ciclo; nos presos: %s", strings.Join(presos, "; "))
	}

	return ordem, nil
}

// listaUnica junta nomes repetidos numa lista ordenada e sem duplicatas.
func listaUnica(itens []string) string {
	visto := make(map[string]bool, len(itens))
	var unicos []string
	for _, s := range itens {
		if !visto[s] {
			visto[s] = true
			unicos = append(unicos, s)
		}
	}
	sort.Strings(unicos)
	return strings.Join(unicos, ", ")
}

// String descreve o grafo operacao a operacao.
func (g *Graph) String() string {
	var b strings.Builder
	fmt.Fprintf(&b, "grafo %q: %d operacoes, entradas %v, saidas %v\n",
		g.Name, len(g.ops), g.inputs, g.outputs)
	for i, op := range g.ops {
		fmt.Fprintf(&b, "  %3d  %-22s %v -> %v\n", i, op.opType, op.inputs, op.outputs)
	}
	return b.String()
}
