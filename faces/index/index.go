// Package index guarda vetores de identidade e responde "de quem e este
// rosto".
//
// # A conta
//
// Comparar identidades e medir o angulo entre dois vetores. Se os dois
// estiverem normalizados -- comprimento 1 -- a similaridade de cosseno vira
// simplesmente o produto escalar, sem divisao nenhuma:
//
//	cos(a, b) = a . b        quando |a| = |b| = 1
//
// Por isso o indice normaliza na INSERCAO, uma vez por vetor, em vez de
// normalizar a cada consulta. E por isso uma busca contra N vetores e
// exatamente um produto matriz-vetor, resolvido em score.go -- que explica
// por que o pacote tem o proprio em vez de chamar o kernel.
//
// # Varios vetores por pessoa
//
// Uma foto so de cadastro rende resultado ruim em condicao real: luz
// diferente, capacete, oculos, angulo. Com 3 a 5 amostras por pessoa a taxa
// de acerto sobe bastante, porque a busca compara contra todas e fica com a
// melhor.
//
// Por isso o indice mapeia MUITOS vetores para uma identidade, e a busca
// devolve identidades distintas -- nao vetores.
//
// # O que este pacote nao faz
//
// Nao decide se duas pessoas sao a mesma. Ele devolve um numero; o limiar e
// de quem chama, e precisa ser calibrado com imagens do uso real. Ver
// Sugestao para uma ordem de grandeza.
package index

import (
	"fmt"
	"math"
	"sort"
)

// Cosine devolve a similaridade de cosseno entre dois vetores.
//
// Aceita vetores nao normalizados: normaliza durante a conta. Para comparar
// muitos vetores, prefira o Index, que normaliza uma vez so.
//
// Devolve de -1 a 1. Vetor de comprimento zero devolve 0, porque angulo com
// um vetor sem direcao nao existe.
func Cosine(a, b []float32) float32 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}

	var dot, na, nb float64
	for i, v := range a {
		x := float64(v)
		y := float64(b[i])
		dot += x * y
		na += x * x
		nb += y * y
	}

	if na == 0 || nb == 0 {
		return 0
	}
	return float32(dot / (math.Sqrt(na) * math.Sqrt(nb)))
}

// Normalize devolve uma copia do vetor com comprimento 1.
//
// A copia e proposital: o vetor de origem costuma vir do espaco de trabalho
// da rede, que sera reaproveitado no proximo rosto.
func Normalize(v []float32) ([]float32, error) {
	if len(v) == 0 {
		return nil, fmt.Errorf("index: vetor vazio")
	}

	var soma float64
	for _, x := range v {
		soma += float64(x) * float64(x)
	}
	if soma == 0 {
		return nil, fmt.Errorf("index: vetor de comprimento zero nao tem direcao")
	}
	if math.IsNaN(soma) || math.IsInf(soma, 0) {
		return nil, fmt.Errorf("index: vetor contem NaN ou infinito")
	}

	inv := 1 / math.Sqrt(soma)
	out := make([]float32, len(v))
	for i, x := range v {
		out[i] = float32(float64(x) * inv)
	}
	return out, nil
}

// Match e um resultado de busca.
type Match struct {
	ID    string
	Score float32
}

// Sugestao traz ordens de grandeza para o limiar de decisao.
//
// NAO sao valores para usar direto. O limiar certo depende do modelo, da
// qualidade das fotos de cadastro e das condicoes do local -- e do custo de
// cada tipo de erro. Em controle de acesso, um falso positivo libera a porta
// para a pessoa errada; num acervo de fotos, apenas sugere uma etiqueta
// errada que alguem corrige.
//
// Meca com as suas proprias imagens antes de escolher.
const (
	// SugestaoRigorosa erra pouco para o lado de aceitar quem nao devia, e
	// em troca rejeita mais gente legitima.
	SugestaoRigorosa = 0.50

	// SugestaoTipica e o ponto de partida mais comum na literatura.
	SugestaoTipica = 0.38

	// SugestaoPermissiva aceita mais variacao de condicao, ao custo de mais
	// falso positivo.
	SugestaoPermissiva = 0.28
)

// Index guarda os vetores cadastrados.
//
// Depois de montado, pode ser consultado por varias goroutines ao mesmo
// tempo. Add e Remove NAO sao seguros para uso concorrente: se o cadastro
// mudar enquanto ha consultas correndo, sincronize por fora.
type Index struct {
	dim int

	// vetores guarda todos os vetores normalizados, um atras do outro, em
	// linhas de dim. Contiguo de proposito: e o formato que o produto
	// matriz-vetor da busca espera, sem copia nem rearranjo.
	vetores []float32

	// dono[i] e o indice em ids da identidade do vetor i.
	dono []int32

	// ids sao as identidades distintas, na ordem em que apareceram.
	ids []string

	// posicao acha a identidade sem varrer ids.
	posicao map[string]int32
}

// New cria um indice vazio para vetores de dim dimensoes.
//
// O SFace produz 128; o ArcFace, 512. O valor certo e o do modelo que voce
// usa -- consulte a forma da saida.
func New(dim int) (*Index, error) {
	if dim <= 0 {
		return nil, fmt.Errorf("index: dimensao invalida (%d)", dim)
	}
	return &Index{dim: dim, posicao: make(map[string]int32)}, nil
}

// Dim devolve o numero de dimensoes dos vetores.
func (ix *Index) Dim() int { return ix.dim }

// Len devolve quantos VETORES estao cadastrados.
func (ix *Index) Len() int { return len(ix.dono) }

// Identidades devolve quantas PESSOAS distintas estao cadastradas.
func (ix *Index) Identidades() int { return len(ix.ids) }

// IDs devolve as identidades cadastradas, na ordem em que apareceram.
func (ix *Index) IDs() []string { return append([]string(nil), ix.ids...) }

// Add cadastra mais um vetor para uma identidade.
//
// A mesma identidade pode receber varios vetores, e deve: uma amostra so de
// cadastro rende resultado ruim em condicao real. O vetor e normalizado e
// copiado.
func (ix *Index) Add(id string, v []float32) error {
	if id == "" {
		return fmt.Errorf("index: identidade vazia")
	}
	if len(v) != ix.dim {
		return fmt.Errorf("index: vetor de %d dimensoes num indice de %d", len(v), ix.dim)
	}

	norm, err := Normalize(v)
	if err != nil {
		return err
	}

	p, existe := ix.posicao[id]
	if !existe {
		p = int32(len(ix.ids))
		ix.ids = append(ix.ids, id)
		ix.posicao[id] = p
	}

	ix.vetores = append(ix.vetores, norm...)
	ix.dono = append(ix.dono, p)
	return nil
}

// Remove apaga uma identidade e todos os vetores dela, devolvendo quantos
// foram removidos.
//
// Existe porque precisa existir. Dado biometrico e dado pessoal sensivel, e
// o direito de eliminacao so vale se houver como exerce-lo de verdade -- nao
// marcando como inativo, apagando.
func (ix *Index) Remove(id string) int {
	p, existe := ix.posicao[id]
	if !existe {
		return 0
	}

	// Compacta vetores e donos, pulando os da identidade removida.
	novoVet := ix.vetores[:0]
	novoDono := ix.dono[:0]
	removidos := 0

	for i, d := range ix.dono {
		if d == p {
			removidos++
			continue
		}
		novoVet = append(novoVet, ix.vetores[i*ix.dim:(i+1)*ix.dim]...)
		novoDono = append(novoDono, d)
	}
	ix.vetores = novoVet
	ix.dono = novoDono

	// Tira a identidade da lista e reindexa as que vieram depois dela.
	ix.ids = append(ix.ids[:p], ix.ids[p+1:]...)
	delete(ix.posicao, id)
	for i := range ix.dono {
		if ix.dono[i] > p {
			ix.dono[i]--
		}
	}
	for nome, q := range ix.posicao {
		if q > p {
			ix.posicao[nome] = q - 1
		}
	}

	return removidos
}

// Search devolve a identidade mais parecida com o vetor dado.
//
// O segundo retorno e falso quando o indice esta vazio. A decisao de aceitar
// ou nao o resultado e de quem chama: compare Score com o seu limiar.
func (ix *Index) Search(v []float32) (Match, bool, error) {
	res, err := ix.SearchK(v, 1)
	if err != nil {
		return Match{}, false, err
	}
	if len(res) == 0 {
		return Match{}, false, nil
	}
	return res[0], true, nil
}

// SearchK devolve as k identidades mais parecidas, da melhor para a pior.
//
// Devolve IDENTIDADES, nao vetores: se uma pessoa tem cinco amostras
// cadastradas, ela aparece uma vez so, com a melhor das cinco pontuacoes.
// Sem isso, um k de 3 poderia devolver a mesma pessoa tres vezes.
func (ix *Index) SearchK(v []float32, k int) ([]Match, error) {
	if k <= 0 {
		return nil, fmt.Errorf("index: k invalido (%d)", k)
	}
	if len(v) != ix.dim {
		return nil, fmt.Errorf("index: consulta de %d dimensoes num indice de %d", len(v), ix.dim)
	}
	if len(ix.dono) == 0 {
		return nil, nil
	}

	consulta, err := Normalize(v)
	if err != nil {
		return nil, err
	}

	// Com tudo normalizado, o cosseno contra cada vetor cadastrado e o
	// produto escalar. Ver score.go para por que este pacote tem o proprio
	// produto matriz-vetor em vez de chamar o kernel.
	n := len(ix.dono)
	pontos := make([]float32, n)
	pontuar(ix.vetores, consulta, pontos, n, ix.dim)

	// Guarda a melhor pontuacao de cada identidade.
	melhor := make([]float32, len(ix.ids))
	visto := make([]bool, len(ix.ids))
	for i, p := range pontos {
		d := ix.dono[i]
		if !visto[d] || p > melhor[d] {
			melhor[d] = p
			visto[d] = true
		}
	}

	return ix.melhores(melhor, visto, k), nil
}

// melhorQue ordena dois resultados: pontuacao maior primeiro, empate
// desfeito pelo nome.
//
// O desempate pelo nome nao e enfeite: sem ele, duas identidades com a mesma
// pontuacao sairiam na ordem em que o cadastro foi montado, e a resposta
// deixaria de ser reproduzivel.
func melhorQue(a, b Match) bool {
	if a.Score != b.Score {
		return a.Score > b.Score
	}
	return a.ID < b.ID
}

// melhores escolhe os k primeiros sem ordenar o cadastro inteiro.
//
// Ordenar tudo para devolver um resultado era o gargalo real da busca: com
// 100 mil identidades, escolher a melhor custava uma ordenacao de 100 mil
// elementos. Os caminhos abaixo evitam isso.
func (ix *Index) melhores(melhor []float32, visto []bool, k int) []Match {
	// k == 1 e o caso de longe mais comum -- "de quem e este rosto". Uma
	// varredura basta.
	if k == 1 {
		var top Match
		achou := false
		for i, ok := range visto {
			if !ok {
				continue
			}
			c := Match{ID: ix.ids[i], Score: melhor[i]}
			if !achou || melhorQue(c, top) {
				top, achou = c, true
			}
		}
		if !achou {
			return nil
		}
		return []Match{top}
	}

	// Para k pequeno, insercao num topo limitado: percorre uma vez e mantem
	// so os k melhores. Custa O(identidades * k), e k aqui e um punhado.
	if k < 32 {
		top := make([]Match, 0, k)
		for i, ok := range visto {
			if !ok {
				continue
			}
			c := Match{ID: ix.ids[i], Score: melhor[i]}

			if len(top) == k && !melhorQue(c, top[len(top)-1]) {
				continue
			}
			pos := sort.Search(len(top), func(j int) bool { return melhorQue(c, top[j]) })
			if len(top) < k {
				top = append(top, Match{})
			}
			copy(top[pos+1:], top[pos:])
			top[pos] = c
		}
		return top
	}

	// k grande: ordenar tudo sai mais barato que inserir k vezes por
	// identidade.
	out := make([]Match, 0, len(ix.ids))
	for i, ok := range visto {
		if ok {
			out = append(out, Match{ID: ix.ids[i], Score: melhor[i]})
		}
	}
	sort.Slice(out, func(a, b int) bool { return melhorQue(out[a], out[b]) })
	if len(out) > k {
		out = out[:k]
	}
	return out
}
