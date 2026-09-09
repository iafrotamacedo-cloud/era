// Package osm le arquivos .osm.pbf, o formato binario em que o
// OpenStreetMap distribui extratos do planeta.
//
// Le e so. Nao monta grafo, nao filtra estrada, nao sabe o que e uma rua:
// entrega no, via e relacao com suas etiquetas, e quem chama decide o que
// fazer. E o mesmo corte que o faces faz entre onnx, que le o modelo, e
// graph, que o executa -- mantem erro de leitura de arquivo e erro de
// interpretacao como problemas distintos, cada um testavel por si.
//
// # O formato
//
// Um .osm.pbf e uma fila de blocos independentes. Cada bloco vem embrulhado
// assim:
//
//	[4 bytes big-endian: tamanho do cabecalho]
//	[BlobHeader: diz o tipo -- OSMHeader ou OSMData -- e o tamanho do corpo]
//	[Blob: o corpo, quase sempre comprimido com zlib]
//
// Dentro de um Blob de dados vem um PrimitiveBlock, com ate 8.000 entidades,
// uma tabela de strings propria e as coordenadas guardadas como inteiros em
// nanograus, cada uma como diferenca para a anterior.
//
// Nada disso precisa de dependencia externa: o protobuf e lido pelo
// internal/protowire e o zlib vem da biblioteca padrao.
//
// # Escala
//
// Um extrato do Brasil tem cerca de 1,5 GB comprimido e dezenas de milhoes
// de nos. Nada aqui carrega o arquivo inteiro na memoria: Scan percorre bloco
// a bloco e chama de volta.
package osm

import (
	"errors"

	"github.com/iafrotamacedo-cloud/era/maps/geo"
)

// Tag e uma etiqueta: o par chave-valor que descreve o que uma entidade e.
//
// Tudo no OpenStreetMap e etiqueta. Uma via so vira rua porque tem
// highway=residential; a mesma geometria com waterway=river e um rio. O
// formato nao tem tipos, so pares de texto.
type Tag struct {
	Key   string
	Value string
}

// Tags sao as etiquetas de uma entidade.
//
// E uma fatia, e nao um map, de proposito. Um map por entidade seriam dezenas
// de milhoes de alocacoes num extrato do Brasil, e nao pagaria: a mediana das
// entidades tem menos de dez etiquetas, e nessa escala varrer uma fatia ganha
// de calcular hash.
//
// As strings apontam para a tabela do bloco em que a entidade foi lida, que e
// decodificada uma vez e compartilhada por todas as entidades daquele bloco.
// Sao strings comuns de Go, imutaveis e seguras de guardar.
type Tags []Tag

// Get devolve o valor de uma chave.
func (t Tags) Get(key string) (string, bool) {
	for _, tag := range t {
		if tag.Key == key {
			return tag.Value, true
		}
	}
	return "", false
}

// Has informa se a chave existe, qualquer que seja o valor.
func (t Tags) Has(key string) bool {
	_, ok := t.Get(key)
	return ok
}

// Node e um ponto: uma coordenada com identificador e etiquetas.
//
// A maioria dos nos nao tem etiqueta nenhuma -- existem apenas para dar forma
// as vias que passam por eles. Os que tem sao semaforo, farmacia, parada de
// onibus, portaria.
type Node struct {
	ID    int64
	Point geo.Point
	Tags  Tags
}

// Way e uma sequencia ordenada de nos.
//
// Uma rua, um rio, o contorno de um predio. Refs traz os identificadores dos
// nos na ordem em que se ligam; as coordenadas ficam nos nos, e e por isso
// que montar geometria a partir de um .osm.pbf exige duas passadas.
//
// Refs aponta para um buffer reaproveitado entre entidades. Se precisar
// guardar a via depois que o callback retornar, copie.
type Way struct {
	ID   int64
	Refs []int64
	Tags Tags
}

// MemberType diz o que um membro de relacao e.
type MemberType int8

// Os tres tipos de membro possiveis.
const (
	MemberNode MemberType = iota
	MemberWay
	MemberRelation
)

// Member e um participante de uma relacao, com o papel que exerce nela.
type Member struct {
	ID   int64
	Type MemberType
	Role string // "outer", "inner", "via", "from", "to"...
}

// Relation agrupa entidades que juntas significam algo.
//
// Uma rodovia inteira, os limites de um municipio, e -- o caso que interessa
// a roteirizacao -- as restricoes de conversao: "quem vem por esta via nao
// pode entrar naquela".
//
// Members aponta para um buffer reaproveitado. Copie se for guardar.
type Relation struct {
	ID      int64
	Members []Member
	Tags    Tags
}

// Header e o cabecalho do arquivo, lido antes de qualquer entidade.
type Header struct {
	// BBox e o retangulo que o extrato cobre. Zero quando o arquivo nao
	// declara -- nem todo gerador escreve.
	BBox    geo.Box
	HasBBox bool

	// RequiredFeatures sao capacidades que o leitor precisa ter para
	// entender o arquivo. Scan falha se encontrar alguma desconhecida, em
	// vez de ler pela metade e devolver um mapa com buracos.
	RequiredFeatures []string
	OptionalFeatures []string

	WritingProgram string
	Source         string
}

// Handler recebe as entidades lidas.
//
// E um struct de funcoes, e nao uma interface, por um motivo pratico: um
// campo nil quer dizer "nao decodifique isto", e o leitor pula o trabalho
// inteiro em vez de decodificar para jogar fora.
//
// Isso importa porque montar um grafo rodoviario exige duas passadas. A
// primeira le so as vias, para descobrir quais nos sao referenciados; a
// segunda le so esses nos. Num extrato do Brasil, deixar Node nil na primeira
// passada e a diferenca entre decodificar dezenas de milhoes de coordenadas
// ou nenhuma.
//
//	// Passada 1: so as vias.
//	osm.Scan(f, osm.Handler{Way: func(w osm.Way) error { ... }})
//
// Os callbacks sao sempre chamados de uma goroutine so, na ordem do arquivo,
// mesmo com a decodificacao acontecendo em paralelo. Nao precisam ser seguros
// para uso concorrente.
//
// Devolver erro interrompe a leitura, e Scan devolve esse erro. Devolver
// ErrParar interrompe sem erro.
type Handler struct {
	Header   func(Header) error
	Node     func(Node) error
	Way      func(Way) error
	Relation func(Relation) error
}

// ErrParar interrompe a leitura sem que Scan considere isso uma falha.
//
// Serve a quem ja achou o que queria: um callback que devolve ErrParar faz
// Scan retornar nil.
var ErrParar = errors.New("osm: leitura interrompida pelo handler")

// ErrFormato indica arquivo que nao e um .osm.pbf valido, ou esta corrompido.
var ErrFormato = errors.New("osm: formato invalido")

// ErrNaoSuportado indica recurso do formato que este leitor nao implementa --
// uma compressao incomum, ou uma capacidade exigida pelo arquivo.
//
// E erro, e nao aviso, de proposito: um mapa lido pela metade em silencio
// vira rota errada, e rota errada em logistica vira caminhao no lugar errado.
var ErrNaoSuportado = errors.New("osm: recurso do formato nao suportado")
