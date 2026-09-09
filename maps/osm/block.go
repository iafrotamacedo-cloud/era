package osm

import (
	"fmt"

	"github.com/iafrotamacedo-cloud/era/internal/protowire"
	"github.com/iafrotamacedo-cloud/era/maps/geo"
)

// bloco e um PrimitiveBlock decodificado, com as entidades que ele trazia.
//
// Os buffers sao reaproveitados de um bloco para o outro dentro do mesmo
// trabalhador: um extrato de estado tem milhares de blocos, e realocar tudo
// a cada um daria trabalho a toa ao coletor de lixo.
type bloco struct {
	nos      []Node
	vias     []Way
	relacoes []Relation

	// Buffers compartilhados pelas entidades deste bloco. As fatias em
	// Way.Refs e Relation.Members apontam para dentro deles.
	tabela  []string
	tags    []Tag
	refs    []int64
	membros []Member

	// Area de descompressao. Fica aqui, e nao numa variavel local, para ser
	// reciclada junto com o bloco: um corpo descomprimido chega a dezenas de
	// MB, e aloca-lo por bloco seria o maior desperdicio do leitor.
	//
	// E seguro recicla-la porque nada do que sai da decodificacao aponta para
	// ela: as strings da tabela sao copias, e as etiquetas apontam para a
	// tabela.
	descomp []byte
}

func (b *bloco) limpar() {
	b.nos = b.nos[:0]
	b.vias = b.vias[:0]
	b.relacoes = b.relacoes[:0]
	b.tabela = b.tabela[:0]
	b.tags = b.tags[:0]
	b.refs = b.refs[:0]
	b.membros = b.membros[:0]
}

// Valores padrao do PrimitiveBlock quando o arquivo nao os declara.
const (
	granularidadePadrao = 100 // nanograus por unidade de coordenada
)

// decodificarBloco le um PrimitiveBlock inteiro.
//
// O que h pede decide o que e decodificado: campo nil no Handler faz a
// entidade correspondente ser pulada sem custo. Numa primeira passada que so
// quer vias, os DenseNodes -- que sao a maior parte do arquivo -- nunca sao
// tocados.
func decodificarBloco(buf []byte, h *Handler, b *bloco) error {
	b.limpar()

	r := protowire.New(buf)

	// A tabela de strings vem antes dos grupos no arquivo, mas o formato nao
	// obriga. Guardamos os grupos para decodificar depois de ter a tabela.
	var grupos [][]byte

	granularidade := int64(granularidadePadrao)
	var latOffset, lonOffset int64

	for !r.Done() {
		campo, typ, err := r.Tag()
		if err != nil {
			return fmt.Errorf("%w: bloco de dados ilegivel: %v", ErrFormato, err)
		}
		switch campo {
		case 1: // stringtable
			m, err := r.Message()
			if err != nil {
				return fmt.Errorf("%w: tabela de strings ilegivel: %v", ErrFormato, err)
			}
			if err := decodificarTabela(m, b); err != nil {
				return err
			}
		case 2: // primitivegroup
			g, err := r.Bytes()
			if err != nil {
				return fmt.Errorf("%w: grupo ilegivel: %v", ErrFormato, err)
			}
			grupos = append(grupos, g)
		case 17: // granularity
			v, err := r.Int32()
			if err != nil {
				return fmt.Errorf("%w: granularidade ilegivel: %v", ErrFormato, err)
			}
			granularidade = int64(v)
		case 19: // lat_offset
			latOffset, err = r.Int64()
		case 20: // lon_offset
			lonOffset, err = r.Int64()
		default:
			err = r.Skip(typ)
		}
		if err != nil {
			return fmt.Errorf("%w: bloco de dados ilegivel: %v", ErrFormato, err)
		}
	}

	if granularidade <= 0 {
		return fmt.Errorf("%w: granularidade %d, que zeraria toda coordenada", ErrFormato, granularidade)
	}

	esc := escala{granularidade: granularidade, latOffset: latOffset, lonOffset: lonOffset}
	for _, g := range grupos {
		if err := decodificarGrupo(g, h, b, esc); err != nil {
			return err
		}
	}
	return nil
}

// escala converte coordenada inteira em graus.
//
// O OpenStreetMap guarda coordenada como inteiro em nanograus, e nao como
// float: e exato, comprime melhor e permite guardar a diferenca para o valor
// anterior em poucos bytes. A conta de volta e sempre a mesma.
type escala struct {
	granularidade int64
	latOffset     int64
	lonOffset     int64
}

func (e escala) ponto(lat, lon int64) geo.Point {
	return geo.Point{
		Lat: 1e-9 * float64(e.latOffset+e.granularidade*lat),
		Lon: 1e-9 * float64(e.lonOffset+e.granularidade*lon),
	}
}

// decodificarTabela le a tabela de strings do bloco.
//
// E a primeira das duas compressoes que o formato usa. Cada bloco tem sua
// tabela; as entidades guardam indices para ela, nao texto. Num bloco de
// 8.000 entidades a palavra "highway" aparece uma vez, e milhares de indices
// apontam para ela.
//
// Cada string e convertida uma vez, e todas as tags do bloco apontam para o
// mesmo valor. E o que evita dezenas de milhoes de alocacoes num extrato.
func decodificarTabela(r *protowire.Reader, b *bloco) error {
	for !r.Done() {
		campo, typ, err := r.Tag()
		if err != nil {
			return fmt.Errorf("%w: tabela de strings ilegivel: %v", ErrFormato, err)
		}
		if campo != 1 {
			if err := r.Skip(typ); err != nil {
				return fmt.Errorf("%w: tabela de strings ilegivel: %v", ErrFormato, err)
			}
			continue
		}
		s, err := r.String()
		if err != nil {
			return fmt.Errorf("%w: tabela de strings ilegivel: %v", ErrFormato, err)
		}
		b.tabela = append(b.tabela, s)
	}
	return nil
}

// str devolve a string de um indice da tabela, ou erro se o indice nao existe.
func (b *bloco) str(i int) (string, error) {
	if i < 0 || i >= len(b.tabela) {
		return "", fmt.Errorf("%w: indice %d fora de uma tabela de %d strings", ErrFormato, i, len(b.tabela))
	}
	return b.tabela[i], nil
}

func decodificarGrupo(buf []byte, h *Handler, b *bloco, esc escala) error {
	r := protowire.New(buf)
	for !r.Done() {
		campo, typ, err := r.Tag()
		if err != nil {
			return fmt.Errorf("%w: grupo ilegivel: %v", ErrFormato, err)
		}

		// Um campo que ninguem quer e pulado sem ser decodificado. E o que
		// faz a primeira passada de quem so quer vias custar quase nada.
		querNo := h.Node != nil
		querVia := h.Way != nil
		querRel := h.Relation != nil

		switch {
		case campo == 1 && querNo: // nodes
			m, err := r.Message()
			if err != nil {
				return fmt.Errorf("%w: no ilegivel: %v", ErrFormato, err)
			}
			err = decodificarNo(m, b, esc)
			if err != nil {
				return err
			}
		case campo == 2 && querNo: // dense
			m, err := r.Message()
			if err != nil {
				return fmt.Errorf("%w: nos densos ilegiveis: %v", ErrFormato, err)
			}
			if err := decodificarDensos(m, b, esc); err != nil {
				return err
			}
		case campo == 3 && querVia: // ways
			m, err := r.Message()
			if err != nil {
				return fmt.Errorf("%w: via ilegivel: %v", ErrFormato, err)
			}
			if err := decodificarVia(m, b); err != nil {
				return err
			}
		case campo == 4 && querRel: // relations
			m, err := r.Message()
			if err != nil {
				return fmt.Errorf("%w: relacao ilegivel: %v", ErrFormato, err)
			}
			if err := decodificarRelacao(m, b); err != nil {
				return err
			}
		default:
			if err := r.Skip(typ); err != nil {
				return fmt.Errorf("%w: grupo ilegivel: %v", ErrFormato, err)
			}
		}
	}
	return nil
}

// decodificarNo le um no avulso.
//
// Existe por completude do formato: praticamente todo extrato publicado usa
// DenseNodes, que e muito menor. Mas o formato admite os dois, e um leitor
// que so entende um deles quebra em arquivo gerado por ferramenta menos
// comum.
func decodificarNo(r *protowire.Reader, b *bloco, esc escala) error {
	var (
		n               Node
		chaves, valores []int32
		lat, lon        int64
	)

	for !r.Done() {
		campo, typ, err := r.Tag()
		if err != nil {
			return fmt.Errorf("%w: no ilegivel: %v", ErrFormato, err)
		}
		switch campo {
		case 1:
			n.ID, err = r.SInt64()
		case 2:
			chaves, err = lerInt32Empacotados(r, typ, chaves[:0])
		case 3:
			valores, err = lerInt32Empacotados(r, typ, valores[:0])
		case 8:
			lat, err = r.SInt64()
		case 9:
			lon, err = r.SInt64()
		default:
			err = r.Skip(typ)
		}
		if err != nil {
			return fmt.Errorf("%w: no ilegivel: %v", ErrFormato, err)
		}
	}

	if len(chaves) != len(valores) {
		return fmt.Errorf("%w: no %d tem %d chaves para %d valores",
			ErrFormato, n.ID, len(chaves), len(valores))
	}

	inicio := len(b.tags)
	for i := range chaves {
		k, err := b.str(int(chaves[i]))
		if err != nil {
			return err
		}
		v, err := b.str(int(valores[i]))
		if err != nil {
			return err
		}
		b.tags = append(b.tags, Tag{Key: k, Value: v})
	}

	n.Point = esc.ponto(lat, lon)
	n.Tags = b.tags[inicio:len(b.tags):len(b.tags)]
	b.nos = append(b.nos, n)
	return nil
}

// decodificarDensos le o bloco de nos densos, que e onde mora quase todo o
// tamanho de um .osm.pbf.
//
// E a segunda compressao do formato, e a mais engenhosa. Em vez de guardar
// identificador e coordenada de cada no, guarda a *diferenca* para o no
// anterior. Nos vizinhos no arquivo estao vizinhos no mapa, entao as
// diferencas sao numeros pequenos, e varint gasta um ou dois bytes onde o
// valor absoluto gastaria oito.
//
// As etiquetas vem numa lista unica e achatada -- chave, valor, chave, valor,
// zero, chave, valor, zero -- em que o zero separa um no do proximo. Nos sem
// etiqueta nenhuma, que sao a maioria, nao ocupam nada alem do seu zero.
func decodificarDensos(r *protowire.Reader, b *bloco, esc escala) error {
	var ids, lats, lons []int64
	var chavesValores []int32

	for !r.Done() {
		campo, typ, err := r.Tag()
		if err != nil {
			return fmt.Errorf("%w: nos densos ilegiveis: %v", ErrFormato, err)
		}
		switch campo {
		case 1: // id
			ids, err = lerSInt64Empacotados(r, typ, ids)
		case 8: // lat
			lats, err = lerSInt64Empacotados(r, typ, lats)
		case 9: // lon
			lons, err = lerSInt64Empacotados(r, typ, lons)
		case 10: // keys_vals
			chavesValores, err = lerInt32Empacotados(r, typ, chavesValores)
		default:
			err = r.Skip(typ)
		}
		if err != nil {
			return fmt.Errorf("%w: nos densos ilegiveis: %v", ErrFormato, err)
		}
	}

	if len(ids) != len(lats) || len(ids) != len(lons) {
		return fmt.Errorf("%w: nos densos com %d ids, %d latitudes e %d longitudes",
			ErrFormato, len(ids), len(lats), len(lons))
	}

	// As tres listas sao diferencas acumuladas: cada valor e somado ao
	// anterior.
	var id, lat, lon int64
	kv := 0

	for i := range ids {
		id += ids[i]
		lat += lats[i]
		lon += lons[i]

		n := Node{ID: id, Point: esc.ponto(lat, lon)}

		if kv < len(chavesValores) {
			inicio := len(b.tags)
			for kv < len(chavesValores) && chavesValores[kv] != 0 {
				if kv+1 >= len(chavesValores) {
					return fmt.Errorf("%w: no %d tem chave sem valor no fim da lista", ErrFormato, id)
				}
				k, err := b.str(int(chavesValores[kv]))
				if err != nil {
					return err
				}
				v, err := b.str(int(chavesValores[kv+1]))
				if err != nil {
					return err
				}
				b.tags = append(b.tags, Tag{Key: k, Value: v})
				kv += 2
			}
			kv++ // pula o zero que separa este no do proximo
			n.Tags = b.tags[inicio:len(b.tags):len(b.tags)]
		}

		b.nos = append(b.nos, n)
	}
	return nil
}

func decodificarVia(r *protowire.Reader, b *bloco) error {
	var (
		w               Way
		chaves, valores []int32
	)

	inicioRefs := len(b.refs)

	for !r.Done() {
		campo, typ, err := r.Tag()
		if err != nil {
			return fmt.Errorf("%w: via ilegivel: %v", ErrFormato, err)
		}
		switch campo {
		case 1:
			w.ID, err = r.Int64()
		case 2:
			chaves, err = lerInt32Empacotados(r, typ, chaves[:0])
		case 3:
			valores, err = lerInt32Empacotados(r, typ, valores[:0])
		case 8: // refs, tambem por diferenca
			b.refs, err = lerSInt64Empacotados(r, typ, b.refs)
		default:
			err = r.Skip(typ)
		}
		if err != nil {
			return fmt.Errorf("%w: via ilegivel: %v", ErrFormato, err)
		}
	}

	// Desfaz as diferencas dos refs desta via.
	var ref int64
	for i := inicioRefs; i < len(b.refs); i++ {
		ref += b.refs[i]
		b.refs[i] = ref
	}
	w.Refs = b.refs[inicioRefs:len(b.refs):len(b.refs)]

	if len(chaves) != len(valores) {
		return fmt.Errorf("%w: via %d tem %d chaves para %d valores",
			ErrFormato, w.ID, len(chaves), len(valores))
	}
	inicioTags := len(b.tags)
	for i := range chaves {
		k, err := b.str(int(chaves[i]))
		if err != nil {
			return err
		}
		v, err := b.str(int(valores[i]))
		if err != nil {
			return err
		}
		b.tags = append(b.tags, Tag{Key: k, Value: v})
	}
	w.Tags = b.tags[inicioTags:len(b.tags):len(b.tags)]

	b.vias = append(b.vias, w)
	return nil
}

func decodificarRelacao(r *protowire.Reader, b *bloco) error {
	var (
		rel             Relation
		chaves, valores []int32
		papeis          []int32
		memids          []int64
		tipos           []int32
	)

	for !r.Done() {
		campo, typ, err := r.Tag()
		if err != nil {
			return fmt.Errorf("%w: relacao ilegivel: %v", ErrFormato, err)
		}
		switch campo {
		case 1:
			rel.ID, err = r.Int64()
		case 2:
			chaves, err = lerInt32Empacotados(r, typ, chaves[:0])
		case 3:
			valores, err = lerInt32Empacotados(r, typ, valores[:0])
		case 8: // roles_sid
			papeis, err = lerInt32Empacotados(r, typ, papeis[:0])
		case 9: // memids, por diferenca
			memids, err = lerSInt64Empacotados(r, typ, memids[:0])
		case 10: // types
			tipos, err = lerInt32Empacotados(r, typ, tipos[:0])
		default:
			err = r.Skip(typ)
		}
		if err != nil {
			return fmt.Errorf("%w: relacao ilegivel: %v", ErrFormato, err)
		}
	}

	if len(memids) != len(papeis) || len(memids) != len(tipos) {
		return fmt.Errorf("%w: relacao %d tem %d membros, %d papeis e %d tipos",
			ErrFormato, rel.ID, len(memids), len(papeis), len(tipos))
	}

	inicioMembros := len(b.membros)
	var memid int64
	for i := range memids {
		memid += memids[i]
		papel, err := b.str(int(papeis[i]))
		if err != nil {
			return err
		}
		if tipos[i] < 0 || tipos[i] > int32(MemberRelation) {
			return fmt.Errorf("%w: relacao %d tem membro de tipo %d, que nao existe",
				ErrFormato, rel.ID, tipos[i])
		}
		b.membros = append(b.membros, Member{
			ID:   memid,
			Type: MemberType(tipos[i]),
			Role: papel,
		})
	}
	rel.Members = b.membros[inicioMembros:len(b.membros):len(b.membros)]

	if len(chaves) != len(valores) {
		return fmt.Errorf("%w: relacao %d tem %d chaves para %d valores",
			ErrFormato, rel.ID, len(chaves), len(valores))
	}
	inicioTags := len(b.tags)
	for i := range chaves {
		k, err := b.str(int(chaves[i]))
		if err != nil {
			return err
		}
		v, err := b.str(int(valores[i]))
		if err != nil {
			return err
		}
		b.tags = append(b.tags, Tag{Key: k, Value: v})
	}
	rel.Tags = b.tags[inicioTags:len(b.tags):len(b.tags)]

	b.relacoes = append(b.relacoes, rel)
	return nil
}

// lerSInt64Empacotados e lerInt32Empacotados leem uma lista de numeros.
//
// O protobuf admite duas formas para lista de numeros: empacotada, com todos
// os valores num campo delimitado so, ou repetida, um campo por valor. As
// duas aparecem em arquivos reais -- a empacotada em tudo que e recente, a
// repetida em arquivos antigos --, e um leitor que so entende uma delas
// quebra sem aviso claro.

func lerSInt64Empacotados(r *protowire.Reader, typ protowire.Type, dest []int64) ([]int64, error) {
	if typ == protowire.Varint {
		v, err := r.SInt64()
		if err != nil {
			return dest, err
		}
		return append(dest, v), nil
	}

	m, err := r.Message()
	if err != nil {
		return dest, err
	}
	for !m.Done() {
		v, err := m.SInt64()
		if err != nil {
			return dest, err
		}
		dest = append(dest, v)
	}
	return dest, nil
}

func lerInt32Empacotados(r *protowire.Reader, typ protowire.Type, dest []int32) ([]int32, error) {
	if typ == protowire.Varint {
		v, err := r.Int32()
		if err != nil {
			return dest, err
		}
		return append(dest, v), nil
	}

	m, err := r.Message()
	if err != nil {
		return dest, err
	}
	for !m.Done() {
		v, err := m.Int32()
		if err != nil {
			return dest, err
		}
		dest = append(dest, v)
	}
	return dest, nil
}
