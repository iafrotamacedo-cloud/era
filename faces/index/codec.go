package index

import (
	"encoding/binary"
	"fmt"
	"hash/crc32"
	"math"
)

// Formato binario do indice.
//
//	magico     8 bytes   "ERAIDX\0\0"
//	versao     uint32
//	dim        uint32
//	identidades uint32
//	vetores    uint32
//	nomes      identidades x (uint32 tamanho + bytes UTF-8)
//	donos      vetores x uint32
//	valores    vetores x dim x float32
//	crc32      uint32    de tudo acima
//
// Tudo little-endian, explicitamente. Um indice gravado num servidor x86
// precisa ser lido num coletor ARM sem surpresa; deixar a ordem de bytes
// implicita e como esse tipo de erro nasce.
//
// O CRC nao protege contra adulteracao -- nao e essa a funcao dele. Protege
// contra o caso comum e chato: gravacao interrompida, disco com defeito,
// arquivo truncado numa copia. Sem ele, meio arquivo carregaria e produziria
// respostas erradas em silencio.
const (
	magico  = "ERAIDX\x00\x00"
	versao1 = 1

	tamCabecalho = len(magico) + 4*4 // magico + versao + dim + identidades + vetores
	tamCRC       = 4
)

// MarshalBinary serializa o indice.
func (ix *Index) MarshalBinary() ([]byte, error) {
	nVet := len(ix.dono)
	if len(ix.vetores) != nVet*ix.dim {
		return nil, fmt.Errorf("index: indice inconsistente, %d valores para %d vetores de %d dimensoes",
			len(ix.vetores), nVet, ix.dim)
	}

	tamNomes := 0
	for _, id := range ix.ids {
		tamNomes += 4 + len(id)
	}

	buf := make([]byte, 0, tamCabecalho+tamNomes+nVet*4+len(ix.vetores)*4+tamCRC)

	buf = append(buf, magico...)
	buf = binary.LittleEndian.AppendUint32(buf, versao1)
	buf = binary.LittleEndian.AppendUint32(buf, uint32(ix.dim))
	buf = binary.LittleEndian.AppendUint32(buf, uint32(len(ix.ids)))
	buf = binary.LittleEndian.AppendUint32(buf, uint32(nVet))

	for _, id := range ix.ids {
		buf = binary.LittleEndian.AppendUint32(buf, uint32(len(id)))
		buf = append(buf, id...)
	}
	for _, d := range ix.dono {
		buf = binary.LittleEndian.AppendUint32(buf, uint32(d))
	}
	for _, v := range ix.vetores {
		buf = binary.LittleEndian.AppendUint32(buf, math.Float32bits(v))
	}

	return binary.LittleEndian.AppendUint32(buf, crc32.ChecksumIEEE(buf)), nil
}

// UnmarshalBinary carrega um indice serializado, substituindo o conteudo
// atual.
//
// Cada tamanho lido do arquivo e conferido contra o que resta no buffer
// ANTES de qualquer alocacao. Sem isso, um arquivo corrompido cujo campo de
// tamanho virou lixo faria o programa tentar reservar gigabytes -- e um
// arquivo corrompido nao pode derrubar quem so quis carregar um cadastro.
func (ix *Index) UnmarshalBinary(b []byte) error {
	if len(b) < tamCabecalho+tamCRC {
		return fmt.Errorf("index: dados truncados (%d bytes, minimo %d)", len(b), tamCabecalho+tamCRC)
	}
	if string(b[:len(magico)]) != magico {
		return fmt.Errorf("index: assinatura desconhecida; o arquivo nao e um indice da ERA")
	}

	corpo := b[:len(b)-tamCRC]
	gravado := binary.LittleEndian.Uint32(b[len(b)-tamCRC:])
	if calculado := crc32.ChecksumIEEE(corpo); calculado != gravado {
		return fmt.Errorf("index: checksum nao confere (gravado %08x, calculado %08x); arquivo corrompido ou truncado",
			gravado, calculado)
	}

	p := len(magico)
	leU32 := func() uint32 {
		v := binary.LittleEndian.Uint32(corpo[p:])
		p += 4
		return v
	}

	if v := leU32(); v != versao1 {
		return fmt.Errorf("index: versao de formato %d nao suportada (esta ERA le a %d)", v, versao1)
	}

	dim := int(leU32())
	nIDs := int(leU32())
	nVet := int(leU32())

	if dim <= 0 {
		return fmt.Errorf("index: dimensao invalida (%d)", dim)
	}
	if nIDs < 0 || nVet < 0 {
		return fmt.Errorf("index: contagens invalidas (%d identidades, %d vetores)", nIDs, nVet)
	}
	if nVet > 0 && nIDs == 0 {
		return fmt.Errorf("index: %d vetores sem nenhuma identidade", nVet)
	}

	// Confere o tamanho antes de alocar qualquer coisa.
	restam := len(corpo) - p
	precisa := nVet*4 + nVet*dim*4 // donos + valores; os nomes vem alem disso
	if precisa < 0 || restam < precisa {
		return fmt.Errorf("index: cabecalho pede %d vetores de %d dimensoes, mas restam %d bytes",
			nVet, dim, restam)
	}

	ids := make([]string, nIDs)
	for i := 0; i < nIDs; i++ {
		if len(corpo)-p < 4 {
			return fmt.Errorf("index: dados acabaram no nome %d", i)
		}
		n := int(leU32())
		if n < 0 || len(corpo)-p < n {
			return fmt.Errorf("index: nome %d diz ter %d bytes, restam %d", i, n, len(corpo)-p)
		}
		ids[i] = string(corpo[p : p+n])
		p += n
	}

	if len(corpo)-p != nVet*4+nVet*dim*4 {
		return fmt.Errorf("index: sobram %d bytes depois dos nomes, esperava %d",
			len(corpo)-p, nVet*4+nVet*dim*4)
	}

	dono := make([]int32, nVet)
	for i := range dono {
		d := int32(leU32())
		if d < 0 || int(d) >= nIDs {
			return fmt.Errorf("index: vetor %d aponta para a identidade %d, que nao existe", i, d)
		}
		dono[i] = d
	}

	vetores := make([]float32, nVet*dim)
	for i := range vetores {
		vetores[i] = math.Float32frombits(leU32())
	}

	posicao := make(map[string]int32, nIDs)
	for i, id := range ids {
		if id == "" {
			return fmt.Errorf("index: identidade %d tem nome vazio", i)
		}
		if _, repetida := posicao[id]; repetida {
			return fmt.Errorf("index: identidade %q aparece duas vezes", id)
		}
		posicao[id] = int32(i)
	}

	ix.dim = dim
	ix.ids = ids
	ix.dono = dono
	ix.vetores = vetores
	ix.posicao = posicao
	return nil
}

// Load reconstroi um indice a partir de bytes serializados.
func Load(b []byte) (*Index, error) {
	ix := &Index{}
	if err := ix.UnmarshalBinary(b); err != nil {
		return nil, err
	}
	return ix, nil
}

// crc32IEEE existe para que os testes possam refazer o checksum depois de
// alterar um campo de proposito, sem duplicar a escolha do polinomio.
func crc32IEEE(b []byte) uint32 { return crc32.ChecksumIEEE(b) }
