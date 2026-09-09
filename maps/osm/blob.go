package osm

import (
	"bytes"
	"compress/zlib"
	"encoding/binary"
	"fmt"
	"io"

	"github.com/iafrotamacedo-cloud/era/internal/protowire"
)

// Limites do formato. A especificacao do .osm.pbf recomenda cabecalho abaixo
// de 64 KiB e corpo abaixo de 32 MiB; os numeros aqui tem folga sobre isso.
//
// Existem por dois motivos. Um arquivo corrompido pode trazer um tamanho
// absurdo, e sem limite o leitor tentaria alocar gigabytes antes de descobrir
// que o dado nao presta. E um arquivo que nao e .osm.pbf -- um video, um dump
// -- seria lido como se fosse, ate a memoria acabar.
//
// int64 explicito: sem ele a constante ficaria sem tipo e, passada a
// fmt.Errorf, assumiria int, que em plataforma de 32 bits nao comporta estes
// valores. Compila em amd64 e quebra em arm.
const (
	maxCabecalho int64 = 128 << 10 // 128 KiB
	maxCorpo     int64 = 64 << 20  // 64 MiB
)

// tipoBloco distingue os dois tipos de bloco que um .osm.pbf contem.
const (
	blocoCabecalho = "OSMHeader"
	blocoDados     = "OSMData"
)

// blob e um bloco lido do arquivo, ainda comprimido.
type blob struct {
	tipo  string
	corpo []byte // o Blob serializado, ainda por descomprimir
}

// lerBlob le o proximo bloco do arquivo.
//
// A estrutura de cada bloco:
//
//	[4 bytes big-endian: tamanho do BlobHeader]
//	[BlobHeader: tipo e tamanho do corpo]
//	[Blob: o corpo]
//
// Devolve io.EOF -- limpo, nao ErrFormato -- quando o arquivo acaba
// exatamente na fronteira de um bloco, que e como um arquivo inteiro termina.
//
// O corpo e alocado por bloco, e nao reaproveitado, porque ele viaja para
// outra goroutine: reusar o buffer aqui faria o proximo bloco sobrescrever o
// que a goroutine anterior ainda esta descomprimindo. Sao milhares de
// alocacoes por arquivo, nao milhoes, e cada uma e insignificante perto do
// inflate que vem depois.
func lerBlob(r io.Reader) (blob, error) {
	var tamanho [4]byte
	if _, err := io.ReadFull(r, tamanho[:]); err != nil {
		if err == io.EOF {
			return blob{}, io.EOF // fim limpo
		}
		return blob{}, fmt.Errorf("%w: arquivo termina no meio de um tamanho de bloco", ErrFormato)
	}

	n := int64(binary.BigEndian.Uint32(tamanho[:]))
	if n <= 0 || n > maxCabecalho {
		return blob{}, fmt.Errorf("%w: cabecalho de bloco diz ter %d bytes, fora do limite de %d",
			ErrFormato, n, maxCabecalho)
	}

	cab := make([]byte, n)
	if _, err := io.ReadFull(r, cab); err != nil {
		return blob{}, fmt.Errorf("%w: arquivo termina no meio de um cabecalho de bloco", ErrFormato)
	}

	tipo, tamCorpo, err := lerBlobHeader(cab)
	if err != nil {
		return blob{}, err
	}

	corpo := make([]byte, tamCorpo)
	if _, err := io.ReadFull(r, corpo); err != nil {
		return blob{}, fmt.Errorf("%w: bloco %q diz ter %d bytes e o arquivo acabou antes",
			ErrFormato, tipo, tamCorpo)
	}

	return blob{tipo: tipo, corpo: corpo}, nil
}

// lerBlobHeader decodifica o cabecalho: qual e o tipo do bloco e quanto mede
// o corpo que vem a seguir.
func lerBlobHeader(buf []byte) (tipo string, tamCorpo int64, err error) {
	r := protowire.New(buf)
	for !r.Done() {
		campo, typ, err := r.Tag()
		if err != nil {
			return "", 0, fmt.Errorf("%w: cabecalho de bloco ilegivel: %v", ErrFormato, err)
		}
		switch campo {
		case 1: // type
			tipo, err = r.String()
		case 3: // datasize
			tamCorpo, err = r.Int64()
		default:
			err = r.Skip(typ)
		}
		if err != nil {
			return "", 0, fmt.Errorf("%w: cabecalho de bloco ilegivel: %v", ErrFormato, err)
		}
	}

	if tipo == "" {
		return "", 0, fmt.Errorf("%w: cabecalho de bloco sem tipo", ErrFormato)
	}
	if tamCorpo <= 0 || tamCorpo > maxCorpo {
		return "", 0, fmt.Errorf("%w: bloco %q diz ter %d bytes, fora do limite de %d",
			ErrFormato, tipo, tamCorpo, maxCorpo)
	}
	return tipo, tamCorpo, nil
}

// descomprimir extrai o conteudo util de um Blob.
//
// O corpo pode vir cru ou comprimido. Na pratica, todo extrato publicado usa
// zlib -- que esta na biblioteca padrao. Os demais algoritmos que o formato
// admite (lzma, lz4, zstd) exigiriam dependencia externa, e a ERA nao tem
// nenhuma; encontrar um deles e erro claro, nao leitura pela metade.
//
// O resultado e escrito em dest, reaproveitado entre blocos.
func descomprimir(corpo []byte, dest *[]byte) ([]byte, error) {
	r := protowire.New(corpo)

	var (
		cru      []byte
		zlibData []byte
		tamCru   int64
		outro    string
	)

	for !r.Done() {
		campo, typ, err := r.Tag()
		if err != nil {
			return nil, fmt.Errorf("%w: corpo de bloco ilegivel: %v", ErrFormato, err)
		}
		switch campo {
		case 1: // raw
			cru, err = r.Bytes()
		case 2: // raw_size
			tamCru, err = r.Int64()
		case 3: // zlib_data
			zlibData, err = r.Bytes()
		case 4:
			outro, err = "lzma", r.Skip(typ)
		case 5:
			outro, err = "bzip2", r.Skip(typ)
		case 6:
			outro, err = "lz4", r.Skip(typ)
		case 7:
			outro, err = "zstd", r.Skip(typ)
		default:
			err = r.Skip(typ)
		}
		if err != nil {
			return nil, fmt.Errorf("%w: corpo de bloco ilegivel: %v", ErrFormato, err)
		}
	}

	switch {
	case cru != nil:
		return cru, nil

	case zlibData != nil:
		if tamCru <= 0 || tamCru > maxCorpo {
			return nil, fmt.Errorf("%w: bloco comprimido declara %d bytes descomprimidos, fora do limite",
				ErrFormato, tamCru)
		}
		if int64(cap(*dest)) < tamCru {
			*dest = make([]byte, tamCru)
		}
		saida := (*dest)[:tamCru]

		zr, err := zlib.NewReader(bytes.NewReader(zlibData))
		if err != nil {
			return nil, fmt.Errorf("%w: zlib invalido: %v", ErrFormato, err)
		}
		defer zr.Close()

		if _, err := io.ReadFull(zr, saida); err != nil {
			return nil, fmt.Errorf("%w: bloco prometeu %d bytes descomprimidos e entregou menos: %v",
				ErrFormato, tamCru, err)
		}
		return saida, nil

	case outro != "":
		return nil, fmt.Errorf("%w: bloco comprimido com %s; a ERA le apenas zlib, "+
			"que e o que os extratos publicados usam", ErrNaoSuportado, outro)

	default:
		return nil, fmt.Errorf("%w: bloco sem conteudo", ErrFormato)
	}
}
