// Package faces e a API publica do motor de reconhecimento facial da ERA.
//
// Ela amarra os pacotes internos num pipeline unico:
//
//	imagem -> detect (YuNet) -> align -> embed (SFace) -> comparar / buscar
//
// Quem importa so este pacote nao precisa montar detect, align e graph na
// mao. O indice 1:N continua no pacote index — a persistencia e decisao de
// quem usa.
//
// Exemplo:
//
//	eng, err := faces.Open(faces.Config{
//	    YuNet: caminhoYuNet,
//	    SFace: caminhoSFace,
//	})
//	if err != nil { ... }
//
//	rostos, err := eng.Embed(foto)
//	if err != nil { ... }
//
//	ix, _ := index.New(eng.Dim())
//	ix.Add("maria", rostos[0].Vector)
//
//	id, err := eng.Recognize(foto, ix, index.SugestaoTipica)
package faces
