package extract

import (
	"fmt"
	"regexp"
	"strconv"
	"strings"
	"time"
)

// meses mapeia o nome do mes em portugues, minusculo e sem acento, para o
// numero do mes -- normalizaMes cuida de tirar o acento antes da consulta.
var meses = map[string]int{
	"janeiro": 1, "fevereiro": 2, "marco": 3, "abril": 4,
	"maio": 5, "junho": 6, "julho": 7, "agosto": 8, "setembro": 9,
	"outubro": 10, "novembro": 11, "dezembro": 12,
}

var dataNumericaPattern = regexp.MustCompile(`\b(\d{1,2})[/-](\d{1,2})[/-](\d{2}|\d{4})\b`)
var dataExtensoPattern = regexp.MustCompile(`(?i)\b(\d{1,2})\s+de\s+(\p{L}+)\s+de\s+(\d{4})\b`)

// ParseDateBR le uma data no padrao brasileiro: numerica (DD/MM/AAAA,
// DD/MM/AA, DD-MM-AAAA) ou por extenso ("14 de setembro de 2026") -- os
// dois formatos que aparecem em nota fiscal, boleto e ordem de compra.
//
// Ano de 2 digitos usa o pivo comum: >= 70 vira 1900+, menor que isso vira
// 2000+. Documento fiscal brasileiro nao data de antes de 1970; se algum
// dia precisar, o pivo muda aqui.
func ParseDateBR(s string) (time.Time, error) {
	s = strings.TrimSpace(s)

	if m := dataExtensoPattern.FindStringSubmatch(s); m != nil {
		dia, _ := strconv.Atoi(m[1])
		mes, ok := meses[normalizaMes(m[2])]
		if !ok {
			return time.Time{}, fmt.Errorf("extract: mes desconhecido em %q: %q", s, m[2])
		}
		ano, _ := strconv.Atoi(m[3])
		return construirData(ano, mes, dia, s)
	}

	if m := dataNumericaPattern.FindStringSubmatch(s); m != nil {
		dia, _ := strconv.Atoi(m[1])
		mes, _ := strconv.Atoi(m[2])
		ano, _ := strconv.Atoi(m[3])
		if len(m[3]) == 2 {
			if ano >= 70 {
				ano += 1900
			} else {
				ano += 2000
			}
		}
		return construirData(ano, mes, dia, s)
	}

	return time.Time{}, fmt.Errorf("extract: %q nao parece uma data", s)
}

// normalizaMes tira o acento de "marco" -- o unico nome de mes em
// portugues que precisa disso -- e poe em minusculo.
func normalizaMes(s string) string {
	s = strings.ToLower(s)
	return strings.NewReplacer("ã", "a", "ç", "c").Replace(s)
}

// construirData monta a data e confere que ela e valida de verdade --
// time.Date normaliza datas impossiveis (31 de abril vira 1 de maio) em
// vez de dar erro, o que esconderia justamente o tipo de erro de
// reconhecimento mais importante de pegar aqui.
func construirData(ano, mes, dia int, original string) (time.Time, error) {
	if mes < 1 || mes > 12 {
		return time.Time{}, fmt.Errorf("extract: mes invalido em %q: %d", original, mes)
	}
	t := time.Date(ano, time.Month(mes), dia, 0, 0, 0, 0, time.UTC)
	if t.Year() != ano || int(t.Month()) != mes || t.Day() != dia {
		return time.Time{}, fmt.Errorf("extract: data invalida: %q", original)
	}
	return t, nil
}
