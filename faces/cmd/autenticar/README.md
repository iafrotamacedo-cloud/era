# Teste real de autenticação (webcam)

Simula o fluxo do FrotaHub **antes** de integrar: cadastro do rosto + login com
anti-spoof, liveness e SFace.

Limiares da calibração 2026-09-14 (`docs/faces-modelos.md`):
- spoof `Limiar` 0,27
- liveness `MinVar` 0,012, 15 frames
- reconhecimento cosseno ≥ 0,38

## Rodar

Na raiz do ERA:

```bash
# 1) Cadastrar seu rosto (uma vez)
python faces/cmd/autenticar/autenticar.py cadastrar

# 2) Tentar autenticar
python faces/cmd/autenticar/autenticar.py verificar
```

O template fica em `faces/cmd/autenticar/template_local.bin` (local, não versionar).

## O que acontece

1. **Cadastrar** — coleta 15 frames, exige movimento de cabeça, passa anti-spoof,
   gera vetor SFace e salva.
2. **Verificar** — repete o mesmo; compara cosseno com o template.
3. **ESC** cancela; na verificação, qualquer outra tecla tenta de novo.

## Testar spoof

Depois de cadastrar, rode `verificar` mostrando **foto sua na tela** em vez do
rosto vivo — deve negar (anti-spoof ou cosseno baixo).
