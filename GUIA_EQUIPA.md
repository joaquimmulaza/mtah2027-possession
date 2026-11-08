# Guia da equipa: do zero à submissão

Prazo: **17 de novembro de 2026**. Sem leaderboard ao vivo, sem feedback.

## ⚠️ Regra que não podem quebrar
Os termos **proíbem enviar os dados do jogo (vídeo, tracking, tagging ou extratos) para qualquer
LLM na cloud**: Claude Code, Cursor, Gemini e Grok incluídos. Usem as IAs só para **código**. Nunca
lhes peçam para abrir ou ler ficheiros em `data/`, nem colem lá output com coordenadas.
O pipeline só imprime contagens agregadas, e essas são seguras.

## 1. Correr a solução (amigo de Python)
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-lock.txt
# descompactar o pacote de dados para ./data
python -m possession.pipeline --team NOME_EQUIPA
python evaluation/validate_submission.py evaluation/results/NOME_EQUIPA.csv
python evaluation/run_evaluation.py      # só para confirmar que o pipeline está CORRETO
python -m tests.test_pipeline            # teste num jogo sintético (não usa dados reais)
```
A primeira corrida lê os 199.319 frames (1–3 min) e grava uma cache em `data/cache/`.

## 2. O que verificar com os dados reais
O pipeline foi testado num jogo **sintético**, não no real. Com os dados reais confirmem:
- a percentagem de frames com bola vista ≈ 57% (DATA.md) → leitura do ficheiro certa;
- a bola parada a ~20–25% dos frames avaliados;
- o número de posses: 5 min públicos = 17 posses, ~14 s cada. No jogo inteiro esperem ~350–450;
- na tabela do Smart Tagging, códigos como `build up` "kept" e `defending in ...` "flipped".

Se algum número estiver muito fora, o problema é de **correção** (unidades, eixos, formato), não
de afinação.

## 3. O papel de quem percebe de futebol
Segundo o email de clarificação da organização, que prevalece sobre tudo o resto:
- **Permitido:** explicar o que conta como posse no futebol, em geral. Também é permitido ver os
  dados e o vídeo do jogo **para perceber** o que conta como posse.
- **Permitido:** ver e comentar **jogos externos** (os jogos de exemplo da Metrica em
  `data/external/`) e ajudar a afinar as regras nesses jogos.
- **Proibido:** verificar o resultado do sistema contra o vídeo ou contra o jogo da competição,
  ou dar notas e avaliações humanas sobre o resultado neste jogo. Também é proibido escrever
  rótulos à mão ou corrigir frames.
- Os parâmetros estão em `possession/config.py`, afinados só em jogos externos
  (`python -m tools.external`).

## 4. Não façam isto
- **Não afinem nem calibrem nada no ground truth (nem nos 5 min públicos) nem no vídeo.** Os
  parâmetros afinam-se só com dados externos.
- Não enviem dados do jogo para serviços de IA.
- Não façam commit de `data/`. O repositório tem de ser público, mas os dados não.
- Não usem dados de eventos (Opta, StatsBomb…) deste jogo.

## 5. Entrega (Google Form, uma resposta por equipa)
1. `NOME_EQUIPA.csv`, validado com `validate_submission.py` (exatamente o ficheiro que enviam).
2. O URL do repositório público (fork deste, com `possession/`, `tests/` e `requirements-lock.txt`).
3. O documento técnico: `TECHNICAL_DOCUMENT.md` com o nome da equipa preenchido, enviado como
   `NOME_EQUIPA.md` ou convertido para PDF.

Equipa de 3 a 5 pessoas: 3 é o mínimo para a Final Pitch.

## 6. Próximas melhorias (por ordem de ganho esperado)
1. **Recuperar a bola em falta (41%)** com um detetor YOLO no vídeo tático (o repositório
   football_analysis serve aqui), treinado só com dados de **outros jogos**.
2. Afinar mais parâmetros nos jogos externos (`tools/external.py`).
3. Melhorar a deteção de livres rápidos sem tag.
