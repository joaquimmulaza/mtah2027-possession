# Guia de validação para quem percebe de futebol

Não precisas de saber nada de tecnologia. Precisas só de saber de futebol.

## O que o nosso sistema faz, em palavras simples
Imagina um árbitro assistente que **não vê o jogo, só vê pontos num mapa**: onde estão os 22
jogadores e onde está a bola, 25 vezes por segundo. Para cada instante ele decide uma de três
coisas:

- **a bola é do Barcelona**,
- **a bola é do Athletic**, ou
- **a bola está parada** (fora, falta, golo, intervalo de um lance).

O "árbitro" segue regras de futebol que lhe ensinámos:
1. **Ter a bola** é ter um jogador colado a ela, sem adversário igualmente perto.
2. **A posse só muda** quando o adversário **fica mesmo com a bola**: segura-a mais de meio
   segundo, ou faz um passe para um colega.
3. **Não muda a posse:** um toque, um ressalto, um corte que não fica com ninguém, um duelo
   aéreo ganho mas sem controlo.
4. **Bola parada:** a bola saiu do campo, ou toda a gente abrandou como numa falta, até alguém
   voltar a chutar a bola.

O teu papel é dizer **se estas regras dão um resultado que faz sentido no futebol**, e onde
falham.

## O que vais receber (o colega de Python prepara)
1. **Um relatório de 1 página** (`relatorio.txt`) com a posse de cada equipa, o número de posses,
   o tempo útil de jogo e os períodos de domínio.
2. **Um replay animado** (`replay.html`, abre-se com duplo clique). Mostra 30 momentos do jogo
   vistos de cima: bolinhas **azuis** (Barcelona), **vermelhas** (Athletic) e a bola **branca**.
   No topo, uma faixa colorida diz o que o sistema decidiu.

## Parte 1 — o relatório (10 minutos)
Lê e responde por palavras:
- A divisão da posse parece-te normal para um Barça–Athletic em casa?
- Cerca de 290 posses no jogo, de ~12 segundos cada, parece-te realista?
- Cerca de 59% de tempo útil (bola em jogo) parece-te normal na LaLiga?
- Há algum período de 15 minutos com um domínio que te pareça estranho?

## Parte 2 — o replay (30 a 45 minutos)
Para cada momento da lista à esquerda: carrega em **▶ Ver**. Usa **Velocidade: lenta** quando
precisares. Pensa nestas três perguntas:
1. Quando a faixa **muda de cor**, a outra equipa ficou **mesmo** com a bola? Ou foi só um toque,
   um ressalto ou um duelo?
2. A mudança acontece **na altura certa**, cedo demais ou tarde demais?
3. Quando aparece **"bola parada"**, o jogo estava mesmo parado? E o recomeço é marcado quando a
   bola é chutada?

Escreve na caixa de notas **o que viste**. Exemplos de notas úteis:
- "mudou de equipa num ressalto de cabeça; ninguém ficou com a bola"
- "marcou bola parada mas era só a equipa a circular devagar atrás"
- "o recomeço do lançamento lateral foi marcado tarde"

No fim, carrega em **Copiar todas as notas** e manda-as ao colega de Python.

**Dá mais atenção à "bola parada"**: o nosso teste de robustez mostra que é a regra mais sensível.

## ⚠️ Regras da competição: muito importante
- **Não contes acertos nem dês notas** ("acertou 20 em 30"). As regras proíbem usar avaliações
  humanas deste jogo para medir ou afinar o sistema.
- **Não indiques correções de momentos concretos** ("no minuto 23 era do Barça"). Isso é
  anotação à mão e dá desqualificação.
- **Descreve padrões**, que valem para qualquer jogo: "quando há um ressalto, o sistema muda cedo
  demais". O colega de Python transforma isso numa regra geral.
- **Não partilhes o relatório, o replay nem as notas fora da equipa**, nem os envies para o
  ChatGPT, o Claude, o Gemini ou outra IA. São dados protegidos pelos termos da competição.

## Para o colega de Python
```bash
cd ~/Documents/mtah2027/mtah
.venv/bin/python -m tools.review        # cria data/review/relatorio.txt e replay.html
.venv/bin/python -m tools.stability     # robustez: cada limiar ±25%
```
Os ficheiros ficam em `data/review/`, que está fora do git. Passa-os ao colega por AirDrop ou
pen USB, não por email nem por cloud. Cada padrão que ele descrever vira uma regra geral em
`possession/`, com a justificação futebolística escrita em `config.py`.
