# CNN · Inception-v4 · Sea Animals

Transfer learning e fine-tuning com **Inception-v4** para classificar imagens de animais marinhos.
Projeto de CNN da disciplina **Machine Learning e Deep Learning (CC6NA 2026.2)**, Profa. Polyana Santos Fonseca Nascimento.

**Equipe Inception (vaga 1):** Alberto Acosta · Davi Corrêa · Gabriel Albuquerque

| | |
|---|---|
| Arquitetura | Inception-v4 — Szegedy, Ioffe, Vanhoucke & Alemi, *Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning*, AAAI 2017 ([arXiv:1602.07261](https://arxiv.org/abs/1602.07261)) |
| Implementação | PyTorch + [`timm`](https://huggingface.co/timm/inception_v4.tf_in1k) (`inception_v4.tf_in1k`, pesos ImageNet-1k convertidos do TF-Slim oficial) |
| Dataset | [Sea Animals Image Dataset 🌊 (Kaggle)](https://www.kaggle.com/datasets/vencerlanz09/sea-animals-image-dataste) — ~13,7 mil imagens, 23 pastas |
| Problema | Classificação multiclasse de imagens. Target: **tipo de animal marinho** |
| Métrica principal | **F1-macro** na validação (regra de seleção nas três condições) |
| Validação do dataset | ✅ Validado pela professora |

## Situação atual

| Parte | Notebook | Código | Executado com GPU (saídas salvas) |
|---|---|---|---|
| 1 · Arquitetura e narrativa | [`01_equipe_dataset_arquitetura.ipynb`](notebooks/01_equipe_dataset_arquitetura.ipynb) | ✅ pronto | ⏳ falta |
| 2 · Dados e base congelada | [`02_dados_e_base_congelada.ipynb`](notebooks/02_dados_e_base_congelada.ipynb) | ✅ pronto | ⏳ falta |
| 3 · Fine-tuning, avaliação e explicabilidade | [`03_fine_tuning_e_avaliacao.ipynb`](notebooks/03_fine_tuning_e_avaliacao.ipynb) | ✅ pronto | ⏳ falta |

Os notebooks 02 e 03 foram testados de ponta a ponta em CPU com um dataset sintético pequeno (`SMOKE_TEST=1`): todas as células executam. **Ainda não há resultados reais**, porque nenhum notebook foi rodado no dataset completo.

## Cronograma

| Data | Marco |
|---|---|
| **23 e 25/11** | **Defesa intermediária.** O notebook precisa ter: arquitetura pesquisada, dados analisados e preparados, teste reservado, métricas e divisão definidas, base congelada treinada com curvas e resultados na validação, análise visual preliminar e **as duas estratégias de fine-tuning planejadas** |
| **30/11** | Submissão no Classroom (notebook, slides, modelo salvo) |
| 30/11 e 2/12 | Apresentação oral (máx. 12 min, todos falam) |

## Fluxo do projeto

```
01 · pesquisa da arquitetura (sem treino)
02 · dados ─► split.csv ─► base congelada ─► modelo_base_congelada.pt
03 · cópias do modelo base ─► estratégia A ─┐
                           └─► estratégia B ─┼─► comparação na validação ─► escolha ─► teste ─► modelo_recomendado.pt ─► imagem da internet
                 base congelada ─────────────┘
```

O conjunto de **teste é separado primeiro** (notebook 02) e só é usado **uma vez**, no notebook 03, depois que o modelo recomendado é escolhido pela validação.

---

## Parte 1 · Equipe, dataset e arquitetura — Davi

Notebook: [`notebooks/01_equipe_dataset_arquitetura.ipynb`](notebooks/01_equipe_dataset_arquitetura.ipynb)

- [x] Identificação da equipe, arquitetura, implementação e mapa dos três notebooks
- [x] Dataset: as 23 classes originais, o problema (target = tipo de animal), finalidade (catalogação de fotos de mergulho e levantamentos de biodiversidade) e **justificativa da escolha** (atende o mínimo da lauda, fontes variadas, classes parecidas entre si, cobertura parcial pelo ImageNet)
- [x] Tratamento do "foco em caranguejos" da planilha: como o dataset tem uma única classe *Crabs*, o foco vira **acompanhamento específico da classe Crabs** na análise de erros
- [x] Limitações previstas (classe *Fish* genérica, sobreposição com Sharks/Puffers/Eel/Seahorse etc.)
- [x] Inception-v4: autores, artigo, publicação, entrada, parâmetros, resultados no ImageNet
- [x] Linha do tempo da família Inception (v1 → v2 → v3 → v4 → Inception-ResNet)
- [x] **Diagrama** da arquitetura desenhado em código (stem, Inception-A/B/C, Reduction-A/B, classificador novo), com o índice de `model.features` de cada bloco, e tabela explicando cada bloco
- [x] Conferência no código: formatos de saída e parâmetros por bloco batem com o artigo (stem 35×35×384, Reduction-A 17×17×1024, Reduction-B 8×8×1536); ~metade dos parâmetros nos 7 Inception-B
- [x] Principal inovação frente a uma CNN convencional (convoluções paralelas, gargalo 1×1, fatoração, pooling global, reduções em ramos), com medição da economia de parâmetros e comparação com a VGG-16
- [x] Diferencial da v4 em relação à v3 e por que não a Inception-ResNet-v2; por que PyTorch + `timm` (o Keras não tem a v4)
- [x] Referências completas (8)
- [ ] **Declaração de uso de IA Generativa** (ferramenta, finalidade, extensão da contribuição, como foi validado) — ainda não está em nenhum notebook
- [ ] Montar o **notebook final único** juntando as partes (todas as células executando em ordem, com saídas)
- [ ] Slides (narrativa, não código) e roteiro da apresentação de 12 min

Figuras em `docs/`: `inception_v4_macro.png` (visão macro) e `inception_v4_modulo_A.png` (módulo Inception-A).

## Parte 2 · Dados e base congelada — Gabriel

Notebook: [`notebooks/02_dados_e_base_congelada.ipynb`](notebooks/02_dados_e_base_congelada.ipynb) · [Abrir no Colab](https://colab.research.google.com/github/Gaalbu/cnn-inceptionv4-sea-animals/blob/main/notebooks/02_dados_e_base_congelada.ipynb)

- [x] Inventário e verificação de cada imagem (decodificação completa, ilegíveis, modo de cor, formato real × extensão, imagens animadas)
- [x] Distribuição das classes, desbalanceamento e um exemplo por classe
- [x] Tamanhos e proporções em relação à entrada de 299×299
- [x] Duplicatas exatas (MD5) e arquivos idênticos com rótulos diferentes, para evitar vazamento
- [x] Divisão estratificada **70/15/15**, teste separado primeiro, com verificação de que não há imagem repetida entre conjuntos → `split.csv`, `classes.json`
- [x] Pré-processamento dos pesos (299×299, RGB, [−1, 1], resize 342 + recorte central; transparência sobre fundo branco) e aumento de dados só no treino
- [x] Métricas e erros mais prejudiciais (F1-macro como regra de seleção, acurácia balanceada, acurácia, por classe)
- [x] Inception-v4 pré-treinada, classificador novo (dropout 0,3 + linear), **base congelada com BatchNorm congelado**
- [x] Treino com AdamW, `ReduceLROnPlateau`, early stopping, curvas de loss e F1, matriz de confusão e principais confusões
- [x] Análise visual preliminar de acertos e erros → `val_preds_base_congelada.csv`
- [x] Modelo de referência salvo com pesos, classes e pré-processamento, e conferido ao recarregar → `modelo_base_congelada.pt`
- [x] Correção: a coluna `erro` da inspeção é sempre criada (antes dava `KeyError` quando nenhuma imagem era ilegível)
- [ ] **Rodar no Colab com GPU** (`USE_DRIVE = True`) e salvar o notebook com as saídas
- [ ] Escrever a análise das curvas e dos erros com os resultados reais
- [ ] Subir `modelo_base_congelada.pt` no Drive da equipe e colocar o link aqui
- [x] Target explícito na seção 1 e análise 19 × 23 classes na seção 2.4 (decisão atual: **23**; confirmar com a equipe)
- [x] Acompanhamento da classe *Crabs* na base congelada (seção 3.3)
- [x] No Colab, os artefatos vão para o Drive automaticamente (mesmo caminho que o notebook 03 lê)

## Parte 3 · Fine-tuning, avaliação e explicabilidade — Alberto

Notebook: [`notebooks/03_fine_tuning_e_avaliacao.ipynb`](notebooks/03_fine_tuning_e_avaliacao.ipynb) · [Abrir no Colab](https://colab.research.google.com/github/Gaalbu/cnn-inceptionv4-sea-animals/blob/main/notebooks/03_fine_tuning_e_avaliacao.ipynb)

Parte de `split.csv` e de **cópias** de `modelo_base_congelada.pt`, com as mesmas funções da parte 2 (`set_train_mode`, `run_epoch`, `fit`, `compute_metrics`, `load_checkpoint`). Antes de treinar, confere que o modelo base carregado reproduz as previsões de validação salvas na parte 2.

- [x] **Planejamento de A e B** (seção 2): o que fica fixo, o que muda, hipótese e riscos de cada uma, e parâmetros treináveis de cada condição
- [x] Execução de A e B, registrando camadas descongeladas, lr, épocas, critério de parada e tempo → `modelo_A.pt`, `modelo_B.pt`, `history_*.csv`, `resumo_*.json`, `val_preds_*.csv`
- [x] Tabela comparativa das três condições (métricas, parâmetros treináveis, épocas, tempo), curvas sobrepostas, 3 matrizes de confusão, F1 por classe e acompanhamento da classe *Crabs*
- [x] Análise visual nas mesmas imagens: erros corrigidos, persistentes e novos
- [x] **Grad-CAM** (implementado com hooks, sem dependência extra) nas mesmas imagens nos três modelos, camada `model.features[-1]` (último Inception-C)
- [x] Modelo recomendado escolhido **pela validação** (maior F1-macro; empate < 0,002 → o mais simples); só então os três no **teste** → `resultados_teste.csv`
- [x] `modelo_recomendado.pt` salvo, carregado sem retreinar e conferido; previsão numa imagem externa com Grad-CAM e top-5
- [ ] **Rodar no Colab com GPU** depois da parte 2 e salvar com as saídas
- [ ] Escolher a **imagem da internet** (preencher `IMG_URL`, `IMG_FONTE`, `CLASSE_ESPERADA` na seção 9)
- [ ] Escrever as análises marcadas com *(escrever/completar com os resultados reais)*

### Estratégias de fine-tuning

| | Estratégia A | Estratégia B |
|---|---|---|
| Ideia | Descongelar só o topo, em etapa única | Descongelar mais da rede, progressivamente, com lr menor nos blocos |
| Camadas | `features[18:]` (Reduction-B + 3× Inception-C) + classificador | Etapa 1 (até 5 épocas): topo como em A; etapa 2: também `features[11:18]` (7× Inception-B) |
| Learning rate | 1e-4 | Classificador 1e-4, blocos descongelados 1e-5 |
| Hipótese | Adaptar só as features de alto nível basta e é barato | Adaptar também as features intermediárias ajuda a separar animais parecidos; lr menor e descongelamento gradual protegem o pré-treino |

**Fixo nas três condições:** split, aumento de dados, batch 32, AdamW (wd 1e-4), `ReduceLROnPlateau` (fator 0,5, paciência 2), early stopping (paciência 4) no F1-macro da validação, orçamento de 20 épocas, semente 42.

Estrutura da `features` no `timm`: `0–2` stem, `3–5` Mixed, `6–9` Inception-A, `10` Reduction-A, `11–17` Inception-B, `18` Reduction-B, `19–21` Inception-C.

---

## Decisões técnicas (valem para as três condições)

| Decisão | Escolha | Por quê |
|---|---|---|
| Entrada | 299×299, RGB | Resolução nativa da Inception-v4 e dos pesos ImageNet |
| Normalização | `(x − 0,5) / 0,5` → [−1, 1] | Pré-processamento estilo Inception dos pesos originais (não a média/desvio da ResNet) |
| Avaliação | Resize do menor lado para 342 + recorte central 299 | Protocolo dos pesos (`crop_pct = 0,875`), determinístico |
| Aumento de dados | Recorte aleatório (0,6–1,0), espelhamento horizontal, leve variação de cor | Fontes e enquadramentos variados; sem rotação forte nem espelhamento vertical |
| Divisão | Holdout estratificado 70/15/15, semente 42, teste separado primeiro | Mesmos conjuntos nas três condições, sem vazamento |
| Seleção | Época de maior F1-macro na validação; modelo recomendado também pelo F1-macro da validação | Dá o mesmo peso a todas as classes, apesar do desbalanceamento |
| BatchNorm | Congelado nos blocos congelados, treina nos descongelados | Evita que a base "congelada" mude pelas estatísticas móveis |
| Ponto de partida do fine-tuning | Modelo base com o classificador já treinado | Evita que gradientes de um classificador aleatório destruam o pré-treino |

## Artefatos (`artifacts/`)

| Arquivo | Gerado por | Usado por | No git? |
|---|---|---|---|
| `split.csv`, `classes.json` | 02 | 03 | pode ir (pequeno) |
| `modelo_base_congelada.pt` | 02 | 03 | não (~165 MB, Drive) |
| `history_base_congelada.csv`, `val_preds_base_congelada.csv`, `resumo_base_congelada.json` | 02 | 03 | pode ir |
| `modelo_A.pt`, `modelo_B.pt`, `modelo_recomendado.pt` | 03 | entrega final | não (Drive) |
| `history_A/B.csv`, `val_preds_A/B.csv`, `resumo_A/B.json`, `resultados_teste.csv` | 03 | análise e slides | pode ir |

## Pendências

1. **19 ou 23 classes?** A planilha validada diz 19, mas o dataset tem 23 pastas e as ~13,7 mil imagens correspondem às 23. Decisão atual no código: **manter as 23** (argumentos na seção 2.4 do notebook 02). Confirmar com a equipe e avisar a professora da diferença; para usar 19, preencher `CLASSES_EXCLUIR`.
2. **"Foco em caranguejos / espécie de caranguejo".** Já tratado no notebook 01 (target = tipo de animal; Crabs com acompanhamento específico). Confirmar que a descrição na planilha e nos slides está coerente com isso.
3. **Declaração de uso de IA Generativa** no notebook final.
4. **Rodar tudo no Colab com GPU**, na ordem 01 → 02 → 03, e escrever as análises com os resultados reais.
5. Imagem da internet para a seção 9 do notebook 03.
6. Notebook final único, slides e roteiro da apresentação.
7. Conferir com a lauda a numeração das etapas citada nos notebooks (o 01 diz "4 em diante" para o notebook 03).

## Como rodar

1. Abrir o notebook 02 no Colab (link acima) e escolher **Ambiente de execução → GPU (T4)**.
2. No Colab o 02 monta o Drive sozinho e salva os artefatos em `MyDrive/cnn-inceptionv4-sea-animals/artifacts` (autorizar o acesso quando pedir). O dataset é baixado com `kagglehub` (pode pedir login do Kaggle).
3. Rodar o 03 no Colab: ele monta o Drive sozinho e confere se os artefatos da parte 2 estão lá antes de começar.
4. Para testar o código rapidamente em CPU, sem treino real: variável de ambiente `SMOKE_TEST=1` (e, se quiser, `SEA_DATA_DIR` e `ARTIFACTS_DIR`).

Dependências: `requirements.txt`.

## Estrutura

```
notebooks/
  01_equipe_dataset_arquitetura.ipynb  # parte 1 (etapa 1 da lauda)
  02_dados_e_base_congelada.ipynb      # parte 2 (etapas 2 e 3)
  03_fine_tuning_e_avaliacao.ipynb     # parte 3 (fine-tuning, comparação, Grad-CAM, teste, modelo final)
docs/
  inception_v4_macro.png               # diagrama macro da arquitetura
  inception_v4_modulo_A.png            # diagrama do módulo Inception-A
artifacts/                             # gerado pelos notebooks (modelos .pt fora do git)
requirements.txt
```
