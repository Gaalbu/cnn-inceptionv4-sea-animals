# CNN · Inception-v4 · Sea Animals

Transfer learning e fine-tuning com **Inception-v4** para classificar imagens de animais marinhos.
Projeto de CNN da disciplina **Machine Learning e Deep Learning (CC6NA 2026.2)**, Profa. Polyana Santos Fonseca Nascimento.

**Equipe Inception (vaga 1):** Alberto Acosta · Davi Corrêa · Gabriel Albuquerque

| | |
|---|---|
| Arquitetura | Inception-v4 — Szegedy, Ioffe, Vanhoucke & Alemi, *Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning*, AAAI 2017 ([arXiv:1602.07261](https://arxiv.org/abs/1602.07261)) |
| Implementação | PyTorch + [`timm`](https://huggingface.co/timm/inception_v4.tf_in1k) (`inception_v4.tf_in1k`, pesos ImageNet-1k) |
| Dataset | [Sea Animals Image Dataset 🌊 (Kaggle)](https://www.kaggle.com/datasets/vencerlanz09/sea-animals-image-dataste) |
| Problema | Classificação multiclasse de imagens de animais marinhos. Target: tipo de animal |
| Validação do dataset | ✅ Validado pela professora |

## Cronograma

| Data | Marco |
|---|---|
| **23 e 25/11** | **Defesa intermediária.** O notebook precisa ter: arquitetura pesquisada, dados analisados e preparados, teste reservado, métricas e divisão definidas, base congelada treinada com curvas e resultados na validação, análise visual preliminar e **as duas estratégias de fine-tuning planejadas** |
| **30/11** | Submissão no Classroom (notebook, slides, modelo salvo) |
| 30/11 e 2/12 | Apresentação oral (máx. 12 min, todos falam) |

## Divisão de tarefas

Todos precisam conseguir explicar **todas** as partes na arguição. A nota da defesa e da apresentação é individual.

### Parte 1 · Pesquisa da arquitetura e narrativa — *(responsável: a definir)*
- [ ] Seção inicial do notebook: integrantes, **declaração de uso de IA Generativa** (ferramenta, finalidade, extensão da contribuição, como foi validado), dataset e justificativa da escolha
- [ ] Inception-v4: autores, ano, referência, **diagrama** com explicação dos blocos (stem, Inception-A/B/C, Reduction-A/B)
- [ ] Principal inovação em relação a uma CNN convencional (módulos Inception: convoluções paralelas de tamanhos diferentes, fatoração de convoluções, 1×1 para reduzir dimensão)
- [ ] Diferencial da **v4** em relação à v3 (stem reformulado, blocos uniformes, reduções dedicadas) e por que não é a Inception-ResNet
- [ ] Referências completas
- [ ] Montar o **notebook final único** juntando as partes (todas as células executando em ordem, com saídas)
- [ ] Slides (narrativa, não código) e roteiro da apresentação de 12 min

### Parte 2 · Dados e base congelada — **Gabriel** ✅ código pronto
Notebook: [`notebooks/02_dados_e_base_congelada.ipynb`](notebooks/02_dados_e_base_congelada.ipynb) · [Abrir no Colab](https://colab.research.google.com/github/Gaalbu/cnn-inceptionv4-sea-animals/blob/main/notebooks/02_dados_e_base_congelada.ipynb)
- [x] Inventário e verificação das imagens (ilegíveis, modos de cor, tamanhos, proporções)
- [x] Distribuição das classes e amostras visuais
- [x] Duplicatas exatas (MD5) e rótulos conflitantes, para evitar vazamento
- [x] Divisão estratificada 70/15/15, teste separado primeiro → `artifacts/split.csv`
- [x] Pré-processamento dos pesos (299×299, RGB, [−1, 1], resize 342 + recorte central) e aumento de dados
- [x] Métricas (F1-macro como regra de seleção, acurácia balanceada, acurácia, por classe)
- [x] Inception-v4 pré-treinada, classificador novo, base congelada (BatchNorm congelado)
- [x] Treino com early stopping, curvas de loss e F1, matriz de confusão, principais confusões
- [x] Análise visual preliminar de acertos e erros
- [x] Modelo de referência salvo e conferido → `artifacts/modelo_base_congelada.pt`
- [ ] **Rodar no Colab com GPU** e salvar o notebook com as saídas
- [ ] Escrever a análise das curvas e dos erros com os resultados reais
- [ ] Subir `modelo_base_congelada.pt` no Drive da equipe e colocar o link aqui
- [ ] Fechar com a equipe as **19 classes** (ver pendências)

### Parte 3 · Fine-tuning, avaliação e explicabilidade — *(responsável: a definir)*
Partir de `artifacts/split.csv` e de **cópias** de `artifacts/modelo_base_congelada.pt`, reaproveitando `set_train_mode`, `run_epoch`, `fit`, `compute_metrics` e `load_checkpoint` do notebook da parte 2 (mesmos dados, métricas e regra de seleção).
- [ ] **Planejar A e B até a defesa** (o que muda, o que fica fixo, por quê)
- [ ] Executar A e B; registrar camadas descongeladas, lr, épocas, critério de parada, tempo
- [ ] Tabela comparativa das três condições (métricas, parâmetros treináveis, épocas, tempo), curvas e matrizes de confusão
- [ ] Análise visual: mesmas imagens nos três modelos (erros corrigidos, persistentes, novos), usando `val_preds_base_congelada.csv`
- [ ] **Grad-CAM** nas mesmas imagens nos três modelos (camada `model.features[-1]`, último bloco Inception-C)
- [ ] Escolher o modelo recomendado **pela validação**; só então avaliar os três no **teste**
- [ ] Salvar o modelo escolhido, carregar sem retreinar e prever uma **imagem da internet** (mostrar fonte, classe esperada, previsão e score)

**Sugestão de estratégias** (a equipe decide):

| | Estratégia A | Estratégia B |
|---|---|---|
| Ideia | Descongelar só o topo, em etapa única | Descongelar mais da rede, progressivamente, com lr menor na base |
| Camadas | `features[18:]` (Reduction-B + 3× Inception-C) + classificador | Etapa 1: topo como em A; etapa 2: também `features[11:18]` (7× Inception-B) |
| Learning rate | 1e-4 | Classificador 1e-4, blocos descongelados 1e-5 |
| Fixo | Split, aumento de dados, batch 32, AdamW, early stopping (paciência 4) no F1-macro da validação, semente |

Estrutura da `features` no `timm`: `0–2` stem, `3–5` Mixed, `6–9` Inception-A, `10` Reduction-A, `11–17` Inception-B, `18` Reduction-B, `19–21` Inception-C.

## Pendências para a equipe decidir

1. **19 ou 23 classes?** A planilha validada diz 19 classes, mas o dataset original tem 23 pastas. Definir quais saem e por quê, e preencher `CLASSES_EXCLUIR` no notebook da parte 2.
2. **"Foco em caranguejos / espécie de caranguejo".** O dataset tem uma única classe *Crabs* e não separa espécies, então o target real é o **tipo de animal marinho**. Ajustar a descrição do problema para não haver questionamento na defesa.
3. Responsáveis das partes 1 e 3.

## Como rodar

1. Abrir o notebook no Colab (link acima) e escolher **Ambiente de execução → GPU (T4)**.
2. Executar tudo. O dataset é baixado com `kagglehub` (pode pedir login do Kaggle). Para salvar os artefatos no Drive, usar `USE_DRIVE = True`.
3. Arquivos gerados em `artifacts/`. Os CSV/JSON pequenos podem ir para o git; o `.pt` (~165 MB) não (ver `.gitignore`).

Dependências: `requirements.txt`.

## Estrutura

```
notebooks/
  02_dados_e_base_congelada.ipynb   # parte 2 (etapas 2 e 3 da lauda)
artifacts/                          # gerado pelo notebook (modelo .pt fora do git)
```
