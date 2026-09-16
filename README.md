# Classificador Iris com Rede Neural Feedforward

Este projeto apresenta uma rede neural feedforward desenvolvida manualmente em Python para classificar flores do dataset Iris. A implementação utiliza **NumPy** nas operações matemáticas e **Pandas** na leitura, organização e análise dos dados. O objetivo foi compreender as etapas fundamentais de uma rede neural, sem utilizar frameworks prontos de deep learning ou classificadores já implementados.

## Processo de decisão do grupo

As principais decisões do projeto foram tomadas de forma democrática. Antes da implementação, os integrantes discutiram e validaram em conjunto quais linguagem, biblioteca, dataset, arquitetura e configurações seriam utilizados. Dessa forma, as escolhas não foram definidas por apenas uma pessoa, mas resultaram de um acordo entre os participantes, considerando o objetivo da atividade, o nível de complexidade adequado e a possibilidade de todos compreenderem e explicarem o código.

As opções de hiperparâmetros foram pesquisadas em materiais técnicos, exemplos didáticos e referências disponíveis na internet. A equipe utilizou essas referências como ponto de partida e escolheu uma configuração simples e compatível com o tamanho do dataset. Portanto, os valores adotados não pretendem ser universais ou necessariamente os melhores possíveis; eles foram selecionados para produzir um experimento compreensível, reproduzível e adequado à proposta da disciplina.

## Escolha da linguagem e das bibliotecas

O grupo escolheu **Python** porque a linguagem possui sintaxe acessível, ampla utilização em inteligência artificial e boas bibliotecas para manipulação numérica e tabular. Essa escolha também facilita que todos os integrantes acompanhem a implementação e executem o projeto em seus próprios ambientes.

O **NumPy** foi escolhido para implementar manualmente a rede neural, incluindo a propagação direta, a retropropagação, os gradientes e as atualizações dos pesos. O **Pandas** foi escolhido para carregar o dataset, organizar as colunas, converter os rótulos e salvar os resultados. A decisão de não utilizar TensorFlow, PyTorch ou um classificador pronto foi intencional: o foco era demonstrar o funcionamento interno de uma rede feedforward.

## Escolha do dataset Iris

O dataset Iris foi escolhido por consenso do grupo porque é pequeno, conhecido e adequado para um primeiro projeto de classificação supervisionada. Ele possui 150 amostras, quatro características numéricas e três classes de flores. Seu tamanho permite realizar vários testes rapidamente e facilita a análise dos resultados.

Outro fator considerado foi a simplicidade das variáveis. As medidas de sépala e pétala podem ser utilizadas diretamente como entradas numéricas, sem exigir processamento complexo de imagens ou texto. Isso permite que o grupo concentre o trabalho na construção e no treinamento da rede neural.

O dataset é dividido em 80% para treinamento e 20% para teste. A divisão é estratificada para manter uma proporção semelhante das três espécies nos dois conjuntos. A semente aleatória 42 foi definida para que todos os integrantes obtenham a mesma divisão e consigam reproduzir os resultados.

## Preparação dos dados

As classes são convertidas para valores numéricos: `0` para *Iris-setosa*, `1` para *Iris-versicolor* e `2` para *Iris-virginica*. Em seguida, as quatro características são padronizadas usando a média e o desvio padrão calculados somente no conjunto de treinamento.

Essa decisão foi tomada porque as variáveis podem apresentar escalas e dispersões diferentes. A padronização ajuda o algoritmo de otimização a trabalhar com entradas mais equilibradas. O uso exclusivo dos dados de treinamento no cálculo da média e do desvio padrão evita que informações do conjunto de teste influenciem o treinamento.

## Escolha da arquitetura da rede

A arquitetura definida pelo grupo foi:

```text
4 neurônios de entrada → 8 neurônios → 8 neurônios → 3 neurônios de saída
```

A camada de entrada possui quatro neurônios porque o dataset apresenta quatro características. Foram escolhidas duas camadas ocultas com oito neurônios cada para permitir que a rede aprenda relações não lineares sem se tornar excessivamente complexa.

Uma rede maior não foi considerada necessária, pois o Iris possui poucas amostras. Um número muito alto de camadas ou neurônios poderia aumentar o risco de sobreajuste e dificultar a explicação do projeto. A arquitetura escolhida representa um equilíbrio entre capacidade de aprendizado, simplicidade e facilidade de apresentação.

As camadas ocultas utilizam a função de ativação **ReLU**, escolhida por ser simples, eficiente e comum em redes neurais. A camada de saída utiliza **Softmax**, pois o problema possui três classes mutuamente exclusivas e a função transforma as saídas em probabilidades.

## Escolha dos hiperparâmetros e treinamento

Os hiperparâmetros foram definidos após consulta a exemplos e materiais disponíveis na internet. O grupo optou pelos seguintes valores:

| Hiperparâmetro | Valor | Justificativa |
|---|---:|---|
| Função de perda | Entropia cruzada categórica | Adequada para classificação multiclasse com saída Softmax. |
| Otimizador | Adam | Geralmente apresenta boa convergência com pouca configuração manual. |
| Taxa de aprendizado | `0.01` | Valor inicial simples e compatível com um dataset pequeno e padronizado. |
| Número de épocas | `500` | Permite que a rede tenha tempo suficiente para convergir. |
| Tamanho do lote | Dataset completo | Simplifica o treinamento e é viável porque o Iris possui apenas 150 amostras. |
| Inicialização | He | Compatível com as camadas que utilizam ReLU. |

A função de perda mede a diferença entre as probabilidades previstas e a classe correta. O Adam foi escolhido porque adapta as atualizações dos pesos e costuma ser uma opção prática para experimentos iniciais. A taxa de aprendizado e o número de épocas foram adotados como valores de referência, e não como resultado de uma busca exaustiva.

Durante o treinamento, a perda e a acurácia são registradas a cada época. Esse acompanhamento permite verificar se a perda diminui e se a acurácia melhora, indicando convergência do modelo. O histórico é salvo em `results/training_history.csv`.

## Teste e avaliação

Depois do treinamento, os pesos não são mais ajustados e o modelo é avaliado no conjunto de teste separado anteriormente. Essa separação permite medir o comportamento do modelo em dados que não foram utilizados para calcular os gradientes.

As métricas escolhidas foram acurácia, precisão, recall e F1-score. A acurácia mostra a proporção geral de previsões corretas. A precisão, o recall e o F1-score ajudam a analisar o desempenho individual em cada espécie. Também é gerada uma matriz de confusão para mostrar quais classes foram classificadas corretamente e quais foram confundidas.

Como o conjunto de teste possui apenas 30 amostras, cada erro representa aproximadamente 3,33 pontos percentuais da acurácia. Por esse motivo, o grupo decidiu não analisar somente a acurácia e incluiu também a matriz de confusão e as métricas por classe.

No experimento realizado, a rede alcançou aproximadamente **98,33% de acurácia no treinamento** e **96,67% no teste**. Esses valores indicam um bom desempenho para a configuração escolhida, embora não representem uma garantia de desempenho em outros datasets ou em divisões diferentes dos dados.

## Como executar

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train.py
```

Os resultados serão salvos no diretório `results/`:

```text
results/
├── confusion_matrix.csv
├── metrics.json
└── training_history.csv
```

## Estrutura do projeto

```text
.
├── data/
│   └── iris.data
├── results/
│   ├── confusion_matrix.csv
│   ├── metrics.json
│   └── training_history.csv
├── train.py
├── requirements.txt
└── README.md
```

## Conclusão

O projeto foi construído a partir de decisões discutidas e aprovadas coletivamente. A escolha do Python, do NumPy, do Pandas e do Iris considerou a acessibilidade da implementação e a capacidade de todos os integrantes compreenderem o processo. A arquitetura e os hiperparâmetros foram escolhidos com base no tamanho do dataset, no objetivo didático e em referências pesquisadas pelo grupo. O resultado é um modelo simples, reproduzível e suficiente para demonstrar a criação, o treinamento e a avaliação de uma rede neural feedforward.

## Referências

[1]: https://archive.ics.uci.edu/ml/datasets/iris "UCI Machine Learning Repository — Iris Dataset"
