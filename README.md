<h1># Própria-Rede-Neural</h1>h1>
 Desenvolver, treinar e avaliar uma rede neural artificial do zero utilizando uma linguagem de programação e bibliotecas de aprendizado de máquina.
<br> 
<h2>Classificador Iris com em NumPy</h3>
<p>
Este projeto implementa uma rede neural feedforward para classificar flores do dataset público Iris em três espécies, são elas: Iris-versicolor e Iris-virginica. O modelo foi escrito do zero com Numpy. O Pandas é usado para carregar, organizar e exportar os dados e resultados. Não são usados frameworks de deep learning nem um classificador pronto.
</p>
<br>
<h3>Processo de decisão do grupo</h3>
<p>As principais decisões do projeto foram tomadas de forma democrática. Antes da implementação, os integrantes discutiram e validaram em conjunto quais linguagem, biblioteca, dataset, arquitetura e configurações seriam utilizados. Dessa forma, as escolhas não foram definidas por apenas uma pessoa, mas resultaram de um acordo entre os participantes, considerando o objetivo da atividade, o nível de complexidade adequado e a possibilidade de todos compreenderem e explicarem o código.</p>
<p>As opções de hiperparâmetros foram pesquisadas em materiais técnicos, exemplos didáticos e referências disponíveis na internet. A equipe utilizou essas referências como ponto de partida e escolheu uma configuração simples e compatível com o tamanho do dataset. Portanto, os valores adotados não pretendem ser universais ou necessariamente os melhores possíveis; eles foram selecionados para produzir um experimento compreensível, reproduzível e adequado à proposta da disciplina.</p>
<br>
<h3>Escolha dos hiperparâmetros e treinamento</h3>
<p>Os hiperparâmetros foram definidos após consulta a exemplos e materiais disponíveis na internet. O grupo optou pelos seguintes valores:</p>
| Hiperparâmetro | Valor | Justificativa |
|---|---:|---|
| Função de perda | Entropia cruzada categórica | Adequada para classificação multiclasse com saída Softmax. |
| Otimizador | Adam | Geralmente apresenta boa convergência com pouca configuração manual. |
| Taxa de aprendizado | `0.01` | Valor inicial simples e compatível com um dataset pequeno e padronizado. |
| Número de épocas | `500` | Permite que a rede tenha tempo suficiente para convergir. |
| Tamanho do lote | Dataset completo | Simplifica o treinamento e é viável porque o Iris possui apenas 150 amostras. |
| Inicialização | He | Compatível com as camadas que utilizam ReLU. |

A função de perda mede a diferença entre as probabilidades previstas e a classe correta. O Adam foi escolhido porque adapta as atualizações dos pesos e costuma ser uma opção prática para experimentos iniciais. A taxa de aprendizado e o número de épocas foram adotados como valores de referência, e não como resultado de uma busca exaustiva.

Durante o treinamento, a perda e a acurácia são registradas a cada época. Esse acompanhamento permite verificar se a perda diminui e se a acurácia melhora, indicando convergência do modelo. O histórico é salvo em `resultado/training_history.csv`.

<br>
<h3>Escolha da linguagem e das bibliotecas</h3>
<p>O grupo escolheu Python porque a linguagem possui sintaxe acessível, ampla utilização em inteligência artificial e boas bibliotecas para manipulação numérica e tabular. Essa escolha também facilita que todos os integrantes acompanhem a implementação e executem o projeto em seus próprios ambientes.</p>
<p>O NumPy foi escolhido para implementar manualmente a rede neural, incluindo a propagação direta, a retropropagação, os gradientes e as atualizações dos pesos. O Pandas foi escolhido para carregar o dataset, organizar as colunas, converter os rótulos e salvar os resultados. A decisão de não utilizar TensorFlow, PyTorch ou um classificador pronto foi intencional: o foco era demonstrar o funcionamento interno de uma rede feedforward.</p>
<br>
<h3>1. Data set e preparação</h3>
<p>O dataset foi escolhido por ser pequeno, clássico e adequado para demonstrar todas as etapas de um problema supervisionado de classificação. Cada amostra possui quatro medidas: comprimento e largura da sépala, além do comprimento e largura da pétala. Há 150 observações, distribuídas igualmente entre três classes.</p>
<br>
<p>Os dados são carregados a partir do arquivo data/iris.data, obtido do UCI Machine Learning Repository . As classes são convertidas para os índices 0, 1 e 2. A divisão entre treino e teste usa 80% para treinamento e 20% para teste, com estratificação por classe. A semente aleatória 42 torna o experimento reproduzível.
</p>
<br>
<p>Antes do treinamento, cada atributo é padronizado usando somente média e desvio padrão do conjunto de treino. Essa decisão evita vazamento de informação do teste e coloca as quatro variáveis em escalas comparáveis, o que ajuda a otimização baseada em gradiente.
</p>
<h3>2. Stack e Tecnólogia</h3>
<p>Para este trabalho começamos a estruturar os dados com Python e o tratamento com Pandas vamos prosseguir com esta linguagem mesmo. Focando em Machine Learning adicionando a biblioteca do NumPy. 
</p>

<br>
<h3>3. Criação do Modelo e Arquitetura da rede</h3>
<p>A rede possui a arquitetura 4->8->8->3. A camada de entrada recebe as quatro características do Iris. As duas camadas ocultas têm oito neurônios cada e usam ReLU, escolhida por ser simples, eficiente e capaz de representar relações não lineares. A camada de saída possui três neurônios, um por espécie, e usa Softmax para produzir probabilidades cuja soma é 1.
</p>
<br>
<p>Os pesos são inicializados com uma distribuição normal escalada pela inicialização de He. Os vieses começam em zero. Essa arquitetura é pequena de propósito: o dataset tem poucas amostras, portanto uma rede muito maior aumentaria o risco de sobreajuste sem necessidade
</p>
<br>
<h3>4. Treinamento</h3>
<p>A função de perda é a entropia cruzada categórica, apropriada para classificação multiclasse com rótulos one-hot. O otimizador implementado é o Adam, que combina momento e adaptação individual da taxa de atualização. A taxa de aprendizado é 0.01, o treinamento usa 500 épocas e cada época calcula o gradiente sobre todo o conjunto de treino. Essa configuração é adequada ao tamanho reduzido do Iris e permite acompanhar a convergência de maneira determinística.
</p>
<p>Durante cada época são registrados a perda e a acurácia de treino. O histórico completo é salvo em results/training_history.csv. A implementação inclui forward pass, retropropagação, atualização Adam e predição, todos feitos diretamente com operações vetorizadas do NumPy.
</p>
<br>
<h3>Teste e Validação</h3>
<p>Após o treinamento, o modelo é avaliado no conjunto de teste separado antes do ajuste dos pesos. O projeto calcula acurácia, precisão, recall e F1-score por classe. Também salva a matriz de confusão em results/confusion_matrix.csv. O relatório consolidado, incluindo arquitetura, hiperparâmetros, estatísticas de padronização e métricas, fica em results/metrics.json.
</p>
<p>
A acurácia deve ser interpretada junto da matriz de confusão. Como o conjunto de teste tem apenas 30 amostras, uma única previsão incorreta altera a acurácia em aproximadamente 3,33 pontos percentuais. Por isso, a matriz e as métricas por classe ajudam a mostrar quais espécies foram confundidas.
</p>
<h3>Como executar</h3>
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train.py
```
<br>
<p>Os resultados serão gerados no diretório results/. O arquivo data/iris.data já está incluído no repositório para que a execução não dependa de uma nova consulta à internet.</p>
<br>
<h3>Estrutura do Projeto</h3>
```Arquitetura do Projeto
.
├──dados_limpos
│   ├── iris_limpo.csv
│   ├── Projeto_ANN.ipynb
├──normalização_dados
│   ├── Projeto_ANN.ipynb
├──treino_e_teste
│   ├── treino_e_teste.py
├── resultado/
│   ├── confusion_matrix.csv
│   ├── metrics.json
│   └── training_history.csv
├── treino.py
└── README.md
```
<h3>Referências</h3>
* Rede Neural Perceptron Multicamadas. **Medium**. 2026. Disponível em: https://medium.com/ensina-ai/rede-neural-perceptron-multicamadas-f9de8471f1a9. Acesso em: 12 set. 2026.
* Iris **Archive.ics**. 2026. Disponível em: https://archive.ics.uci.edu/dataset/53/iris. Acesso em: 14 set. 2026.
