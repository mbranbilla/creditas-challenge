# creditas-challenge

## Desafio Creditas - Data Science

### 1. Objetivo

### 2. Descrições dos Arquivos

Os arquivos de modelagem dividem-se em um *pipeline* composto do modelo que apresentou a performance mais adequada ao problema proposto e consolidado em *scripits* Python, juntamente de um *Jpyter Notebook* contendo o processo de avaliação de diversos modelos propostos. 



#### 2.1 evaluation.ipynb

Notebook utilizado para avaliar os modelos selecionados. Os modelos selecionados para teste são:

1. Randon Forest;

2. Suport Vector Machines Classifier;

3. Gradient Boost Classifiers;

4. Multilayer Perceptron Classifier.

Sendo os modelos (1.) e (3.) compostos de classificadores baseados em árvores de decisão, (2.) baseado em máquinas de vetores de suporte e (4.) um tipo de rede neural artificial.

Encontra-se neste notebook a avaliação individual de cada modelo no mesmo conjunto de dados de treino e de teste. As seguintes métricas são utilizadas:

- Cross Validation Score;
- ROC e AUC;
- Confusion Matrix;
- Avaliação do desempenho do modelo em função da complexidade.

Um resumo sobre o funcionamento e a descrição sobre os métodos de avaliação desses modelos podem ser visualizadas no arquivo [do relatório deste desafio]('https://github.com/mbranbilla/creditas-challenge/report/report.pdf')


#### 2.2 pipeline.py

Script python que executa todo o pipeline de modelo, exceto as etapas de validação.

#### 2.3 utils.py

Script que contém todas as definições de funções utilizadas.



### 3. Pipeline do Modelo
#### 3.1 Preparando os Dados
#### 3.2 Treinando o Modelo
#### 3.3 Obtendo Métricas de Validação
#### 3.4 Realizando Predições


### 4. Como executar (Modelo Implementado)

A modelagem foi escrita utilizando linguagem Python3 (versão 3.7), utilizando o interpretador iPython3 e Jupyter Notebook para a avaliação dos modelos propostos.

#### 4.1. Dependências
Instale as bibliotecas python utilizadas com o comando:

```pip install -r requirements.txt```

#### 4.2 Visualizando o processo de avaliação e escolha dos modelos citados

#### 4.3 Treinando o Modelo

#### 4.4 Obtendo predições a partir de uma base no formato `.csv`

### 5. Como executar (Jupyter Notebook)

Será necessário ter instalado o interpretador `iPython` e o pacote `jupyter-notebook`. 

Um tutorial de instalação do Jupyter Notebook pode ser econttrado [neste link](https://www.datacamp.com/community/tutorials/tutorial-jupyter-notebook?utm_source=adwords_ppc&utm_campaignid=1455363063&utm_adgroupid=65083631748&utm_device=c&utm_keyword=&utm_matchtype=b&utm_network=g&utm_adpostion=1t1&utm_creative=278443377095&utm_targetid=aud-390929969673:dsa-473406581035&utm_loc_interest_ms=&utm_loc_physical_ms=1001773&gclid=CjwKCAjwp_zkBRBBEiwAndwD9WrCe_iarDgZMybUpj9SLeDjpBXTrovZQydNu1V0_g3oeUQmXcY1gBoCs2kQAvD_BwE).

**Observação:** Ao abrir os notebooks, mudar o interpretador python para o seu interpretador padrão. 

Recomenda-se utilizar `virtual environment`. Veja mais sobre *virtual environments* em Python [neste link](https://virtualenv.pypa.io/en/latest/installation/)