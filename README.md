# creditas-challenge

## Desafio Creditas - Data Science

### 1. Objetivo

O objetivo da modelagem é predizer se, dado as características de um cliente que necessita de um empréstimo financeiro, este será enviado para análise de crédito. 

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
Breve descrição de todo processo de treinamento e predição. Refere-se aos códigos contidos na pasta `scripts`.
#### 3.1 Preparando os Dados

- Script `scripts/load_data.py`

Realiza a leitura dos dados, especificando o tipo de cada variável, e salva em um arquivo `picle` para ser tratado.

- Script `scripts/preprocessing.py`

Realiza o tratamento de dados. Nesta etapa o conjunto de dados é tratado, removendo variáveis que não são preditoras, realizando indentificação e tratamento de valores faltantes e outliers. Nesta fase, também é realizado a normalização dos dados.

Neste processo, foram excluídas variáveis que apresentam mais de 30% de entradas nulas. 

    1. Preenchimento de Valores Faltantes:

        - Para variáveis categóricas, foi preenchido com a label `missing`;

        - Para variáveis binárias, foi preenchido com o valor `-1`, com o objetivo de informar ao modelo que existe uma informação fantante, uma vez que há quantidades razoáveis desses valores e que poderiam estar relacionadas com o valor a ser predito;

        - Para variáveis numéricas, foram preenchidos com a `mediana`.
    
    2. Tratamento de Valores `outliers`:

        - Valores considerados outliers foram preenchidos com o valor médio da variável em questão. 
        
        - Considera-se um valor outlier aquele que diverge da média da variável em mais do que três desvios padrões.

- Script `feature_generation.py`

Realiza a criação de novas features. Aqui são aplicadas técnicas de binarização de features categóricas e geração de relações polinomiais de até segunda ordem para features numéricas.

    1. Variáveis Categóricas

    É feito uma representação,associando a cada categoria um numero binário que será transcrito em `n`colunas, sendo `n`o número de dígitos desta representação.

#### 3.2 Treinando o Modelo

Com os parâmetros definidos explicados no arquivo `evaluation.ipynb`. é realizado o treino do modelo com o uso da biblioteca `scikit-learn`, que contém a implementação do modelo escolhido. 


### 4. Como executar (Modelo Implementado)

A modelagem foi escrita utilizando linguagem Python3 (versão 3.7), utilizando o interpretador iPython3 e Jupyter Notebook para a avaliação dos modelos propostos.

#### 4.1. Dependências
Instale as bibliotecas python utilizadas com o comando:

```pip install -r requirements.txt```

#### 4.2 Visualizando o processo de avaliação e escolha dos modelos citados

Caso não possua o Jupyter instalado, [será necessário seguir os passos descritos neste link](https://jupyter.readthedocs.io/en/latest/install.html).


1. Dentro do diretório do projeto, iniciar uma instância do Jupyter Notebook por linha de comando. 

`jupyter-notebook .`

2. Utilize o navegador para visualizar a interface do Jupyter. Em geral, caso as configurações padrões não tenham sido alteradas e não existam outras execuções do Jupyter, a página ficará disponível em `localhost://8888`. 

Nesta interface, clique no arquivo com nome `evaluation.ipynb`

3. Utilize o comando 

`shift + enter`

para executar as células do notebook e, assim, visualizar o processo de validação do modelo. Caso queira apenas visualizar, sem executar, o notebook foi salvo com os resultados da ultima execução.

**Nota:** Na pasta `report` encontra-se uma cópia em formato PDF deste mesmo arquivo. 


#### 4.3 Treinando o Modelo

Para treinar o modelo, será necessário navegar até o diretório do projeto utilizando linhas de comando. 

O treinamento do modelo é feito com o comando

`python3 train_pipeline.py`

#### 4.4 Obtendo predições a partir de uma base no formato `.csv`

Para realizar predições com o modelo treinado anteriormente, será necessário navegar até o diretório do projeto utilizando linhas de comando. 

1. Preencher dados de entrada

Será necessário o preenchimento de um arquivo `input.csv`(com exatamente este nome), dentro do diretório `inputs`. Neste local já existe um arquivo preenchido que pode ser utilizado como exemplo. 

2. Realizando a predição

Executar o comando 

`python3 predict.py`

### 5. Como executar (Jupyter Notebook)

Será necessário ter instalado o interpretador `iPython` e o pacote `jupyter-notebook`. 

Um tutorial de instalação do Jupyter Notebook pode ser econttrado [neste link](https://www.datacamp.com/community/tutorials/tutorial-jupyter-notebook?utm_source=adwords_ppc&utm_campaignid=1455363063&utm_adgroupid=65083631748&utm_device=c&utm_keyword=&utm_matchtype=b&utm_network=g&utm_adpostion=1t1&utm_creative=278443377095&utm_targetid=aud-390929969673:dsa-473406581035&utm_loc_interest_ms=&utm_loc_physical_ms=1001773&gclid=CjwKCAjwp_zkBRBBEiwAndwD9WrCe_iarDgZMybUpj9SLeDjpBXTrovZQydNu1V0_g3oeUQmXcY1gBoCs2kQAvD_BwE).

**Observação:** Ao abrir os notebooks, mudar o interpretador python para o seu interpretador padrão. 

Recomenda-se utilizar `virtual environment`. Veja mais sobre *virtual environments* em Python [neste link](https://virtualenv.pypa.io/en/latest/installation/)