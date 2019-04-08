## Breve descrição dos modelos escolhidos:

### 1. Rede Neural Artificial Multi Layer Perceptron:

Trata-se de uma **rede neural artificial** de multiplas camadas, com arquitetura *feed-forward* sem retroalimentação. 

Essa rede caracteríza-se por uma camada de entrada, onde cada neurônio desta camada recebe o valor de uma feature. Nas camadas ocultas (configurada no parêmetro `ridem_layers`), ocorre a combinação desses parâmetros em uma função de ativação, cujo seu resultado é transmitido para a próxima camada. 

Neste tipo de rede neural, não há retroalimentação, ou seja, as camadas de entrada não recebem valores de resultados das camadas subsequêntes. 
