# tradutor-de-libras
Projeto de Reconhecimento de Sinais em Libras
Este projeto visa reconhecer sinais em Libras (Língua Brasileira de Sinais) a partir de vídeos utilizando captura de dados em tempo real, aprendizado de máquina e redes neurais. O projeto utiliza bibliotecas populares como MediaPipe, TensorFlow, OpenCV, NumPy, Matplotlib, entre outras.

Pré-requisitos
Certifique-se de ter instalado as seguintes bibliotecas em seu ambiente:


```
pip install mediapipe tensorflow 
pip install opencv
pip install numpy sklearn 
pip install matplotlib
pip install pandas
pip install warnings
```

Arquivos Principais
captura_dados.py: Script para capturar dados e preencher o arquivo dataset_libras.csv. Ele coleta as informações necessárias para o treinamento do modelo.
reconhecimento.py: Script para reconhecer os sinais a partir de vídeos. Este script utiliza o modelo treinado para classificar os sinais em Libras.
dataset_libras.csv: Arquivo onde os dados capturados serão armazenados. Ele contém as características dos sinais que serão usados para treinar o modelo.
Como Usar
1. Capturando Dados
Antes de treinar o modelo, você precisará capturar seus próprios dados de sinais, caso deseje. Para isso, siga os passos abaixo:

Apague o conteúdo do arquivo dataset_libras.csv, caso deseje iniciar um novo conjunto de dados.

2. Treinando o Modelo
Com os dados capturados no dataset_libras.csv, você está pronto para treinar o modelo de reconhecimento de sinais. O processo de treinamento criará classes de cada sinal capturado. Certifique-se de que o CSV esteja preenchido com uma quantidade suficiente de dados para garantir uma boa precisão no reconhecimento.

O treinamento é feito diretamente dentro do script captura_dados.py ou em um script separado que você pode adaptar conforme a necessidade.

3. Reconhecimento de Sinais
Após treinar o modelo, execute o script reconhecimento.py para iniciar o reconhecimento dos sinais a partir de vídeos. O script capturará frames do vídeo, aplicará o modelo treinado e fornecerá a classe correspondente para o sinal detectado.

bash
Copiar código
python reconhecimento.py
4. Testando com Novos Sinais
Se quiser adicionar novos sinais ao seu modelo, repita o processo de captura de dados e reentreine o modelo. Isso permite melhorar continuamente o desempenho e a abrangência do sistema.
