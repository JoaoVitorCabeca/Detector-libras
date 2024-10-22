import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Carregando o dataset
data = pd.read_csv('../data/dataset_libras.csv')

# Separa características (X) e rótulos (y)
X = data.iloc[:, 1:].values  # Todos os valores de landmarks
y = data.iloc[:, 0].values   # Os rótulos (sinais)

# Codifica os rótulos
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Salva as classes para uso posterior na inferência
np.save('classes.npy', label_encoder.classes_)

# Divide o dataset em treino e validação
X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Normalizando os dados (opcional, caso os valores não estejam entre 0 e 1)
# X_train = X_train / np.max(X_train)
# X_val = X_val / np.max(X_val)

# Criação do modelo com Dropout para regularização
model = tf.keras.models.Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),  # Dropout de 30% para evitar overfitting
    Dense(64, activation='relu'),
    Dropout(0.2),  # Dropout adicional para regularização
    Dense(len(label_encoder.classes_), activation='softmax')  # Camada de saída com softmax
])

# Compilando o modelo com Adam e uma taxa de aprendizado menor
model.compile(optimizer=Adam(learning_rate=0.0001),  # Taxa de aprendizado ajustada
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Callback de Early Stopping para parar o treino se não houver melhorias
early_stop = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)

# Treinamento do modelo com batch_size menor
history = model.fit(X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=2000,
                    batch_size=16,  # Batch size ajustado para melhorar na CPU
                    callbacks=[early_stop])

# Salvando o modelo treinado
model.save('../models/melhor_modelo_libras.keras')

# Salvando o histórico de treinamento para análise
np.save('../models/history.npy', history.history)

# Plotando a curva de aprendizado
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='Acurácia de Treino')
plt.plot(history.history['val_accuracy'], label='Acurácia de Validação')
plt.xlabel('Épocas')
plt.ylabel('Acurácia')
plt.legend(loc='lower right')
plt.title('Acurácia de Treino e Validação')
plt.show()

