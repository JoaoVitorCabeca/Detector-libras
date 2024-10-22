import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import warnings

# Suprime avisos desnecessários
warnings.filterwarnings("ignore", category=UserWarning)

# Carrega o modelo treinado
model = tf.keras.models.load_model('../models/melhor_modelo_libras.keras')

# Carrega o Label Encoder para decodificar as classes
label_encoder = LabelEncoder()

try:
    label_encoder.classes_ = np.load('../models/classes.npy', allow_pickle=True)
    print("Classes carregadas:", label_encoder.classes_)
except FileNotFoundError:
    print("Arquivo 'classes.npy' não encontrado. Certifique-se de que o arquivo está no diretório correto.")
    label_encoder.classes_ = np.array([])

# Inicializa a detecção de mãos do Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Captura de vídeo da webcam
cap = cv2.VideoCapture(0)

# Função para fazer predições do modelo com as coordenadas da mão
def predict_sign(hand_landmarks):
    landmarks = []
    for lm in hand_landmarks.landmark:
        landmarks.extend([lm.x, lm.y, lm.z])
    landmarks = np.array(landmarks).reshape(1, -1)

    print("Dimensão dos landmarks:", landmarks.shape)  # Depuração da dimensão dos landmarks

    # Faz a previsão do modelo
    prediction = model.predict(landmarks)
    print("Predição bruta:", prediction)  # Depuração da predição bruta
    predicted_class = np.argmax(prediction, axis=1)

    if not label_encoder.classes_.size:
        return ["Classe não disponível"]

    print("Classe prevista:", label_encoder.inverse_transform(predicted_class))  # Depuração da classe prevista
    return label_encoder.inverse_transform(predicted_class)

# Loop para capturar a imagem da webcam
while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Frame vazio.")
        continue

    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Desenha landmarks na imagem
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Prediz o sinal de LIBRAS
            predicted_sign = predict_sign(hand_landmarks)

            # Exibe o sinal previsto na tela
            cv2.putText(image, f'Sinal: {predicted_sign[0]}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2,
                        cv2.LINE_AA)

    cv2.imshow('Reconhecimento de LIBRAS', image)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
