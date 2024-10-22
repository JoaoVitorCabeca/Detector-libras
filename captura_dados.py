import cv2
import mediapipe as mp
import csv

# Inicializa a detecção de mãos do Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Captura de vídeo da webcam
cap = cv2.VideoCapture(0)

# Dicionário de rótulos (mapeia teclas para sinais)
labels = {
    ord('1'): 'Paz',  # Tecla 1 -> 'sinal_1'
    ord('2'): 'OK',  # Tecla 2 -> 'sinal_2'
    ord('3'): 'Meio',  # Tecla 3 -> 'sinal_3'
    ord('4'): 'Faz o L',
    ord('5'): 'H',
    ord('6'): 'I',
    ord('7'): 'J',
    ord('8'): 'M',
    ord('9'): 'N',
    ord('0'): 'O',
    # Adicione mais sinais conforme necessário
}

# Função para salvar landmarks em CSV
def salvar_landmarks_csv(hand_landmarks, label):
    with open('../data/dataset_libras.csv', mode='a', newline='') as f:
        writer = csv.writer(f)
        row = [label]
        for lm in hand_landmarks.landmark:
            row.extend([lm.x, lm.y, lm.z])  # Coordenadas x, y, z de cada landmark
        writer.writerow(row)

# Loop de captura
while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Frame vazio.")
        continue

    image = cv2.flip(image, 1)  # Espelha a imagem
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Desenha os landmarks da mão na imagem
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Verifica se alguma tecla foi pressionada
            key = cv2.waitKey(1) & 0xFF  # Lê a tecla pressionada
            if key in labels:
                sinal = labels[key]
                salvar_landmarks_csv(hand_landmarks, sinal)
                print(f'Landmarks salvos para o sinal: {sinal}')

    # Exibe a imagem na janela
    cv2.imshow('Coletor de Dados de LIBRAS', image)

    # Encerra o loop se a tecla 'q' for pressionada
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
